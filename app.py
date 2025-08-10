import os
import io
import json
import base64
import hashlib
import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from datetime import datetime
import sqlite3
import pandas as pd
from pathlib import Path
import re
from typing import Any, List, Dict, Optional, Tuple
from functools import lru_cache

# -------------------------------
# Config
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision-capable

# Persistence paths
CSV_PATH = Path("detections_log.csv")
SQLITE_PATH = Path("detections_log.db")
MAX_BASE64_BYTES = 4 * 1024 * 1024  # 4 MB

# -------------------------------
# Lazy-loaded resources
# -------------------------------
@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("Please set GROQ_API_KEY in your .env file")
    return Groq(api_key=GROQ_API_KEY)

@st.cache_data
def load_object_categories():
    """Load categories from JSON file with caching"""
    categories_file = Path("categories.json")
    if not categories_file.exists():
        # Fallback to default categories if file doesn't exist
        return [
            "car", "truck", "person", "dog", "cat", 
            "chair", "table", "tv", "laptop", "house"
        ]
    with open(categories_file) as f:
        return json.load(f)

@st.cache_resource
def get_db_connection():
    """Initialize SQLite database with thread-local storage"""
    con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_filename TEXT,
        label TEXT,
        confidence REAL,
        x_min REAL,
        y_min REAL,
        x_max REAL,
        y_max REAL
    )
    """)
    con.commit()
    return con

# -------------------------------
# Pydantic models
# -------------------------------
class Detection(BaseModel):
    label: str
    confidence: float = 1.0
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class ImageDetections(BaseModel):
    filename: str
    detections: List[Detection]

# -------------------------------
# Cached helpers
# -------------------------------
@lru_cache(maxsize=128)
def label_to_color(label: str) -> Tuple[int, int, int]:
    """Generate consistent color for each label using hash."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))

@st.cache_data(max_entries=20)
def compress_to_under_limit(image_bytes: bytes, max_bytes: int = MAX_BASE64_BYTES) -> bytes:
    """Compress image to fit under size limit with caching."""
    if len(image_bytes) <= max_bytes:
        return image_bytes
    
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    
    h, w = img.shape[:2]
    quality_levels = [90, 80, 70, 60, 50, 40, 30]
    scale = 0.95
    
    for _ in range(10):
        w = int(w * scale)
        h = int(h * scale)
        if w < 32 or h < 32:
            break
        
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        for q in quality_levels:
            success, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not success:
                continue
            data = buf.tobytes()
            if len(data) <= max_bytes:
                return data
        scale *= 0.9
    
    success, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    return buf.tobytes() if success else image_bytes

@st.cache_data(max_entries=20)
def draw_boxes_on_bytes(image_bytes: bytes, _detections: List[Detection], 
                       per_label_thresholds: Dict[str, float], global_threshold: float) -> bytes:
    """Draw bounding boxes on image bytes with caching."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image bytes for drawing")

    h, w = img.shape[:2]
    for det in _detections:
        label = det.label
        thresh = per_label_thresholds.get(label, global_threshold)
        if det.confidence < thresh:
            continue
        
        x1 = int(max(0, min(w - 1, det.x_min * w)))
        y1 = int(max(0, min(h - 1, det.y_min * h)))
        x2 = int(max(0, min(w - 1, det.x_max * w)))
        y2 = int(max(0, min(h - 1, det.y_max * h)))
        color = label_to_color(det.label)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        text = f"{det.label} {det.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 6), (x1 + text_width, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    success, buf = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError("Failed to encode annotated image")
    return buf.tobytes()

# -------------------------------
# Core processing functions
# -------------------------------
def generate_bounding_boxes_groq_batch(image_files: List, selected_objects: List[str], 
                                     custom_criteria: str = "") -> List[ImageDetections]:
    """Generate bounding boxes for multiple images using Groq's vision model."""
    if not image_files or len(image_files) > 5:
        return []

    prompt = f"""You are an advanced object detection assistant. Analyze the provided images and output detection results in strict JSON format.

Task Requirements:
1. Detect ONLY these objects: {', '.join(selected_objects)}
2. Apply these additional criteria: {custom_criteria or 'None'}
3. For each image, provide bounding boxes with:
   - Accurate normalized coordinates (0-1 range)
   - Confidence scores (0-1)
   - Precise object labels

Output Format Rules:
- Return a SINGLE JSON array where each element corresponds to an input image
- Each image object MUST contain:
  - "filename": Original filename (exactly as provided)
  - "detections": Array of detection objects with:
    - "label": String (must match selected objects)
    - "confidence": Float (0-1)
    - "x_min", "y_min", "x_max", "y_max": Floats (0-1)

Example of VALID output:
[
  {{
    "filename": "image1.jpg",
    "detections": [
      {{
        "label": "car",
        "confidence": 0.85,
        "x_min": 0.25,
        "y_min": 0.3,
        "x_max": 0.75,
        "y_max": 0.9
      }}
    ]
  }}
]"""

    content = [{"type": "text", "text": prompt}]
    prepared_images = []
    
    for f in image_files:
        if isinstance(f, str) and (f.startswith("http://") or f.startswith("https://")):
            filename = os.path.basename(f)
            content.append({"type": "image_url", "image_url": {"url": f}})
            prepared_images.append((filename, None, None))
        else:
            filename = f.name if hasattr(f, "name") else os.path.basename(f)
            raw = f.read() if hasattr(f, "read") else open(f, "rb").read()
            safe = compress_to_under_limit(raw, MAX_BASE64_BYTES)
            b64 = base64.b64encode(safe).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            prepared_images.append((filename, safe, b64))

    try:
        groq_client = get_groq_client()
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        raw_output = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {str(e)}")

    try:
        parsed = json.loads(raw_output)
        if not isinstance(parsed, list):
            if isinstance(parsed, dict) and "detections" in parsed:
                parsed = [parsed]
            else:
                raise RuntimeError("Expected JSON array but got different structure")
    except Exception as e:
        raise RuntimeError(f"JSON parsing failed: {str(e)}\nRaw output:\n{raw_output[:500]}...")

    results = []
    for i, (filename, _, _) in enumerate(prepared_images):
        try:
            item = parsed[i] if i < len(parsed) else {}
            detections = []
            
            for d in item.get("detections", []):
                try:
                    detections.append(Detection(
                        label=str(d.get("label", "")),
                        confidence=float(d.get("confidence", 0.5)),
                        x_min=float(d.get("x_min", 0.0)),
                        y_min=float(d.get("y_min", 0.0)),
                        x_max=float(d.get("x_max", 1.0)),
                        y_max=float(d.get("y_max", 1.0))
                    ))
                except (ValueError, TypeError) as ve:
                    st.warning(f"Invalid detection format in {filename}: {str(ve)}")
                    continue
            
            results.append(ImageDetections(filename=filename, detections=detections))
        except Exception as e:
            st.warning(f"Error processing image {filename}: {str(e)}")
            results.append(ImageDetections(filename=filename, detections=[]))

    return results

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="EarthMinds Lens", layout="wide")
    st.title("👁️ EarthMinds Lens Object Detection")

    # Load categories only when needed
    object_categories = load_object_categories()

    # Sidebar controls
    st.sidebar.header("Detection Thresholds")
    global_threshold = st.sidebar.slider("Global default threshold", 0.0, 1.0, 0.5, 0.01)

    label_selector = st.sidebar.multiselect(
        "Pick labels to set custom thresholds for", 
        object_categories, 
        default=["person", "car"]
    )
    
    per_label_thresholds = {}
    for lbl in label_selector:
        per_label_thresholds[lbl] = st.sidebar.slider(
            f"Threshold for '{lbl}'", 
            0.0, 1.0, 0.5, 0.01
        )

    st.sidebar.markdown("---")
    st.sidebar.write("Persistence:")
    save_csv = st.sidebar.checkbox("Save to FileSystem", value=True)
    save_sqlite = st.sidebar.checkbox("Save to DB Vector", value=True)

    # Main UI
    uploaded_files = st.file_uploader(
        "📷 Upload Images", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True
    )
    
    selected_objects = st.multiselect(
        "Select objects to detect", 
        object_categories, 
        default=["person", "car"]
    )
    
    custom_criteria = st.text_input("Additional criteria (optional)", 
                                  placeholder="e.g., 'only red cars' or 'facing left'")

    if uploaded_files and st.button("🔍 earthminds lens"):
        if len(uploaded_files) > 5:
            st.error("Please upload at most 5 images.")
        else:
            try:
                with st.spinner("Detecting objects..."):
                    results = generate_bounding_boxes_groq_batch(uploaded_files, selected_objects, custom_criteria)

                # Persist results if enabled
                if save_csv or save_sqlite:
                    try:
                        rows = []
                        for imgdet in results:
                            for d in imgdet.detections:
                                rows.append({
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "image_filename": imgdet.filename,
                                    "label": d.label,
                                    "confidence": d.confidence,
                                    "x_min": d.x_min,
                                    "y_min": d.y_min,
                                    "x_max": d.x_max,
                                    "y_max": d.y_max
                                })
                        
                        if rows:
                            df = pd.DataFrame(rows)
                            if save_csv:
                                df.to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
                            if save_sqlite:
                                con = get_db_connection()
                                df.to_sql("detections", con, if_exists="append", index=False)
                                con.commit()
                        
                        st.success("Results saved successfully")
                    except Exception as e:
                        st.error(f"Failed to save results: {str(e)}")

                # Display results
                for res in results:
                    st.markdown(f"### {res.filename}")
                    orig_file = next((f for f in uploaded_files if f.name == res.filename), None)
                    
                    if orig_file:
                        orig_file.seek(0)
                        raw_bytes = orig_file.read()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(raw_bytes, caption=f"Original: {res.filename}", use_container_width=True)
                        
                        with col2:
                            try:
                                annotated = draw_boxes_on_bytes(raw_bytes, res.detections, 
                                                              per_label_thresholds, global_threshold)
                                st.image(annotated, caption=f"Annotated: {res.filename}", use_container_width=True)
                            except Exception as draw_e:
                                st.error(f"Failed to annotate image: {str(draw_e)}")
                    
                    # Show detections
                    with st.expander("Detection Details"):
                        st.json([d.dict() for d in res.detections])

            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
                st.exception(e)

    st.sidebar.markdown("---")
    st.sidebar.header("Detection History")
    st.info("Per-label thresholds let you tune detection sensitivity. Results are saved for analytics.")

if __name__ == "__main__":
    main()