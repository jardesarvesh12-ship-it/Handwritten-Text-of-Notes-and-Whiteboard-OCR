"""
app/main.py
Flask application for Handwritten Text & Whiteboard OCR.
"""

import os
import io
import uuid
import logging
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocessor import ImagePreprocessor, image_to_base64
from utils.ocr_engine import OCRManager

# Load env variables explicitly
from dotenv import load_dotenv
load_dotenv()

from app.models import db, OCRResult

# ------------------------------------------------------------------ #
#  App setup                                                           #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI", "sqlite:///ocr_database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app)

# Initialize Database
db.init_app(app)
with app.app_context():
    db.create_all()

# Lazy-initialize heavy models
preprocessor = ImagePreprocessor()
ocr_manager = None


def get_ocr_manager() -> OCRManager:
    global ocr_manager
    if ocr_manager is None:
        ocr_manager = OCRManager(engines=["easyocr", "tesseract"])
    return ocr_manager


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_from_request(req) -> tuple[Image.Image, str]:
    """Load image from multipart upload or base64 JSON body."""
    if "file" in req.files:
        f = req.files["file"]
        if not f.filename:
            raise ValueError("No file selected")
        if not allowed_file(f.filename):
            raise ValueError(f"File type not allowed: {f.filename}")
        img_bytes = f.read()
        filename = secure_filename(f.filename)
    elif req.is_json and "image_base64" in req.json:
        b64 = req.json["image_base64"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        filename = req.json.get("filename", "upload.png")
    else:
        raise ValueError("No image provided. Send multipart 'file' or JSON 'image_base64'.")

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image, filename


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    mgr = get_ocr_manager()
    return jsonify({
        "status": "ok",
        "available_engines": mgr.available_engines,
    })


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    """
    Main OCR endpoint.

    Form fields / JSON keys:
      file          - image file (multipart) OR
      image_base64  - base64 encoded image (JSON)
      pipeline      - preprocessing pipeline: auto|whiteboard|handwritten|flowchart|document
      engine        - OCR engine: auto|tesseract|easyocr|trocr
      return_preview- boolean, return preprocessed image as base64
    """
    try:
        image, filename = load_image_from_request(request)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Image load error")
        return jsonify({"success": False, "error": "Failed to read image"}), 400

    # Options
    pipeline = (request.form.get("pipeline") or
                (request.json or {}).get("pipeline", "auto"))
    engine = (request.form.get("engine") or
              (request.json or {}).get("engine", "auto"))
    return_preview = str(
        request.form.get("return_preview") or
        (request.json or {}).get("return_preview", "true")
    ).lower() == "true"

    # --- Preprocess ---
    try:
        pre_result = preprocessor.preprocess(image, pipeline=pipeline)
        processed_img = pre_result["image"]
        pipeline_used = pre_result["pipeline_used"]
        steps = pre_result["steps"]
    except Exception as e:
        logger.exception("Preprocessing error")
        processed_img = image
        pipeline_used = "none"
        steps = [{"step": 1, "message": f"Preprocessing failed: {e}"}]

    # --- OCR ---
    try:
        mgr = get_ocr_manager()
        ocr_result = mgr.extract(processed_img, engine=engine, mode=pipeline_used)
    except Exception as e:
        logger.exception("OCR error")
        return jsonify({"success": False, "error": f"OCR failed: {e}"}), 500

    # --- Build response ---
    response = {
        "success": True,
        "filename": filename,
        "pipeline": pipeline_used,
        "engine_used": ocr_result.get("engine", "unknown"),
        "text": ocr_result.get("text", ""),
        "confidence": ocr_result.get("confidence", 0),
        "word_count": ocr_result.get("word_count", len(ocr_result.get("text", "").split())),
        "processing_time_ms": ocr_result.get("processing_time_ms", 0),
        "preprocessing_steps": steps,
        "available_engines": mgr.available_engines,
    }

    # --- Save to DB ---
    try:
        new_record = OCRResult(
            filename=filename,
            extracted_text=response["text"],
            confidence=response["confidence"],
            engine_used=response["engine_used"],
            pipeline=response["pipeline"]
        )
        db.session.add(new_record)
        db.session.commit()
    except Exception as e:
        logger.exception("Failed to save to database")

    if return_preview:
        response["original_preview"] = "data:image/png;base64," + image_to_base64(
            image.resize(_thumb_size(image), Image.LANCZOS)
        )
        response["processed_preview"] = "data:image/png;base64," + image_to_base64(
            processed_img.resize(_thumb_size(processed_img), Image.LANCZOS)
        )

    return jsonify(response)


@app.route("/api/engines")
def list_engines():
    mgr = get_ocr_manager()
    return jsonify({
        "available": mgr.available_engines,
        "all": ["tesseract", "easyocr", "trocr"],
    })


@app.route("/api/history")
def get_history():
    try:
        results = OCRResult.query.order_by(OCRResult.created_at.desc()).limit(100).all()
        return jsonify({
            "success": True,
            "count": len(results),
            "history": [r.to_dict() for r in results]
        })
    except Exception as e:
        logger.exception("Failed to fetch history")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _thumb_size(img: Image.Image, max_dim: int = 800):
    w, h = img.size
    if max(w, h) <= max_dim:
        return w, h
    scale = max_dim / max(w, h)
    return int(w * scale), int(h * scale)


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting OCR server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
