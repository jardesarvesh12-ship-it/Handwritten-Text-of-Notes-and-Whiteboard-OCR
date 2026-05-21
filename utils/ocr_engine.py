"""
utils/ocr_engine.py
Multi-engine OCR manager.
Supports: Tesseract, EasyOCR, and TrOCR (transformer-based handwriting).
Falls back gracefully when a dependency is unavailable.
"""

import os
import re
import time
import logging
from typing import Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Engine wrappers                                                     #
# ------------------------------------------------------------------ #

class TesseractEngine:
    """Wrapper around pytesseract."""

    def __init__(self):
        import pytesseract
        self.pytesseract = pytesseract
        # Allow override via env var (useful on Windows)
        custom_path = os.getenv("TESSERACT_CMD")
        if custom_path:
            pytesseract.pytesseract.tesseract_cmd = custom_path
        self.available = self._check()

    def _check(self) -> bool:
        try:
            self.pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False

    def extract(self, image: Image.Image, mode: str = "auto") -> dict:
        if not self.available:
            return {"text": "", "confidence": 0, "engine": "tesseract", "error": "not installed"}

        # PSM mode mapping
        psm_map = {
            "auto": 3,
            "single_block": 6,
            "single_line": 7,
            "whiteboard": 3,
            "handwritten": 6,
            "flowchart": 11,  # sparse text
        }
        psm = psm_map.get(mode, 3)
        config = f"--oem 3 --psm {psm}"

        try:
            data = self.pytesseract.image_to_data(
                image, config=config,
                output_type=self.pytesseract.Output.DICT
            )
            words = [w for w, c in zip(data["text"], data["conf"])
                     if str(w).strip() and int(c) > 0]
            confs = [int(c) for c in data["conf"]
                     if int(c) > 0]
            text = self.pytesseract.image_to_string(image, config=config).strip()
            avg_conf = int(np.mean(confs)) if confs else 0
            return {
                "text": text,
                "confidence": avg_conf,
                "word_count": len(words),
                "engine": "tesseract",
            }
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return {"text": "", "confidence": 0, "engine": "tesseract", "error": str(e)}


class TextractEngine:
    """Wrapper around AWS Textract."""

    def __init__(self):
        self.available = False
        try:
            import boto3
            self.boto3 = boto3
            # Check for credentials in env
            self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            
            if self.aws_access_key and self.aws_secret_key:
                self.available = self._check()
            else:
                logger.warning("AWS credentials not found in environment. TextractEngine disabled.")
        except ImportError:
            logger.warning("boto3 not installed. TextractEngine disabled.")
        except Exception as e:
            logger.warning(f"TextractEngine initialization error: {e}")

    def _check(self) -> bool:
        try:
            # Initialize boto3 client to test environment variables
            self.boto3.client(
                "textract",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            return True
        except Exception as e:
            logger.warning(f"AWS Textract initialization check failed: {e}")
            return False

    def extract(self, image: Image.Image, mode: str = "auto") -> dict:
        if not self.available:
            return {"text": "", "confidence": 0, "engine": "textract", "error": "not configured or credentials missing"}

        import io
        try:
            # Convert PIL image to JPEG format bytes
            img_byte_arr = io.BytesIO()
            image.convert("RGB").save(img_byte_arr, format="JPEG", quality=95)
            img_bytes = img_byte_arr.getvalue()

            # Create the client and invoke the API
            client = self.boto3.client(
                "textract",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )

            response = client.detect_document_text(Document={"Bytes": img_bytes})

            # Parse lines and confidence scores
            lines = []
            confs = []
            
            for block in response.get("Blocks", []):
                if block.get("BlockType") == "LINE":
                    text = block.get("Text", "").strip()
                    conf = block.get("Confidence", 0)  # Float from 0-100
                    if text:
                        lines.append(text)
                        confs.append(conf)

            full_text = "\n".join(lines)
            avg_conf = int(np.mean(confs)) if confs else 0
            
            return {
                "text": full_text,
                "confidence": avg_conf,
                "word_count": len(full_text.split()),
                "engine": "textract",
            }
        except Exception as e:
            logger.error(f"AWS Textract error: {e}")
            return {"text": "", "confidence": 0, "engine": "textract", "error": str(e)}


class EasyOCREngine:
    """Wrapper around EasyOCR."""

    def __init__(self, languages: list = None):
        self.languages = languages or ["en"]
        self.reader = None
        self.available = False
        self._init()

    def _init(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self._gpu_available(),
                verbose=False
            )
            self.available = True
            logger.info("EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

    def _gpu_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def extract(self, image: Image.Image, mode: str = "auto") -> dict:
        if not self.available:
            return {"text": "", "confidence": 0, "engine": "easyocr", "error": "not installed"}
        try:
            img_arr = np.array(image.convert("RGB"))
            results = self.reader.readtext(img_arr, detail=1, paragraph=False)
            
            # Sort boxes top-to-bottom
            results.sort(key=lambda item: item[0][0][1])
            
            # Group into lines and sort left-to-right
            lines_grouped = []
            for item in results:
                if not lines_grouped:
                    lines_grouped.append([item])
                else:
                    last_line = lines_grouped[-1]
                    avg_top = sum(b[0][0][1] for b in last_line) / len(last_line)
                    avg_bot = sum(b[0][2][1] for b in last_line) / len(last_line)
                    center_y = (item[0][0][1] + item[0][2][1]) / 2
                    if avg_top - 5 <= center_y <= avg_bot + 5:
                        last_line.append(item)
                    else:
                        lines_grouped.append([item])
            
            full_text_lines = []
            confs = []
            bboxes = []
            for line in lines_grouped:
                line.sort(key=lambda item: item[0][0][0])
                line_texts = []
                for bbox, text, conf in line:
                    if text.strip():
                        line_texts.append(text.strip())
                        confs.append(conf)
                        bboxes.append(bbox)
                if line_texts:
                    full_text_lines.append(" ".join(line_texts))
                    
            full_text = "\n".join(full_text_lines)
            avg_conf = int(np.mean(confs) * 100) if confs else 0
            return {
                "text": full_text,
                "confidence": avg_conf,
                "word_count": len(full_text.split()),
                "bboxes": bboxes,
                "engine": "easyocr",
            }
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return {"text": "", "confidence": 0, "engine": "easyocr", "error": str(e)}


class TrOCREngine:
    """
    Microsoft TrOCR – transformer-based handwriting recognition.
    Uses trocr-base-handwritten for best handwriting results.
    """

    MODEL_ID = "microsoft/trocr-base-handwritten"

    def __init__(self):
        self.processor = None
        self.model = None
        self.available = False
        self._init()

    def _init(self):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            logger.info(f"TrOCR loaded on {self.device}")
        except Exception as e:
            logger.warning(f"TrOCR not available: {e}")

    def _split_into_lines(self, image: Image.Image) -> list:
        """Very simple horizontal-strip splitting to feed TrOCR line by line."""
        import cv2
        gray = np.array(image.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_sums = binary.sum(axis=1)
        threshold = row_sums.max() * 0.05
        in_line = False
        lines = []
        start = 0
        for i, s in enumerate(row_sums):
            if s > threshold and not in_line:
                in_line = True
                start = i
            elif s <= threshold and in_line:
                in_line = False
                if i - start > 5:
                    lines.append((start, i))
        if in_line:
            lines.append((start, len(row_sums)))

        # Merge very close lines
        merged = []
        for s, e in lines:
            if merged and s - merged[-1][1] < 8:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        strips = []
        w = image.width
        for s, e in merged:
            pad = 4
            strip = image.crop((0, max(0, s - pad), w, min(image.height, e + pad)))
            strips.append(strip)
        return strips if strips else [image]

    def extract(self, image: Image.Image, mode: str = "auto") -> dict:
        if not self.available:
            return {"text": "", "confidence": 0, "engine": "trocr", "error": "not installed"}
        try:
            import torch
            strips = self._split_into_lines(image)
            texts = []
            for strip in strips[:40]:  # cap at 40 lines
                strip_rgb = strip.convert("RGB")
                pixel_values = self.processor(
                    images=strip_rgb, return_tensors="pt"
                ).pixel_values.to(self.device)
                with torch.no_grad():
                    ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if text:
                    texts.append(text)
            full_text = "\n".join(texts)
            return {
                "text": full_text,
                "confidence": 85 if full_text else 0,
                "word_count": len(full_text.split()),
                "engine": "trocr",
            }
        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return {"text": "", "confidence": 0, "engine": "trocr", "error": str(e)}


# ------------------------------------------------------------------ #
#  OCR Manager                                                         #
# ------------------------------------------------------------------ #

class OCRManager:
    """
    Orchestrates multiple OCR engines and merges/selects best result.
    Strategy: run selected engines, pick highest-confidence non-empty result.
    """

    def __init__(self, engines: list = None, trocr_enabled: bool = False):
        """
        Args:
            engines: list of engine names to load ('tesseract', 'easyocr', 'trocr', 'textract')
            trocr_enabled: whether to load the heavy TrOCR model (slow first load)
        """
        self.engines = {}
        requested = engines or ["easyocr", "tesseract", "textract"]

        if "tesseract" in requested:
            self.engines["tesseract"] = TesseractEngine()
        if "easyocr" in requested:
            self.engines["easyocr"] = EasyOCREngine()
        if "trocr" in requested or trocr_enabled:
            self.engines["trocr"] = TrOCREngine()
        if "textract" in requested:
            self.engines["textract"] = TextractEngine()

        available = [k for k, v in self.engines.items() if getattr(v, "available", False)]
        logger.info(f"OCR engines available: {available}")

    def extract(self, image: Image.Image,
                engine: str = "auto",
                mode: str = "auto") -> dict:
        """
        Extract text from image.

        Args:
            image: PIL Image (already preprocessed)
            engine: 'auto' | 'tesseract' | 'easyocr' | 'trocr' | 'textract'
            mode: hint for engine config ('whiteboard','handwritten','flowchart','auto')

        Returns:
            dict with text, confidence, engine, metadata
        """
        t0 = time.time()

        if engine == "auto":
            results = self._run_all(image, mode)
            best = self._pick_best(results)
        elif engine in self.engines:
            best = self.engines[engine].extract(image, mode)
            results = {engine: best}
        else:
            best = {"text": f"Engine '{engine}' not loaded.", "confidence": 0}
            results = {}

        best["processing_time_ms"] = int((time.time() - t0) * 1000)
        best["all_results"] = results
        best["text"] = self._postprocess(best.get("text", ""))
        return best

    def _run_all(self, image: Image.Image, mode: str) -> dict:
        results = {}
        for name, eng in self.engines.items():
            if getattr(eng, "available", False):
                try:
                    results[name] = eng.extract(image, mode)
                except Exception as e:
                    results[name] = {"text": "", "confidence": 0, "error": str(e)}
        return results

    def _pick_best(self, results: dict) -> dict:
        best = {"text": "", "confidence": 0, "engine": "none"}
        for name, r in results.items():
            text = r.get("text", "").strip()
            conf = r.get("confidence", 0)
            # Score: confidence × sqrt(word_count)  (prefer longer high-conf)
            wc = len(text.split()) if text else 0
            score = conf * (wc ** 0.5)
            best_wc = len(best.get("text", "").split())
            best_score = best.get("confidence", 0) * (best_wc ** 0.5)
            if score > best_score and text:
                best = r
        return best

    def _postprocess(self, text: str) -> str:
        """Clean up common OCR artifacts."""
        if not text:
            return text
        # Remove null bytes
        text = text.replace("\x00", "")
        # Collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common l/1 and O/0 confusion in numbers
        text = re.sub(r"(?<=\d)l(?=\d)", "1", text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()

    @property
    def available_engines(self) -> list:
        return [k for k, v in self.engines.items() if getattr(v, "available", False)]
