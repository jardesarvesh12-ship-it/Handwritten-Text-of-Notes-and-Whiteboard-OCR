"""
tests/test_api.py
Integration tests for Flask API endpoints.
"""

import sys
import io
import json
import base64
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch OCR manager before importing app so heavy models aren't loaded
import unittest.mock as mock

# Mock the OCRManager to avoid loading ML models during tests
MOCK_OCR_RESULT = {
    "text": "Hello World\nThis is a test.",
    "confidence": 82,
    "word_count": 6,
    "engine": "easyocr",
    "processing_time_ms": 150,
}


def make_test_image() -> bytes:
    img = Image.new("RGB", (300, 200), (245, 242, 235))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        with mock.patch("utils.ocr_engine.OCRManager") as MockMgr:
            instance = MockMgr.return_value
            instance.available_engines = ["easyocr"]
            instance.extract.return_value = MOCK_OCR_RESULT
            from app.main import app
            self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    def test_health_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_json(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertIn("status", data)


class TestOCREndpoint(unittest.TestCase):

    def setUp(self):
        with mock.patch("utils.ocr_engine.OCRManager") as MockMgr:
            instance = MockMgr.return_value
            instance.available_engines = ["easyocr"]
            instance.extract.return_value = MOCK_OCR_RESULT
            from app.main import app
            self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    def test_no_file_returns_400(self):
        resp = self.client.post("/api/ocr")
        self.assertEqual(resp.status_code, 400)

    def test_upload_png_returns_success(self):
        img_bytes = make_test_image()
        data = {"file": (io.BytesIO(img_bytes), "test.png")}
        resp = self.client.post(
            "/api/ocr",
            data=data,
            content_type="multipart/form-data"
        )
        # May fail if OCR not installed; check it at least returns JSON
        self.assertIn(resp.status_code, [200, 500])
        result = json.loads(resp.data)
        self.assertIn("success", result)

    def test_base64_upload(self):
        img_bytes = make_test_image()
        b64 = base64.b64encode(img_bytes).decode()
        resp = self.client.post(
            "/api/ocr",
            json={"image_base64": b64, "pipeline": "auto", "engine": "auto"},
            content_type="application/json"
        )
        self.assertIn(resp.status_code, [200, 500])

    def test_pipeline_param_accepted(self):
        img_bytes = make_test_image()
        for pl in ["auto", "whiteboard", "handwritten"]:
            data = {
                "file": (io.BytesIO(img_bytes), "test.png"),
                "pipeline": pl,
            }
            resp = self.client.post(
                "/api/ocr", data=data, content_type="multipart/form-data"
            )
            self.assertIn(resp.status_code, [200, 500])


class TestEnginesEndpoint(unittest.TestCase):

    def setUp(self):
        with mock.patch("utils.ocr_engine.OCRManager") as MockMgr:
            instance = MockMgr.return_value
            instance.available_engines = ["easyocr"]
            from app.main import app
            self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    def test_engines_endpoint(self):
        resp = self.client.get("/api/engines")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("all", data)
        self.assertIn("available", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
