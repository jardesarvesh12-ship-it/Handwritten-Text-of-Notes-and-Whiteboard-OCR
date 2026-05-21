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

import unittest.mock as mock

# Mock the OCRResult structure to prevent DB issues during mock tests
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
        from app.main import app
        self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True
        
        # Intercept get_ocr_manager cleanly to return mock
        self.patcher = mock.patch("app.main.get_ocr_manager")
        self.mock_get = self.patcher.start()
        self.mock_mgr = mock.MagicMock()
        self.mock_mgr.available_engines = ["easyocr", "tesseract", "textract"]
        self.mock_mgr.extract.return_value = MOCK_OCR_RESULT
        self.mock_get.return_value = self.mock_mgr

    def tearDown(self):
        self.patcher.stop()

    def test_health_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_json(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertIn("status", data)


class TestOCREndpoint(unittest.TestCase):

    def setUp(self):
        from app.main import app
        self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True
        
        # Intercept get_ocr_manager cleanly to return mock
        self.patcher = mock.patch("app.main.get_ocr_manager")
        self.mock_get = self.patcher.start()
        self.mock_mgr = mock.MagicMock()
        self.mock_mgr.available_engines = ["easyocr", "tesseract", "textract"]
        self.mock_mgr.extract.return_value = MOCK_OCR_RESULT
        self.mock_get.return_value = self.mock_mgr

    def tearDown(self):
        self.patcher.stop()

    def test_no_file_returns_400(self):
        resp = self.client.post("/api/ocr")
        self.assertEqual(resp.status_code, 400)

    def test_upload_png_returns_success(self):
        img_bytes = make_test_image()
        data = {"file": (io.BytesIO(img_bytes), "test.png")}
        resp = self.client.post(
            "/api/ocr",
            data=data
        )
        self.assertEqual(resp.status_code, 200)
        result = json.loads(resp.data)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], MOCK_OCR_RESULT["text"])

    def test_base64_upload(self):
        img_bytes = make_test_image()
        b64 = base64.b64encode(img_bytes).decode()
        resp = self.client.post(
            "/api/ocr",
            json={"image_base64": b64, "pipeline": "auto", "engine": "auto"},
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        result = json.loads(resp.data)
        self.assertTrue(result["success"])

    def test_pipeline_param_accepted(self):
        img_bytes = make_test_image()
        for pl in ["auto", "whiteboard", "handwritten"]:
            data = {
                "file": (io.BytesIO(img_bytes), "test.png"),
                "pipeline": pl,
            }
            resp = self.client.post(
                "/api/ocr", data=data
            )
            self.assertEqual(resp.status_code, 200)


class TestEnginesEndpoint(unittest.TestCase):

    def setUp(self):
        from app.main import app
        self.app = app
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True
        
        # Intercept get_ocr_manager cleanly to return mock
        self.patcher = mock.patch("app.main.get_ocr_manager")
        self.mock_get = self.patcher.start()
        self.mock_mgr = mock.MagicMock()
        self.mock_mgr.available_engines = ["easyocr", "tesseract", "textract"]
        self.mock_get.return_value = self.mock_mgr

    def tearDown(self):
        self.patcher.stop()

    def test_engines_endpoint(self):
        resp = self.client.get("/api/engines")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("all", data)
        self.assertIn("available", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
