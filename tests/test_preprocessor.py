"""
tests/test_preprocessor.py
Unit tests for the image preprocessing pipeline.
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocessor import ImagePreprocessor, image_to_base64


class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.pp = ImagePreprocessor()
        # Create a simple test image: white background, black text simulation
        img = Image.new("RGB", (400, 300), (245, 242, 235))
        self.test_image = img

    # ── Pipeline detection ──────────────────────────────────────── #

    def test_whiteboard_detection(self):
        """Bright, low-std image → whiteboard pipeline."""
        bright = Image.new("RGB", (400, 300), (240, 240, 240))
        result = self.pp.preprocess(bright, pipeline="auto")
        self.assertEqual(result["pipeline_used"], "whiteboard")

    def test_explicit_pipeline(self):
        """Explicit pipeline override is respected."""
        for pl in ["whiteboard", "handwritten", "flowchart", "document"]:
            result = self.pp.preprocess(self.test_image, pipeline=pl)
            self.assertEqual(result["pipeline_used"], pl)

    # ── Output types ────────────────────────────────────────────── #

    def test_returns_pil_image(self):
        result = self.pp.preprocess(self.test_image)
        self.assertIsInstance(result["image"], Image.Image)

    def test_returns_cv_image(self):
        result = self.pp.preprocess(self.test_image)
        self.assertIsInstance(result["cv_image"], np.ndarray)
        self.assertEqual(result["cv_image"].ndim, 3)

    def test_returns_steps(self):
        result = self.pp.preprocess(self.test_image)
        self.assertIsInstance(result["steps"], list)
        self.assertGreater(len(result["steps"]), 0)

    # ── Input type handling ─────────────────────────────────────── #

    def test_numpy_input(self):
        arr = np.array(self.test_image)
        result = self.pp.preprocess(arr)
        self.assertIsInstance(result["image"], Image.Image)

    def test_bytes_input(self):
        import io
        buf = io.BytesIO()
        self.test_image.save(buf, format="PNG")
        result = self.pp.preprocess(buf.getvalue())
        self.assertIsInstance(result["image"], Image.Image)

    # ── Size preservation ───────────────────────────────────────── #

    def test_output_size_reasonable(self):
        result = self.pp.preprocess(self.test_image)
        out_w, out_h = result["image"].size
        # Should not be absurdly different from input
        self.assertGreater(out_w, 100)
        self.assertGreater(out_h, 100)

    # ── Base64 helper ───────────────────────────────────────────── #

    def test_image_to_base64(self):
        b64 = image_to_base64(self.test_image)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 100)
        # Should be valid base64
        import base64
        decoded = base64.b64decode(b64)
        self.assertGreater(len(decoded), 0)


class TestPreprocessorSteps(unittest.TestCase):
    """Test individual preprocessing step methods."""

    def setUp(self):
        self.pp = ImagePreprocessor()
        arr = np.ones((200, 300, 3), dtype=np.uint8) * 200
        self.cv_img = arr

    def test_resize_no_change_needed(self):
        result = self.pp._resize_if_needed(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)

    def test_resize_downscale(self):
        large = np.ones((5000, 6000, 3), dtype=np.uint8) * 128
        result = self.pp._resize_if_needed(large, max_dim=4000)
        self.assertLessEqual(max(result.shape[:2]), 4000)

    def test_remove_shadows_output_shape(self):
        result = self.pp._remove_shadows(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)

    def test_denoise_output_shape(self):
        result = self.pp._denoise(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)

    def test_sharpen_output_shape(self):
        result = self.pp._sharpen(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)

    def test_binarize_adaptive_output_shape(self):
        result = self.pp._binarize_adaptive(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)

    def test_binarize_otsu_output_shape(self):
        result = self.pp._binarize_otsu(self.cv_img)
        self.assertEqual(result.shape, self.cv_img.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
