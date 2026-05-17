"""
utils/preprocessor.py
Image preprocessing pipeline for handwritten text OCR.
Handles shadows, glare, tilt, noise, and quality enhancement.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
import io
import base64


class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline to improve OCR accuracy.
    Handles common issues: shadows, glare, tilt, low contrast, noise.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.steps_log = []

    def preprocess(self, image_input, pipeline: str = "auto") -> dict:
        """
        Main preprocessing entry point.
        
        Args:
            image_input: PIL Image, numpy array, bytes, or file path
            pipeline: 'auto', 'whiteboard', 'handwritten', 'document', 'flowchart'
        
        Returns:
            dict with 'image' (PIL), 'cv_image' (numpy), 'steps', 'pipeline_used'
        """
        self.steps_log = []

        # --- Normalize input to numpy BGR ---
        img = self._to_cv2(image_input)
        self._log("Input loaded", img)

        # --- Detect pipeline if auto ---
        if pipeline == "auto":
            pipeline = self._detect_pipeline(img)
            self._log(f"Auto-detected pipeline: {pipeline}")

        # --- Run selected pipeline ---
        if pipeline == "whiteboard":
            processed = self._whiteboard_pipeline(img)
        elif pipeline == "handwritten":
            processed = self._handwritten_pipeline(img)
        elif pipeline == "flowchart":
            processed = self._flowchart_pipeline(img)
        else:
            processed = self._document_pipeline(img)

        pil_out = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        return {
            "image": pil_out,
            "cv_image": processed,
            "steps": self.steps_log,
            "pipeline_used": pipeline,
        }

    # ------------------------------------------------------------------ #
    #  Pipelines                                                           #
    # ------------------------------------------------------------------ #

    def _whiteboard_pipeline(self, img: np.ndarray) -> np.ndarray:
        img = self._resize_if_needed(img)
        img = self._correct_tilt(img)
        img = self._remove_shadows(img)
        img = self._enhance_whiteboard(img)
        img = self._denoise(img)
        img = self._sharpen(img)
        return img

    def _handwritten_pipeline(self, img: np.ndarray) -> np.ndarray:
        img = self._resize_if_needed(img)
        img = self._correct_tilt(img)
        img = self._normalize_illumination(img)
        img = self._increase_contrast(img)
        img = self._binarize_adaptive(img)
        img = self._morphological_clean(img)
        return img

    def _flowchart_pipeline(self, img: np.ndarray) -> np.ndarray:
        img = self._resize_if_needed(img)
        img = self._correct_tilt(img)
        img = self._remove_shadows(img)
        img = self._increase_contrast(img)
        img = self._denoise(img)
        img = self._sharpen(img)
        return img

    def _document_pipeline(self, img: np.ndarray) -> np.ndarray:
        img = self._resize_if_needed(img)
        img = self._correct_tilt(img)
        img = self._normalize_illumination(img)
        img = self._increase_contrast(img)
        img = self._denoise(img)
        img = self._binarize_otsu(img)
        return img

    # ------------------------------------------------------------------ #
    #  Step implementations                                                #
    # ------------------------------------------------------------------ #

    def _resize_if_needed(self, img: np.ndarray, min_dim: int = 800, max_dim: int = 4000) -> np.ndarray:
        h, w = img.shape[:2]
        longest = max(h, w)
        shortest = min(h, w)

        if longest > max_dim:
            scale = max_dim / longest
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self._log(f"Downscaled to {img.shape[1]}x{img.shape[0]}")
        elif shortest < min_dim:
            scale = min_dim / shortest
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            self._log(f"Upscaled to {img.shape[1]}x{img.shape[0]}")
        return img

    def _correct_tilt(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct skew using Hough transform."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarize to find dark text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate horizontally to connect text into baselines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find edges on the dilated image
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        
        # Threshold proportional to image width to only detect long lines
        min_line_len = int(img.shape[1] * 0.2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min_line_len)

        if lines is None:
            return img

        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return img  # negligible tilt

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        self._log(f"Corrected tilt: {median_angle:.2f}°")
        return rotated

    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """Remove uneven illumination and shadows using morphological approach."""
        rgb_planes = cv2.split(img)
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX)
            result_planes.append(norm)
        result = cv2.merge(result_planes)
        self._log("Shadow removal applied")
        return result

    def _normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """CLAHE-based local contrast normalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self._log("CLAHE illumination normalization applied")
        return result

    def _increase_contrast(self, img: np.ndarray) -> np.ndarray:
        """Increase contrast with gamma correction."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        gamma = 1.0 if 100 < mean_val < 160 else (0.7 if mean_val > 160 else 1.4)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)]).astype(np.uint8)
        result = cv2.LUT(img, table)
        self._log(f"Gamma correction: {gamma:.2f}")
        return result

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Fast Non-Local Means denoising."""
        result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        self._log("NLM denoising applied")
        return result

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask sharpening."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)
        self._log("Sharpening applied")
        return result

    def _binarize_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Adaptive threshold binarization (handles uneven lighting)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._log("Adaptive binarization applied")
        return result

    def _binarize_otsu(self, img: np.ndarray) -> np.ndarray:
        """Otsu's binarization."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._log("Otsu binarization applied")
        return result

    def _enhance_whiteboard(self, img: np.ndarray) -> np.ndarray:
        """Special whiteboard enhancement: boost dark ink on bright bg."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Estimate background by dilating (removes dark text)
        kernel = np.ones((15, 15), np.uint8)
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        # Subtract original from background to get white text on black background
        diff = cv2.subtract(bg, gray)
        # Invert to get dark text on bright background
        inv = cv2.bitwise_not(diff)
        norm = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
        result = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        self._log("Whiteboard enhancement applied")
        return result

    def _morphological_clean(self, img: np.ndarray) -> np.ndarray:
        """Remove small noise blobs via morphological opening."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        self._log("Morphological cleaning applied")
        return result

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _detect_pipeline(self, img: np.ndarray) -> str:
        """Heuristic pipeline detection based on image characteristics."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        std = np.std(gray)

        # Whiteboard: very bright mean, low std
        if mean > 180 and std < 50:
            return "whiteboard"
        # Flowchart: many edges, moderate brightness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.05 and mean > 100:
            return "flowchart"
        # Dark/noisy: handwritten notes on paper
        if mean < 130 or std > 70:
            return "handwritten"
        return "document"

    def _to_cv2(self, image_input) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _log(self, message: str, img: np.ndarray = None):
        entry = {"step": len(self.steps_log) + 1, "message": message}
        if img is not None and self.debug:
            entry["shape"] = img.shape
        self.steps_log.append(entry)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
