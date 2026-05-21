"""
ocr_cli.py — Command-line interface for HandwriteAI OCR

Usage examples:
    python ocr_cli.py image.jpg
    python ocr_cli.py image.jpg --pipeline whiteboard --engine easyocr
    python ocr_cli.py image.jpg --output result.txt
    python ocr_cli.py image.jpg --save-preprocessed out_preprocessed.png
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="HandwriteAI — Handwritten Text & Whiteboard OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s notes.jpg
  %(prog)s whiteboard.png --pipeline whiteboard --engine easyocr
  %(prog)s diagram.jpg   --pipeline flowchart  --output extracted.txt
  %(prog)s scan.jpg      --engine trocr        --save-preprocessed clean.png
        """
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--pipeline", "-p",
        choices=["auto", "whiteboard", "handwritten", "flowchart", "document"],
        default="auto",
        help="Preprocessing pipeline (default: auto)"
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["auto", "tesseract", "easyocr", "trocr", "textract"],
        default="auto",
        help="OCR engine (default: auto)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save extracted text to this file"
    )
    parser.add_argument(
        "--save-preprocessed", metavar="PATH",
        help="Save preprocessed image to this path"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show preprocessing steps"
    )

    args = parser.parse_args()

    # Validate input
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    # Import project modules
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    try:
        from PIL import Image
        from utils.preprocessor import ImagePreprocessor
        from utils.ocr_engine import OCRManager
    except ImportError as e:
        print(f"Import error: {e}\nRun: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    print(f"Loading image: {img_path}", file=sys.stderr)
    image = Image.open(img_path).convert("RGB")

    # Preprocess
    preprocessor = ImagePreprocessor()
    print(f"Preprocessing ({args.pipeline} pipeline)…", file=sys.stderr)
    pre_result = preprocessor.preprocess(image, pipeline=args.pipeline)

    if args.verbose:
        print("\nPreprocessing steps:", file=sys.stderr)
        for s in pre_result["steps"]:
            print(f"  {s['step']}. {s['message']}", file=sys.stderr)

    if args.save_preprocessed:
        pre_result["image"].save(args.save_preprocessed)
        print(f"Preprocessed image saved: {args.save_preprocessed}", file=sys.stderr)

    # Load .env file if available
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
    except ImportError:
        import os

    # OCR
    trocr_enabled = os.getenv("TROCR_ENABLED", "false").lower() == "true" or args.engine == "trocr"
    engines = ["easyocr", "tesseract", "textract"]
    if trocr_enabled:
        engines.append("trocr")
        
    ocr_mgr = OCRManager(engines=engines, trocr_enabled=trocr_enabled)
    print(f"Running OCR ({args.engine} engine)…", file=sys.stderr)
    ocr_result = ocr_mgr.extract(
        pre_result["image"],
        engine=args.engine,
        mode=pre_result["pipeline_used"]
    )

    text = ocr_result.get("text", "")

    # Output
    if args.json:
        out = {
            "text": text,
            "confidence": ocr_result.get("confidence"),
            "engine": ocr_result.get("engine"),
            "pipeline": pre_result["pipeline_used"],
            "word_count": len(text.split()),
            "processing_ms": ocr_result.get("processing_time_ms"),
        }
        print(json.dumps(out, indent=2))
    else:
        print("\n" + "=" * 60)
        print("EXTRACTED TEXT")
        print("=" * 60)
        print(text if text else "(No text detected)")
        print("=" * 60)
        print(f"Engine: {ocr_result.get('engine')} | "
              f"Confidence: {ocr_result.get('confidence')}% | "
              f"Words: {len(text.split())} | "
              f"Time: {ocr_result.get('processing_time_ms')}ms")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nText saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
