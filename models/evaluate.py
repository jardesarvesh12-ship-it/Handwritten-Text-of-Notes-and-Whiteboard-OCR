"""
models/evaluate.py

Benchmarks OCR engines against synthetic dataset.
Computes CER (Character Error Rate) and WER (Word Error Rate).

Usage:
    python models/evaluate.py [--engine auto|tesseract|easyocr] [--limit 20]
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    ref, hyp = ref.lower().strip(), hyp.lower().strip()
    if not ref:
        return 0.0
    # DP edit distance
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n] / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    if not ref_w:
        return 0.0
    m, n = len(ref_w), len(hyp_w)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n] / len(ref_w)


def run_evaluation(engine: str = "auto", limit: int = 20):
    from PIL import Image
    from utils.preprocessor import ImagePreprocessor
    from utils.ocr_engine import OCRManager

    dataset_dir = Path(__file__).parent.parent / "dataset"
    meta_file = dataset_dir / "dataset_meta.json"

    if not meta_file.exists():
        print("Dataset not found. Run: python dataset/prepare_dataset.py first.")
        return

    with open(meta_file) as f:
        meta = json.load(f)

    samples = meta.get("synthetic", [])[:limit]
    if not samples:
        print("No synthetic samples found.")
        return

    preprocessor = ImagePreprocessor()
    ocr_mgr = OCRManager(engines=["easyocr", "tesseract"])

    results = []
    print(f"\nEvaluating {len(samples)} samples with engine='{engine}'…\n")
    print(f"{'#':<4} {'Type':<14} {'CER':>6} {'WER':>6} {'Conf':>6} {'ms':>6}")
    print("-" * 48)

    for i, sample in enumerate(samples):
        fpath = Path(sample["file"])
        if not fpath.exists():
            continue
        ref_text = sample["text"]
        img_type = sample["type"]

        img = Image.open(fpath).convert("RGB")
        pre = preprocessor.preprocess(img, pipeline=img_type)
        t0 = time.time()
        result = ocr_mgr.extract(pre["image"], engine=engine, mode=img_type)
        elapsed = int((time.time() - t0) * 1000)

        hyp_text = result.get("text", "")
        c = cer(ref_text, hyp_text)
        w = wer(ref_text, hyp_text)
        conf = result.get("confidence", 0)

        results.append({
            "file": fpath.name,
            "type": img_type,
            "cer": round(c, 4),
            "wer": round(w, 4),
            "confidence": conf,
            "ms": elapsed,
        })

        print(f"{i+1:<4} {img_type:<14} {c:>6.3f} {w:>6.3f} {conf:>5}% {elapsed:>5}ms")

    if not results:
        print("No results computed.")
        return

    avg_cer = sum(r["cer"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_ms = sum(r["ms"] for r in results) / len(results)

    print("-" * 48)
    print(f"{'AVG':<4} {'':<14} {avg_cer:>6.3f} {avg_wer:>6.3f} {avg_conf:>5.0f}% {avg_ms:>5.0f}ms")
    print(f"\nSummary: {len(results)} samples | CER={avg_cer:.3f} | WER={avg_wer:.3f}")

    # Save report
    report = {
        "engine": engine,
        "samples": len(results),
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "avg_confidence": round(avg_conf, 1),
        "avg_ms": round(avg_ms, 1),
        "per_sample": results,
    }
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Evaluation")
    parser.add_argument("--engine", default="auto", help="OCR engine to test")
    parser.add_argument("--limit", type=int, default=20, help="Max samples to evaluate")
    args = parser.parse_args()
    run_evaluation(args.engine, args.limit)
