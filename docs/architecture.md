# Architecture Notes — HandwriteAI OCR

## Preprocessing Pipeline

Each pipeline is a sequence of image processing steps tuned for its image class.

### Whiteboard Pipeline
1. Resize to working resolution (800–4000 px)
2. Tilt correction via Hough line detection
3. Shadow removal (morphological background subtraction)
4. Whiteboard enhancement (dark-ink isolation)
5. NLM denoising
6. Unsharp mask sharpening

### Handwritten Notes Pipeline
1. Resize
2. Tilt correction
3. Illumination normalization (CLAHE on LAB L-channel)
4. Gamma contrast boost
5. Adaptive Gaussian binarization
6. Morphological noise cleaning (opening)

### Flowchart Pipeline
1. Resize
2. Tilt correction
3. Shadow removal
4. Contrast boost
5. NLM denoising
6. Sharpening

### Document Pipeline
1. Resize
2. Tilt correction
3. Illumination normalization
4. Contrast boost
5. NLM denoising
6. Otsu binarization

## OCR Engine Selection

The `OCRManager._pick_best()` method scores each engine result as:

```
score = confidence × √(word_count)
```

This rewards results that are both confident AND contain more text,
preventing a high-confidence but nearly empty result from winning over
a thorough extraction.

## TrOCR Line Splitting

TrOCR is a line-level model — it expects single lines of handwriting.
`TrOCREngine._split_into_lines()` uses horizontal projection profiles
(row-wise pixel sums on a binarized image) to detect text line extents,
then feeds each strip to TrOCR independently before joining results.
