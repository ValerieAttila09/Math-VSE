# MathVSE

MathVSE is a local-first mathematical solver and explainer built for deterministic equation reasoning and image preprocessing without remote APIs.

## Current status

Phase 1 is implemented and verified. Phase 2 adds validated image input and preprocessing for supported raster files. Phase 3 adds deterministic OCR interfaces and baseline spatial parsing for supported symbol layouts.

## Supported image workflow

- PNG, JPG, and JPEG raster inputs
- file-size and dimension safety limits
- validation of unreadable, corrupted, empty, and oversized files
- grayscale conversion and thresholding variants
- conservative denoising and normalization
- preliminary contour and bounding-box detection for debugging and preview

## OCR and spatial parsing status

- typed recognition and symbol-candidate contracts
- classifier interface for offline model status and missing-weight handling
- baseline spatial parser for horizontal single-line expressions
- explicit needs_review and unsupported_structure outcomes
- no fabricated OCR confidence or accuracy claims
- no real model weights or dataset downloads are required for default tests

## Installation

```bash
cd /workspaces/Math-VSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Example usage

```python
from mathvse.vision.input import load_image_file
from mathvse.vision.preprocessing import preprocess_image

loaded = load_image_file('sample.png')
result = preprocess_image(loaded.image, strategy_name='balanced')
print(result.status)
print(sorted(result.variants.keys()))
print(result.bounding_boxes)
```

## Testing

```bash
cd /workspaces/Math-VSE
pytest -q
```

## Notes

- Runtime remains local and CPU-friendly.
- No OCR is implemented in this phase.
- Detected contours are marked as preliminary candidates and are not treated as final mathematical symbols.
- The solver and explanation engine remain unchanged from Phase 1 and remain verified.
