# MathVSE

MathVSE is a local-first mathematical solver and explainer built for deterministic equation reasoning and image preprocessing without remote APIs.

## Current status

Phase 1 is implemented and verified. Phase 2 adds validated image input and preprocessing for supported raster files. Phase 3 adds deterministic OCR interfaces and baseline spatial parsing. Phase 4 adds the local Streamlit application. Phase 5A adds optional symbol-CNN training, and Phase 5B connects image-derived crops to the classifier and spatial parser. Real equation recognition still depends on a trained local checkpoint and user review; complete equation OCR from photographs is not claimed.

## Supported image workflow

- PNG, JPG, and JPEG raster inputs
- file-size and dimension safety limits
- validation of unreadable, corrupted, empty, and oversized files
- grayscale conversion and thresholding variants
- conservative denoising and normalization
- preliminary contour and bounding-box detection for debugging and preview

## OCR and spatial parsing status

- typed recognition and symbol-candidate contracts
- classifier interface plus optional offline CNN inference with explicit missing-weight handling
- baseline spatial parser for horizontal single-line expressions
- explicit needs_review and unsupported_structure outcomes
- no fabricated OCR confidence or accuracy claims
- no real model weights or dataset downloads are required for default tests

## Phase 5A: Symbol CNN

Training is an explicit optional workflow and is never run during normal application startup. It uses PyTorch on CPU; the default installation and test suite do not require PyTorch, HASYv2, or trained weights.

Install the optional training dependency and the CLI:

```bash
python -m pip install -e '.[training]'
```

Acquire HASYv2 separately from [hasydataset.com](https://www.hasydataset.com/) or its official project distribution. The upstream dataset is published under CC BY-SA 4.0; review the current dataset license, attribution requirements, and terms before use or redistribution. MathVSE does not fetch the dataset. Extract it locally so the selected dataset root has this layout:

```text
HASYv2/
	hasy-data.csv
	symbols.csv
	hasy-data/
		... PNG samples ...
```

For a locally downloaded archive, extract it without changing its contents; pass the directory containing both CSV files as `DATASET_ROOT`:

```bash
mkdir -p data/HASYv2
unzip /path/to/HASYv2.zip -d data/HASYv2
```

The versioned mapping `mathvse-symbols-v1` supports digits `0`-`9`, `x`, `y`, `+`, `-`, `*`, `/`, `=`, `(`, and `)`. Integer class IDs are zero-based in that listed order. HASYv2 aliases `\\cdot` and `\\times` map to `*`; `\\div` maps to `/`. Other dataset labels, including advanced mathematical notation, are counted and reported as unsupported. Images must decode as 32x32 pixels; exact duplicate grayscale samples are removed before splitting. Splits are deterministic and grouped by duplicate image content.

Validate the local files and inspect unsupported labels, class counts, duplicates, and corrupt samples:

```bash
mathvse-recognition validate-dataset /path/to/HASYv2
```

Train and save a versioned checkpoint (defaults: CPU, 20 epochs, batch size 32, seed 42):

```bash
mathvse-recognition train /path/to/HASYv2 --output models/math_symbols.pt --epochs 20 --batch-size 32 --learning-rate 0.001 --seed 42
```

The checkpoint stores weights, architecture and label-map versions, input/preprocessing contract, configuration, epoch-level training/validation metrics, and a distinct held-out test report. Reproduce the test split and report its metrics again with:

```bash
mathvse-recognition evaluate models/math_symbols.pt /path/to/HASYv2
mathvse-recognition inspect models/math_symbols.pt
```

For Python inference, configure `TorchSymbolClassifier(model_path="models/math_symbols.pt")`. A missing checkpoint returns `model_unavailable`; incompatible metadata or output dimensions raise a configuration error. Training accuracy is not reported as test accuracy. Dataset files and model weights are local artifacts and should not be committed to Git. Symbol classification and baseline spatial parsing do not yet amount to reliable recognition of complete equations in photographs.

Run local image-to-expression inference from the CLI:

```bash
mathvse-recognition infer-image /path/to/equation.png --checkpoint models/math_symbols.pt --confidence-threshold 0.55 --ambiguity-margin 0.08
```

Inference reuses the checkpoint's grayscale, as-stored polarity, 32x32, and zero-to-one input contract. Class candidates are ranked by softmax probability; these scores are not calibrated accuracy estimates. Foreground connected components are preliminary crops, not guaranteed characters. Components touching an image edge, unusually elongated components, low-confidence symbols, and close top classes require review. The UI shows crops and alternatives, keeps machine output separate from the corrected expression, and requires an explicit review confirmation before solving. Without weights or PyTorch, image recognition reports `model_unavailable`; manual text solving remains available. Fractions, superscripts, radicals, and reliable full-photo equation OCR remain unsupported.

## Local application launch

```bash
cd /workspaces/Math-VSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
streamlit run app.py
```

The UI is intentionally local-only and requires no remote API or LLM. If no OCR model is configured, the interface shows model_unavailable instead of inventing a prediction.

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
