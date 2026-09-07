# OCR teacher and offline annotations (phase 1)

This checkout supports an isolated CRAFT + official zh_sim_g2 teacher. VOSR is
not involved. The dataset is read-only; the default filename match is the exact,
case-sensitive basename `hr_canonical.png`, recursively beneath `./dataset`.

## Verified source map

- Detection: `easyocr/easyocr.py: Reader.readtext -> Reader.detect`, then
  `easyocr/detection.py: get_textbox -> test_net`, calling
  `easyocr/craft.py: CRAFT.forward` and `craft_utils.getDetBoxes`.
- Recognition: `Reader.recognize -> utils.get_image_list -> recognition.get_text
  -> AlignCollate/NormalizePAD -> recognizer_predict -> Model.forward`.
- Generation 1: `easyocr/model/model.py: Model` uses
  `modules.ResNet_FeatureExtractor`, two `modules.BidirectionalLSTM` blocks,
  then `Prediction: nn.Linear`. Standard hidden dimension is 512.
- Generation 2: `easyocr/model/vgg_model.py: Model` uses
  `modules.VGG_FeatureExtractor`, two `modules.BidirectionalLSTM` blocks,
  then `Prediction: nn.Linear`. Standard hidden dimension is 256.
  `Reader(["ch_sim", "en"])` selects `zh_sim_g2` in the checked-out config.
- CTC: `utils.CTCLabelConverter.decode_greedy`.
- Confidence: `recognition.recognizer_predict` selects nonblank time-step
  probabilities and calls `recognition.custom_mean`. This includes repeated
  nonblank steps, not just collapsed characters.
- Adaptive contrast: `recognition.get_text` runs a second loader/forward for
  low-confidence crops and selects the higher score. Annotation bypasses this.
- The new optional `Model.forward(..., return_features=True)` returns the real
  `SequenceModeling` output and its classifier logits. The default signature
  remains logits-only. State dict names and weights do not change.

`BidirectionalLSTM.forward` itself includes a linear projection after the
bidirectional RNN. F means the final, projected `SequenceModeling` output that
is actually passed to `Prediction`, not an unprojected hidden state.
D and C are read from the loaded `Prediction.in_features/out_features`.

## Server commands

Use your server's compatible PyTorch/torchvision installation (CUDA if desired),
then install this checkout's remaining requirements. Python 3.9 or newer is
required for this annotation interface. From the repository root:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_ocr_teacher.py -q
python tools/build_ocr_annotations.py \
  --input_dir ./dataset/prepared \
  --output_dir ./ocr_annotations/prepared \
  --filename_filter hr_canonical.png \
  --languages ch_sim en \
  --gpu 0 --batch_size 16 \
  --save_features --save_logits --visualize --resume
```

`--gpu cpu` forces CPU. If CUDA is unavailable the teacher falls back to CPU.
Official weights are downloaded into `./.ocr_models`; use
`--model_storage_directory /path/to/models --download_disabled` for offline
servers with weights already present. Quantization is disabled on every device.
The CLI prints one region count per image and exits nonzero on failed or
unexercised required acceptance checks. Features, logits and visualization are
mandatory in phase 1; their CLI flags are accepted for clarity and default on.

The initial non-resume run refuses a nonempty output directory. Resume compares
teacher source/weights/runtime/config identity, original-image hashes, tensor
and visualization hashes, per-record annotation checksums, and tensor structure. Damaged or outdated sample
artifacts are regenerated. Teacher identity changes require a fresh output
directory. Do not run two builders concurrently against the same output directory.

## Python API

```python
from pathlib import Path
import torch
from easyocr.ocr_teacher import OCRPipeline

pipeline = OCRPipeline(languages=["ch_sim", "en"], gpu=True)
image = Path("dataset/example/hr_canonical.png")
boxes = pipeline.detect(image)
for box in boxes:
    crop = pipeline.crop(image, box)       # uint8 grayscale, rectified
    with torch.no_grad():                # caller owns offline no_grad policy
        raw = pipeline.recognize_raw(crop)
    print(raw.feature.shape)             # [1, T, D]
    print(raw.logits.shape)              # [1, T, C], raw and unmasked
    decoded = pipeline.decode(raw.logits)
    print(decoded.text, decoded.confidence)
```

The four components are `TextDetector`, `RecognitionEncoder`,
`CharacterPredictionHead`, and `CTCDecoder`. `OCRRecognizer` composes encoder
and head. `decode_batch` returns one decoded dataclass per batch row.
`recognize_many(crops, batch_size)` yields
`(original_region_index, OCRForwardOutput, [imgH, imgW])`. It groups only
equal-width crops, so padding/T does not depend on neighboring crop widths.

For the future frozen-teacher gradient path:

```python
fake_sr = pipeline.preprocess(crop).detach().requires_grad_(True)
raw = pipeline.recognizer(fake_sr)
loss = raw.feature.abs().mean() + raw.logits.abs().mean()
loss.backward()
assert fake_sr.grad is not None
assert all(p.grad is None for p in pipeline.recognizer.parameters())
```

The recognizer accepts normalized `[B,1,H,W]` floating tensors and preserves
their graph. NumPy/OpenCV/PIL crop preprocessing is an offline operation and is
not differentiable. A later VOSR integration must implement corresponding
differentiable image crop/resize operations and verify their numerical match;
the current test proves gradients through the OCR model, not through a full
SR-image-to-PIL pipeline. On CUDA, input-gradient forwards use native LSTM:
cuDNN's eval-mode RNN forward does not support backward. Annotation forwards
use the normal deterministic inference backend. CPU/CUDA numerical parity is
not claimed; device/runtime identity is recorded.

## Coordinates, preprocessing and decoding

The detector returns CRAFT's original four points, ordered TL/TR/BR/BL, in
original-image pixels. It deliberately does not apply `group_text_box`, a
horizontal-line merge, or an added margin. This differs from default full-image
`readtext()` boxes, while preserving the requested canonical CRAFT geometry.
Coordinates are not normalized or silently clamped. Cropping uses the saved
polygon with `utils.four_point_transform` (OpenCV perspective warp and zero
border). Region IDs follow a stable top-to-bottom, then left-to-right ordering.

After rectification the pipeline reuses `compute_ratio_and_resize` and then
`AlignCollate`: grayscale, configured imgH=64, PIL bicubic resizing, [-1,1]
normalization, and right-edge padding. Width is
`ceil(max(w/h, h/w, 1))*imgH` for that crop alone. This also preserves the
repository's existing handling of tall crops rather than silently changing it.
Both pre-resize crop size and actual OCR input size are recorded per region.

Saved Z is the original raw classifier output, including CTC blank at class 0.
Only the decoder applies the Reader's language character mask and renormalizes
probabilities, exactly as the original recognizer does. P=softmax(Z) is a
separate, unmasked distribution for future distillation. There is no adaptive
contrast, rotation selection, or confidence-based second pass in annotation.

Confidence and text come from the same inference-dtype Z as F. Converting Z to
float16 can change a nearly tied argmax; validation explicitly reports such a
cache-decode difference as a warning. It does not fail GT acceptance or silently
relabel the original result, because the saved JSONL text is decoded from the
same pre-cast forward logits as F and Z.

## Artifacts and validation

`annotations.jsonl` contains one record per successfully processed image,
including images with zero regions. Every region indexes the corresponding
entry in `tensors/<sample_id>.pt`. The payload contains lists `features`
(`[T,D]`) and `logits` (`[T,C]`), detached on CPU in float16, plus sample
and region IDs. JSON tensor paths are relative to the output root; image paths
are relative to the parent of the input root. IDs include a readable relative
parent path plus a deterministic hash, avoiding collisions between nested names.

`meta.json` records checkpoint paths and SHA256, source hashes and commit,
charset and hash, all preprocessing/decoder/detector settings, dimensions,
runtime/device/dtypes and validation tolerances. `visualizations/*_bbox.png`
shows boxes and region IDs on the full-resolution original.

Strict `torch.use_deterministic_algorithms(True)` is deliberately disabled:
CUDA does not implement a deterministic backward for EasyOCR's
`AdaptiveAvgPool2d`. Enabling it prevents the required frozen-teacher input
gradient test from running. Eval mode, disabled TF32, fixed cuDNN selection and
the per-image repeated-forward allclose test remain enforced; the exact setting
and reason are written to `meta.json`.

`validation_report.json` contains input/success/failure counts, per-image
region counts, confidence statistics, empty/low-confidence counts, tensor
shape distributions, successful-cache NaN/Inf counts, old/new differences,
and acceptance flags. Failed images include an error rather than fabricated
regions; their invalid tensors are not counted as successful caches.
An all-empty dataset produces null (unexercised), not true, for forward tests.

- Every region: F/Z dimensions, finite values, Z=Prediction(F), probability
  normalization, exact original `get_text` text consistency with identical
  crop, width and language mask, plus confidence within 5e-5 to account for
  single-crop versus equal-width CUDA batch arithmetic; contrast_ths=0 and
  adjust_contrast=0.
- First region of every nonempty image: repeated forward allclose and frozen
  teacher/input-gradient backward.
- Every saved region: safe tensor-only reload, matching IDs/indices/shapes,
  exact equality after fp16 conversion and fp32 comparison at rtol/atol=1e-3.
- Unit tests also exercise both generations, CTC blank/repeats/masking,
  batch-composition invariance, filtering, ID collisions, empty detections,
  resume, and corruption recovery.

Original `readtext()` with its default merging, shared batch padding or
contrast retry is not the parity reference. Any observed old/new differences
under matched conditions are listed with region IDs in the runtime report.
