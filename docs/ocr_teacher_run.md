# Phase-1 handoff (2026-09-07)

The teacher, builder, validation code and tests are implemented. Actual
numerical acceptance is blocked, not completed.

## Changes and documentation

- `easyocr/model/model.py`, `easyocr/model/vgg_model.py`: true encoder output
  and optional F/Z return, preserving the original default forward/state dict.
- `easyocr/easyocr.py`: quantize tuple fix and selected checkpoint identity.
- `easyocr/ocr_teacher.py`: detector, crop, encoder, classifier, decoder API.
- `easyocr/annotation_builder.py`: discovery, metadata, tensors, JSONL,
  visualization, checksums and resume.
- `easyocr/annotation_validation.py`: numerical and cache acceptance checks.
- `tools/build_ocr_annotations.py`: CLI, including startup failure reports.
- `tests/test_ocr_teacher.py`: both generations, CTC, input gradients,
  preprocessing/batching, filtering, IDs, empty images and corruption recovery.
- `.gitignore`: local runtime, model and output directories.
- `docs/ocr_teacher.md`: source call chain, API, server CLI and contracts.

The complete implementation diff is `docs/ocr_teacher_implementation.patch`.
It includes the files above, excluding the exported patch itself, this handoff
note and generated data. No VOSR or source dataset files were modified.

## Actual local results

- Syntax compilation: passed.
- CLI --help: passed.
- git diff --check using repository line-ending settings: passed.
- git apply --reverse --check on the exported patch: passed (read-only).
- Exact recursive filename scan: 10 images, all 1191 x 1684 pixels.
- Actual CLI launch: stopped at `ModuleNotFoundError: No module named 'torch'`.
- Pytest and numerical/gradient/old-new checks: not executed.

The local Python lacks OCR dependencies. An isolated installation and official
weight downloads were attempted. Bounded curl transfers timed out after
180 seconds with only partial files:

| File | Bytes received | Total bytes |
| --- | ---: | ---: |
| PyTorch wheel | 3762286 | 124114011 |
| zh_sim_g2 archive | 3175570 | 20288076 |
| CRAFT archive | 4521984 | 77251756 |

The ignored `.ocr_runtime` and `.ocr_models` directories are incomplete local
download staging, not a usable server environment.

## Dataset report

Each directory below contains sample_000001 and sample_000002, both with
hr_canonical.png. Every image is currently not attempted, with region count null.

- bedroom_desk__incandescent
- large_office__fluorescent
- living_room_table__multi_source
- outdoor_table__single_point
- small_office__daylight

SHA256 comparison found two unique image contents repeated across the five
directories. All ten paths remain independent samples.

The actual `ocr_annotations/validation_report.json` records:

```json
{
  "status": "blocked_before_inference",
  "discovered_hr_canonical_images": 10,
  "processed_images": 0,
  "failed_images": 0,
  "not_attempted_images": 10,
  "all_tests_passed": null,
  "error": "ModuleNotFoundError: No module named 'torch'"
}
```

There is no actual annotation sample, observed F/Z shape, bbox visualization or
measured old/new difference yet. Unexecuted numerical fields are null.

## Server continuation

From this checkout with compatible dependencies installed:

```bash
python -m pytest tests/test_ocr_teacher.py -q
python tools/build_ocr_annotations.py \
  --input_dir ./dataset --output_dir ./ocr_annotations \
  --filename_filter hr_canonical.png --languages ch_sim en \
  --gpu 0 --batch_size 16 --save_features --save_logits --visualize --resume
```

Official weights download by default. For offline weights and the Python API,
see `docs/ocr_teacher.md`. Successful execution will supply the missing
per-image region counts, annotation examples, shapes and visualizations.

## CUDA retry fix

The first server run exposed a strict-determinism incompatibility in CUDA
AdaptiveAvgPool2d backward during the required input-gradient check. The source
now disables only `torch.use_deterministic_algorithms(True)`, while retaining
eval mode, disabled TF32, deterministic cuDNN selection and repeated-forward
allclose validation. Pull this revision, remove the failed output directory or
use a new output directory, then rerun the command above.
