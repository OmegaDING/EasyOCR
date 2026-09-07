#!/usr/bin/env python
"""Build phase-1 OCR annotations from this checkout, not an installed EasyOCR."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="./dataset")
    parser.add_argument("--output_dir", default="./ocr_annotations")
    parser.add_argument("--filename_filter", default="hr_canonical.png")
    parser.add_argument("--languages", nargs="+", default=["ch_sim", "en"])
    parser.add_argument("--gpu", default="0", help="CUDA index, cuda:N, or cpu; CPU fallback if unavailable")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum equal-width crop batch size")
    parser.add_argument("--model_storage_directory", default="./.ocr_models")
    parser.add_argument("--download_disabled", action="store_true")
    parser.add_argument("--save_features", action="store_true", default=True, help="Always enabled in phase 1")
    parser.add_argument("--save_logits", action="store_true", default=True, help="Always enabled in phase 1")
    parser.add_argument("--visualize", action="store_true", default=True, help="Always enabled in phase 1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--low_confidence_threshold", type=float, default=0.1)
    parser.add_argument("--text_threshold", type=float, default=0.7)
    parser.add_argument("--link_threshold", type=float, default=0.4)
    parser.add_argument("--low_text", type=float, default=0.4)
    parser.add_argument("--canvas_size", type=int, default=2560)
    parser.add_argument("--mag_ratio", type=float, default=1.)
    parser.add_argument("--cpu_threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch_size and cpu_threads must be positive")
    if not 0 <= args.low_confidence_threshold <= 1:
        parser.error("low_confidence_threshold must be in [0,1]")
    root = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    if not root.is_dir():
        parser.error("input_dir must exist")
    if output == root or output.is_relative_to(root):
        parser.error("output_dir must be outside input_dir")
    if (Path(args.filename_filter).name != args.filename_filter
            or any(c in args.filename_filter for c in "*?[]")):
        parser.error("filename_filter must be a literal basename")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name == args.filename_filter)
    print("Selected " + str(len(paths)) + " files named exactly " + args.filename_filter, flush=True)
    try:
        import torch
        from easyocr.annotation_builder import build_annotations
        from easyocr.ocr_teacher import OCRPipeline
        torch.set_num_threads(args.cpu_threads)
        pipeline = OCRPipeline(
            languages=args.languages, gpu=args.gpu,
            model_storage_directory=args.model_storage_directory,
            download_enabled=not args.download_disabled,
            text_threshold=args.text_threshold, link_threshold=args.link_threshold,
            low_text=args.low_text, canvas_size=args.canvas_size, mag_ratio=args.mag_ratio)
    except Exception as error:
        # Do not replace a completed run's report with an environment failure.
        if not (output / "meta.json").exists() and not (output / "annotations.jsonl").exists():
            output.mkdir(parents=True, exist_ok=True)
            report = dict(status="blocked_before_inference", all_tests_passed=None,
                          discovered_hr_canonical_images=len(paths), processed_images=0,
                          failed_images=0, not_attempted_images=len(paths), total_regions=0,
                          error=type(error).__name__ + ": " + str(error),
                          images=[dict(image_path=root.name + "/" + p.relative_to(root).as_posix(),
                                       region_count=None, status="not_attempted") for p in paths])
            for key in ("avg_regions_per_image", "min_regions_per_image", "max_regions_per_image",
                        "avg_recognition_confidence", "min_recognition_confidence", "max_recognition_confidence",
                        "empty_text_count", "low_confidence_count", "feature_nan_count", "feature_inf_count",
                        "logits_nan_count", "logits_inf_count", "feature_shape_distribution", "logits_shape_distribution",
                        "deterministic_test_passed", "gradient_test_passed", "old_new_test_passed",
                        "probability_test_passed", "shape_test_passed", "cached_tensor_test_passed"):
                report[key] = None
            temp = output / "validation_report.json.tmp"
            temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, output / "validation_report.json")
        print("Teacher initialization failed: " + str(error), file=sys.stderr)
        return 2
    report = build_annotations(
        pipeline, args.input_dir, args.output_dir, args.filename_filter,
        args.batch_size, args.resume, args.low_confidence_threshold)
    print("Processed: {processed_images}; failed: {failed_images}; regions: {total_regions}; status: {status}".format(**report),
          flush=True)
    return 0 if report["all_tests_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
