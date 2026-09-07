"""Offline per-image annotations, resumable caches and acceptance reports."""
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch

from . import __version__
from .annotation_validation import (validate_cache, validate_input_gradient,
                                    validate_region, validate_repeat)
from .config import detection_models
from .utils import reformat_input


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(input_dir, filename="hr_canonical.png"):
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Input directory not found: " + str(root))
    if Path(filename).name != filename or any(c in filename for c in "*?[]"):
        raise ValueError("filename_filter must be a literal basename")
    # Explicit equality also enforces case sensitivity on Windows.
    paths = sorted((p for p in root.rglob("*")
                    if p.is_file() and p.name == filename),
                   key=lambda p: p.relative_to(root).as_posix())
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Input symlink escapes dataset: " + str(path))
    return paths


def sample_id(path, input_dir):
    parent = Path(path).relative_to(input_dir).parent.as_posix()
    readable = re.sub(r"[^a-zA-Z0-9_.-]", "_", parent.replace("/", "__"))
    readable = readable.strip(".") or "root"
    return readable[:100] + "--" + hashlib.sha256(parent.encode("utf-8")).hexdigest()[:12]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def write_jsonl(path, records):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def record_checksum(record):
    data = {k: v for k, v in record.items() if k != "annotation_sha256"}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def teacher_metadata(pipeline, input_dir, filename, batch_size):
    package = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=package.parent, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    reader = pipeline.reader
    checkpoint = Path(reader.recognition_checkpoint)
    detector_checkpoint = Path(reader.model_storage_directory) / detection_models["craft"]["filename"]
    charset = reader.converter.character
    libraries = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "opencv-python-headless",
                 "opencv-python", "scipy", "scikit-image"):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    code_files = ["ocr_teacher.py", "annotation_builder.py", "annotation_validation.py",
                  "recognition.py", "easyocr.py", "utils.py", "config.py",
                  "detection.py", "craft.py", "craft_utils.py", "imgproc.py",
                  "model/model.py", "model/vgg_model.py", "model/modules.py"]
    return dict(
        schema_version=1, easyocr_version=__version__, git_commit=commit,
        library_versions=libraries,
        source_sha256={p: sha256_file(package / p) for p in code_files},
        languages=pipeline.languages, detector_model="craft",
        recognition_model=checkpoint.stem, generation=reader.recog_network,
        recognition_checkpoint_path=str(checkpoint.resolve()),
        recognition_checkpoint_sha256=sha256_file(checkpoint),
        detector_checkpoint_path=str(detector_checkpoint.resolve()),
        detector_checkpoint_sha256=sha256_file(detector_checkpoint),
        character_set=charset,
        charset_sha256=hashlib.sha256(json.dumps(charset, ensure_ascii=False).encode("utf-8")).hexdigest(),
        blank_index=0, ignore_indices=pipeline.decoder.ignore_indices,
        feature_dim=pipeline.recognizer.feature_dim, num_classes=pipeline.recognizer.num_classes,
        feature_semantics="Final SequenceModeling output, including BidirectionalLSTM projections",
        logits_semantics="Raw Prediction(F); unmasked, pre-softmax, includes blank",
        imgH=pipeline.imgH, imgW="ceil(max(w/h,h/w,1))*imgH per rectified crop",
        keep_ratio_with_pad=True, decoder="greedy", contrast_ths=0., adjust_contrast=0.,
        deterministic_mode=True, deterministic_algorithms=True,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_benchmark=False, quantize=False, tf32=False,
        detector_thresholds=pipeline.detector.config,
        bbox_format="CRAFT quadrilateral [TL,TR,BR,BL] in original pixels; no grouping or margin",
        crop_policy="utils.four_point_transform; OpenCV linear warp, zero border",
        preprocessing="compute_ratio_and_resize then AlignCollate PIL bicubic, [-1,1], right edge pad",
        gradient_input_contract="normalized [B,1,imgH,imgW]; NumPy/PIL crop preprocessing is not differentiable",
        torch_num_threads=torch.get_num_threads(),
        device=str(pipeline.device), inference_dtype=str(next(pipeline.recognizer.parameters()).dtype),
        saved_tensor_dtype="float16", batch_size=batch_size, batching="equal-width buckets",
        input_root=os.path.relpath(input_dir, Path.cwd()).replace(os.sep, "/"),
        image_path_base="parent of input_root", tensor_path_base="output_root",
        filename_filter=filename, torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        validation_policy="old/new, shape, head, probability and cache: every region; repeat and gradient: first region of every nonempty image",
        tolerances=dict(old_confidence_atol=1e-5, repeat_rtol=1e-5, repeat_atol=1e-6,
                        fp16_rtol=1e-3, fp16_atol=1e-3))


def identity(meta):
    # Moving a complete dataset/cache/weights to another machine must not depend
    # on absolute file locations. Device/runtime/code changes still invalidate it.
    return {k: v for k, v in meta.items()
            if k not in ("recognition_checkpoint_path", "detector_checkpoint_path", "input_root")}


def visualize(image_path, regions, destination):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 600)
    for region in regions:
        points = [tuple(p) for p in region["bbox"]]
        draw.line(points + points[:1], fill=(255, 40, 20), width=line_width)
        x, y = points[0]
        label = str(region["region_id"])
        x, y = max(0, min(x, image.width-20)), max(0, min(y, image.height-15))
        draw.text((x, y), label, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 0))
    temp = destination.with_name(destination.name + ".tmp")
    image.save(temp, format="PNG")
    os.replace(temp, destination)


def process_image(pipeline, path, input_dir, output_dir, batch_size):
    sid = sample_id(path, input_dir)
    # Read grayscale once. Detector reads RGB from the path using EasyOCR.
    _, gray = reformat_input(str(path))
    if gray is None:
        raise ValueError("Cannot read input image")
    boxes = pipeline.detect(path)
    crops = [pipeline.crop(gray, box) for box in boxes]
    regions = [None] * len(boxes)
    original = {"features": [None] * len(boxes), "logits": [None] * len(boxes)}
    checks = [None] * len(boxes)
    with torch.no_grad():
        for i, raw, input_size in pipeline.recognize_many(crops, batch_size):
            decoded = pipeline.decode(raw.logits)
            checks[i] = validate_region(pipeline, crops[i], raw, decoded)
            polygon = np.asarray(boxes[i])
            regions[i] = dict(
                region_id=i, bbox=boxes[i],
                xyxy=[float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                      float(polygon[:, 0].max()), float(polygon[:, 1].max())],
                text=decoded.text, recognition_confidence=decoded.confidence,
                detection_confidence=None, feature_index=i, logits_index=i,
                feature_shape=list(raw.feature.shape[1:]),
                logits_shape=list(raw.logits.shape[1:]),
                crop_size=list(crops[i].shape), recognizer_input_size=input_size)
            original["features"][i] = raw.feature[0].detach().cpu().clone()
            original["logits"][i] = raw.logits[0].detach().cpu().clone()
    repeat = validate_repeat(pipeline, crops[0]) if crops else None
    gradient = validate_input_gradient(pipeline, crops[0]) if crops else None
    record = dict(sample_id=sid, image_path=input_dir.name + "/" + path.relative_to(input_dir).as_posix(),
                  image_sha256=sha256_file(path), width=int(gray.shape[1]), height=int(gray.shape[0]),
                  tensor_path="tensors/" + sid + ".pt",
                  visualization_path="visualizations/" + sid + "_bbox.png", regions=regions)
    payload = {key: [t.to(torch.float16) for t in tensors] for key, tensors in original.items()}
    payload.update(sample_id=sid, region_ids=list(range(len(regions))))
    target = output_dir / record["tensor_path"]
    temp = target.with_name(target.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, target)
    cache, loaded = validate_cache(record, output_dir, original)
    # Quantization can move a near-tied argmax. Preserve original-forward text,
    # and explicitly report whether decoding the stored fp16 logits changed it.
    cached_text_differences = []
    for i, logits in enumerate(loaded["logits"]):
        cached = pipeline.decode(logits.float().unsqueeze(0))
        if cached.text != regions[i]["text"]:
            cached_text_differences.append(dict(region_id=i, original=regions[i]["text"],
                                                cached=cached.text,
                                                reason="float16 logit rounding changed greedy argmax"))
    visualize(path, regions, output_dir / record["visualization_path"])
    record["tensor_sha256"] = sha256_file(target)
    record["visualization_sha256"] = sha256_file(output_dir / record["visualization_path"])
    record["validation"] = dict(regions=checks, deterministic=repeat, gradient=gradient,
                                cache=cache, cached_text_differences=cached_text_differences)
    record["annotation_sha256"] = record_checksum(record)
    return record


def make_report(paths, records, failures, resumed, output_dir, low_threshold):
    counts = [len(r["regions"]) for r in records]
    regions = [region for record in records for region in record["regions"]]
    confidences = [r["recognition_confidence"] for r in regions]
    def mean(values):
        return float(np.mean(values)) if values else None
    def extrema(values, fn):
        return fn(values) if values else None
    checks = [c for r in records for c in r["validation"]["regions"]]
    repeats = [r["validation"]["deterministic"] for r in records if r["validation"]["deterministic"] is not None]
    gradients = [r["validation"]["gradient"] for r in records if r["validation"]["gradient"] is not None]
    finite_counts = dict(feature_nan_count=0, feature_inf_count=0,
                         logits_nan_count=0, logits_inf_count=0)
    for record in records:
        _, payload = validate_cache(record, output_dir)
        for key, prefix in (("features", "feature"), ("logits", "logits")):
            for tensor in payload[key]:
                finite_counts[prefix + "_nan_count"] += int(torch.isnan(tensor).sum())
                finite_counts[prefix + "_inf_count"] += int(torch.isinf(tensor).sum())
    result = dict(
        nonfinite_count_scope="Successfully saved caches; failed images are listed separately",
        discovered_hr_canonical_images=len(paths), processed_images=len(records),
        newly_processed_images=len(records)-resumed, resumed_images=resumed,
        failed_images=len(failures), failures=failures, total_regions=len(regions),
        avg_regions_per_image=mean(counts), min_regions_per_image=extrema(counts, min),
        max_regions_per_image=extrema(counts, max),
        avg_recognition_confidence=mean(confidences),
        min_recognition_confidence=extrema(confidences, min), max_recognition_confidence=extrema(confidences, max),
        empty_text_count=sum(not r["text"] for r in regions),
        low_confidence_threshold=low_threshold,
        low_confidence_count=sum(c < low_threshold for c in confidences),
        feature_shape_distribution=dict(Counter(str(r["feature_shape"]) for r in regions)),
        logits_shape_distribution=dict(Counter(str(r["logits_shape"]) for r in regions)),
        deterministic_test_passed=all(r["passed"] for r in repeats) if repeats else None,
        gradient_test_passed=all(r["passed"] for r in gradients) if gradients else None,
        deterministic_test_count=len(repeats), gradient_test_count=len(gradients),
        old_new_test_count=len(checks),
        old_new_test_passed=all(c["old_new_passed"] for c in checks) if checks else None,
        probability_test_passed=all(c["probability_passed"] for c in checks) if checks else None,
        shape_test_passed=all(c["shape_passed"] for c in checks) if checks else None,
        head_test_passed=all(c["head_passed"] for c in checks) if checks else None,
        cached_tensor_test_passed=all(r["validation"]["cache"]["passed"] for r in records) if records else None,
        old_new_differences=[dict(sample_id=r["sample_id"], region_id=i, **c)
                             for r in records for i, c in enumerate(r["validation"]["regions"])
                             if not c["old_new_passed"]],
        cached_text_differences=[dict(sample_id=r["sample_id"], **d) for r in records
                                 for d in r["validation"]["cached_text_differences"]],
        images=[dict(sample_id=r["sample_id"], image_path=r["image_path"],
                     region_count=len(r["regions"]), visualization_path=r["visualization_path"])
                for r in records], **finite_counts)
    result["all_tests_passed"] = bool(paths and len(records) == len(paths) and not failures
                                     and all(result[k] is True for k in (
                                         "deterministic_test_passed", "gradient_test_passed",
                                         "old_new_test_passed", "probability_test_passed",
                                         "shape_test_passed", "head_test_passed",
                                         "cached_tensor_test_passed"))
                                     and not result["cached_text_differences"])
    result["status"] = "passed" if result["all_tests_passed"] else "failed_or_not_fully_exercised"
    return result


def build_annotations(pipeline, input_dir="./dataset", output_dir="./ocr_annotations",
                      filename_filter="hr_canonical.png", batch_size=16,
                      resume=False, low_confidence_threshold=0.1):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Output must be outside the source dataset")
    paths = discover_images(input_dir, filename_filter)
    print("Discovered images: " + str(len(paths)), flush=True)
    ids = [sample_id(p, input_dir) for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id collision")
    meta = teacher_metadata(pipeline, input_dir, filename_filter, batch_size)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError("Output is nonempty; use --resume or a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("tensors", "visualizations"):
        (output_dir / folder).mkdir(exist_ok=True)
    previous = {}
    meta_path = output_dir / "meta.json"
    jsonl_path = output_dir / "annotations.jsonl"
    if resume and meta_path.exists():
        if identity(json.loads(meta_path.read_text(encoding="utf-8"))) != identity(meta):
            raise ValueError("Teacher/config/source identity changed; use a new output directory")
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["sample_id"] in previous:
                    raise ValueError("Duplicate sample_id in annotations.jsonl")
                previous[record["sample_id"]] = record
    elif resume and jsonl_path.exists():
        raise ValueError("Cannot resume annotations without meta.json")
    atomic_json(meta_path, meta)
    records, failures, resumed = [], [], 0
    for position, (path, sid) in enumerate(zip(paths, ids)):
        try:
            old = previous.get(sid)
            if old is not None:
                try:
                    if old["annotation_sha256"] != record_checksum(old):
                        raise ValueError("Annotation checksum changed")
                    if (old["tensor_path"] != "tensors/" + sid + ".pt"
                            or old["visualization_path"] != "visualizations/" + sid + "_bbox.png"):
                        raise ValueError("Unexpected artifact path")
                    if old["image_sha256"] != sha256_file(path):
                        raise ValueError("Source image changed")
                    if old["tensor_sha256"] != sha256_file(output_dir / old["tensor_path"]):
                        raise ValueError("Tensor checksum changed")
                    if old["visualization_sha256"] != sha256_file(output_dir / old["visualization_path"]):
                        raise ValueError("Visualization checksum changed")
                    validate_cache(old, output_dir)
                    # A failed validation is regenerated, never counted as completed.
                    if (any(not all(c[k] for k in ("shape_passed", "probability_passed", "head_passed", "old_new_passed"))
                            for c in old["validation"]["regions"])
                            or any(old["validation"][k] is not None and not old["validation"][k]["passed"]
                                   for k in ("deterministic", "gradient"))
                            or old["validation"]["cached_text_differences"]):
                        raise ValueError("Previous acceptance checks failed")
                except (OSError, ValueError, KeyError, RuntimeError):
                    old = None
            record = old if old is not None else process_image(pipeline, path, input_dir, output_dir, batch_size)
            resumed += int(old is not None)
            records.append(record)
            print(sid + ": " + str(len(record["regions"])) + " regions" +
                  (" (resumed)" if old is not None else ""), flush=True)
        except Exception as error:
            failures.append(dict(sample_id=sid, image_path=path.relative_to(input_dir).as_posix(),
                                 error=type(error).__name__ + ": " + str(error)))
            print(sid + ": FAILED: " + str(error), flush=True)
        # Preserve unvisited entries if interrupted during a resumed run.
        current = {r["sample_id"]: r for r in records}
        pending = [previous[i] for i in ids[position+1:] if i in previous]
        write_jsonl(jsonl_path, list(current.values()) + pending)
    if not paths:
        write_jsonl(jsonl_path, [])
    report = make_report(paths, records, failures, resumed, output_dir, low_confidence_threshold)
    atomic_json(output_dir / "validation_report.json", report)
    return report
