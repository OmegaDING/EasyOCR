"""Offline contract tests. Official-weight/data acceptance runs via the CLI."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from easyocr import annotation_builder as builder
from easyocr.annotation_validation import validate_input_gradient, validate_region, validate_repeat
from easyocr.model.model import Model as Generation1
from easyocr.model.vgg_model import Model as Generation2
from easyocr.ocr_teacher import (CTCDecoder, OCRPipeline, OCRRecognizer,
                                 TextDetector, configure_determinism)
from easyocr.utils import CTCLabelConverter


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    configure_determinism()


@pytest.fixture
def pipeline():
    model = Generation2(input_channel=1, output_channel=32, hidden_size=8, num_class=3)
    result = OCRPipeline.__new__(OCRPipeline)
    result.recognizer = OCRRecognizer(model)
    converter = CTCLabelConverter("ab")
    result.reader = SimpleNamespace(character="ab", recognizer=model, converter=converter)
    result.decoder = CTCDecoder(converter, [])
    result.imgH = 64
    return result


@pytest.mark.parametrize("model_class", [Generation1, Generation2])
def test_real_feature_boundary_and_backward_compatibility(model_class):
    model = model_class(input_channel=1, output_channel=32, hidden_size=8, num_class=4).eval()
    state_keys = set(model.state_dict())
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        visual = model.FeatureExtraction(x)
        visual = model.AdaptiveAvgPool(visual.permute(0, 3, 1, 2)).squeeze(3)
        expected_f = model.SequenceModeling(visual)
        expected_z = model.Prediction(expected_f.contiguous())
        f, z = model(x, None, return_features=True)
        assert torch.equal(f, expected_f)
        assert torch.equal(z, expected_z)
        assert torch.equal(model(x, torch.zeros(2, 5, dtype=torch.long)), expected_z)
    assert state_keys == set(model.state_dict())
    teacher = OCRRecognizer(model)
    x.requires_grad_(True)
    raw = teacher(x)
    (raw.feature.abs().mean() + raw.logits.abs().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_ctc_blank_repeats_and_language_mask():
    converter = CTCLabelConverter("ab")
    decoder = CTCDecoder(converter, [])
    # a,a,blank,a,b,b -> aab (blank separates repeated a).
    logits = torch.full((1, 6, 3), -8.)
    for t, index in enumerate([1, 1, 0, 1, 2, 2]):
        logits[0, t, index] = 8
    decoded = decoder(logits)[0]
    assert decoded.text == "aab"
    assert 0.99 < decoded.confidence <= 1
    blank = decoder(torch.tensor([[[20., 0., 0.]]]))[0]
    assert blank.text == "" and blank.confidence == 0
    masked = CTCDecoder(converter, [2])(torch.tensor([[[0., 5., 10.]]]))[0]
    assert masked.text == "a"


def test_preprocessing_original_consistency_and_repeat(pipeline):
    crop = np.random.default_rng(7).integers(0, 255, (30, 90), dtype=np.uint8)
    with torch.no_grad():
        raw = pipeline.recognize_raw(crop)
        decoded = pipeline.decode(raw.logits)
        checks = validate_region(pipeline, crop, raw, decoded)
    assert all(checks[k] for k in ("shape_passed", "head_passed", "probability_passed", "old_new_passed"))
    assert validate_repeat(pipeline, crop)["passed"]
    assert validate_input_gradient(pipeline, crop)["passed"]


def test_equal_width_batch_is_independent_of_other_crop_widths(pipeline):
    crops = [np.full((30, 90), 80, np.uint8), np.full((30, 90), 180, np.uint8),
             np.full((30, 170), 100, np.uint8), np.full((90, 30), 110, np.uint8)]
    with torch.no_grad():
        outputs = {i: raw for i, raw, _ in pipeline.recognize_many(crops, batch_size=2)}
        for i, crop in enumerate(crops):
            single = pipeline.recognize_raw(crop)
            assert outputs[i].feature.shape == single.feature.shape
            assert torch.allclose(outputs[i].feature, single.feature, atol=1e-6, rtol=1e-5)
            assert torch.allclose(outputs[i].logits, single.logits, atol=1e-6, rtol=1e-5)


def test_craft_polygon_preserved(monkeypatch):
    import easyocr.ocr_teacher as module
    polygon = np.array([[10.5, 5], [80, 15], [77, 38], [8, 28]], np.float32)
    monkeypatch.setattr(module, "test_net", lambda **kwargs: ([[polygon]], [[polygon]]))
    reader = SimpleNamespace(detector=object(), device="cpu")
    boxes = TextDetector(reader)(np.zeros((100, 100, 3), np.uint8))
    assert boxes == [polygon.tolist()]


def test_filename_filter_and_ids(tmp_path):
    for name in ("a/b/hr_canonical.png", "a__b/hr_canonical.png", "hr_canonical.png",
                 "other/HR_CANONICAL.PNG", "a/b/lr.png", "a/b/hr.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    paths = builder.discover_images(tmp_path)
    assert len(paths) == 3 and all(p.name == "hr_canonical.png" for p in paths)
    ids = [builder.sample_id(p, tmp_path) for p in paths]
    assert len(set(ids)) == 3
    assert ids == [builder.sample_id(Path("/relocated") / p.relative_to(tmp_path), Path("/relocated"))
                   for p in paths]


def test_builder_cache_resume_corruption_and_empty_image(tmp_path, pipeline, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "annotations"
    (dataset / "a").mkdir(parents=True)
    (dataset / "b").mkdir()
    img = np.random.default_rng(7).integers(0, 255, (64, 192), dtype=np.uint8)
    Image.fromarray(img).save(dataset / "a" / "hr_canonical.png")
    Image.fromarray(img).save(dataset / "b" / "hr_canonical.png")
    Image.fromarray(img).save(dataset / "a" / "lr.png")
    calls = []
    def detect(path):
        calls.append(path)
        return [] if path.parent.name == "b" else [[[1, 1], [90, 1], [90, 31], [1, 31]],
                                                   [[95, 5], [180, 5], [180, 40], [95, 40]]]
    pipeline.detect = detect
    monkeypatch.setattr(builder, "teacher_metadata", lambda *args: {"test_teacher": 1})
    report = builder.build_annotations(pipeline, dataset, output, batch_size=2)
    assert report["all_tests_passed"]
    assert report["processed_images"] == 2 and report["total_regions"] == 2
    records = [json.loads(line) for line in (output / "annotations.jsonl").read_text().splitlines()]
    assert [len(r["regions"]) for r in records] == [2, 0]
    with pytest.raises(FileExistsError):
        builder.build_annotations(pipeline, dataset, output)
    calls.clear()
    resumed = builder.build_annotations(pipeline, dataset, output, batch_size=2, resume=True)
    assert resumed["resumed_images"] == 2 and not calls
    target = output / records[0]["tensor_path"]
    payload = torch.load(target, weights_only=True)
    payload["features"][0][0, 0] += 1
    torch.save(payload, target)
    resumed = builder.build_annotations(pipeline, dataset, output, batch_size=2, resume=True)
    assert resumed["resumed_images"] == 1 and len(calls) == 1
    assert resumed["all_tests_passed"]
    calls.clear()
    records = [json.loads(line) for line in (output / "annotations.jsonl").read_text().splitlines()]
    records[0]["regions"][0]["text"] = "corrupted annotation"
    builder.write_jsonl(output / "annotations.jsonl", records)
    repaired = builder.build_annotations(pipeline, dataset, output, batch_size=2, resume=True)
    assert repaired["resumed_images"] == 1 and len(calls) == 1
    assert repaired["all_tests_passed"]
    monkeypatch.setattr(builder, "teacher_metadata", lambda *args: {"test_teacher": 2})
    with pytest.raises(ValueError, match="identity"):
        builder.build_annotations(pipeline, dataset, output, resume=True)


def test_all_empty_is_not_a_false_validation_pass(tmp_path, pipeline, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("L", (64, 64)).save(dataset / "hr_canonical.png")
    pipeline.detect = lambda path: []
    monkeypatch.setattr(builder, "teacher_metadata", lambda *args: {"test_teacher": 1})
    report = builder.build_annotations(pipeline, dataset, tmp_path / "annotations")
    assert report["processed_images"] == 1
    assert report["gradient_test_passed"] is None
    assert not report["all_tests_passed"]
