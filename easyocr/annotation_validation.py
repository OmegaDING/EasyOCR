"""Runtime acceptance checks; no synthetic success values in reports."""
import numpy as np
import torch

from .recognition import get_text

OLD_NEW_CONFIDENCE_ATOL = 5e-5


def validate_region(pipeline, crop, raw, decoded):
    f, z = raw.feature, raw.logits
    if f.ndim != 3 or z.ndim != 3 or f.shape[:2] != z.shape[:2]:
        raise ValueError("F/Z batch or time axes do not match")
    if f.shape[-1] != pipeline.recognizer.feature_dim:
        raise ValueError("F dimension differs from Prediction.in_features")
    if z.shape[-1] != pipeline.recognizer.num_classes:
        raise ValueError("Z dimension differs from Prediction.out_features")
    if not torch.isfinite(f).all() or not torch.isfinite(z).all():
        raise ValueError("Non-finite forward output")
    with torch.no_grad():
        head_ok = torch.allclose(pipeline.recognizer.head(f), z, rtol=1e-5, atol=1e-6)
        probability_ok = torch.allclose(z.float().softmax(-1).sum(-1),
                                       torch.ones_like(z[..., 0]), rtol=1e-5, atol=1e-6)
    resized, width = pipeline.prepare_crop(crop)
    ignore_char = "".join(pipeline.reader.character[i-1]
                          for i in pipeline.decoder.ignore_indices)
    # Original get_text/recognizer_predict and original model default forward.
    # Identical intermediate crop, width, language mask, no second pass.
    old = get_text(pipeline.reader.character, pipeline.imgH, width,
                   pipeline.reader.recognizer, pipeline.reader.converter,
                   [(None, resized)], ignore_char=ignore_char,
                   decoder="greedy", batch_size=1, contrast_ths=0.,
                   adjust_contrast=0., workers=0, device=str(pipeline.device))[0]
    delta = abs(float(old[2]) - decoded.confidence)
    text_passed = old[1] == decoded.text
    confidence_passed = delta <= OLD_NEW_CONFIDENCE_ATOL
    if not text_passed:
        reason = "CTC text differs with identical preprocessing and language mask."
    elif not confidence_passed:
        reason = ("Confidence exceeds the permitted single-vs-equal-width-batch "
                  "CUDA numerical tolerance.")
    else:
        reason = None
    return dict(shape_passed=True, probability_passed=bool(probability_ok),
                head_passed=bool(head_ok), text_old=old[1], text_new=decoded.text,
                text_passed=text_passed, confidence_old=float(old[2]),
                confidence_new=float(decoded.confidence), confidence_abs_error=delta,
                confidence_atol=OLD_NEW_CONFIDENCE_ATOL,
                confidence_passed=confidence_passed,
                old_new_passed=(text_passed and confidence_passed),
                difference_reason=reason)


def validate_input_gradient(pipeline, crop):
    with torch.enable_grad():
        fake_sr = pipeline.preprocess(crop).detach().requires_grad_(True)
        raw = pipeline.recognizer(fake_sr)
        loss = raw.feature.abs().mean() + raw.logits.abs().mean()
        loss.backward()
        grad = fake_sr.grad
        return dict(passed=bool(grad is not None and torch.isfinite(grad).all()
                                and grad.abs().sum() > 0
                                and all(p.grad is None and not p.requires_grad
                                        for p in pipeline.recognizer.parameters())),
                    input_grad_abs_sum=float(grad.abs().sum()) if grad is not None else None,
                    input_shape=list(fake_sr.shape), device=str(fake_sr.device))


def validate_repeat(pipeline, crop):
    with torch.no_grad():
        first = pipeline.recognize_raw(crop)
        second = pipeline.recognize_raw(crop)
    f_ok = torch.allclose(first.feature, second.feature, rtol=1e-5, atol=1e-6)
    z_ok = torch.allclose(first.logits, second.logits, rtol=1e-5, atol=1e-6)
    return dict(passed=bool(f_ok and z_ok),
                feature_max_abs_error=float((first.feature-second.feature).abs().max()),
                logits_max_abs_error=float((first.logits-second.logits).abs().max()))


def validate_cache(record, output_dir, original=None):
    from pathlib import Path
    payload = torch.load(Path(output_dir) / record["tensor_path"],
                         map_location="cpu", weights_only=True)
    count = len(record["regions"])
    if payload.get("sample_id") != record["sample_id"]:
        raise ValueError("Cache sample_id mismatch")
    if payload.get("region_ids") != list(range(count)):
        raise ValueError("Cache region IDs mismatch")
    if len(payload["features"]) != count or len(payload["logits"]) != count:
        raise ValueError("Cache region count mismatch")
    max_errors = {"features": 0., "logits": 0.}
    for i, region in enumerate(record["regions"]):
        for key, index_key, shape_key in (("features", "feature_index", "feature_shape"),
                                           ("logits", "logits_index", "logits_shape")):
            if region[index_key] != i:
                raise ValueError("Region/tensor index mismatch")
            tensor = payload[key][i]
            if (tensor.ndim != 2 or list(tensor.shape) != region[shape_key]
                    or tensor.dtype != torch.float16 or tensor.requires_grad
                    or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid cached " + key)
            if original is not None:
                source = original[key][i]
                if not torch.equal(tensor, source.to(torch.float16)):
                    raise ValueError("Saved tensor differs from pre-save fp16 tensor")
                error = float((tensor.float() - source.float()).abs().max())
                max_errors[key] = max(error, max_errors[key])
                if not torch.allclose(tensor.float(), source.float(), rtol=1e-3, atol=1e-3):
                    raise ValueError("FP16 roundtrip exceeded tolerance")
        if payload["features"][i].shape[0] != payload["logits"][i].shape[0]:
            raise ValueError("Cached F/Z time dimensions mismatch")
        if not np.isfinite(region["recognition_confidence"]):
            raise ValueError("Non-finite confidence")
    return dict(passed=True, max_abs_errors=max_errors), payload
