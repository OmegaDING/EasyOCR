"""Composable OCR teacher. NumPy images use EasyOCR's BGR convention.

F is the actual final SequenceModeling output (including the projection inside
EasyOCR's BidirectionalLSTM), not the unprojected recurrent state. Z is raw
Prediction(F), before language masking or softmax.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from .config import imgH
from .detection import test_net
from .easyocr import Reader
from .recognition import AlignCollate, custom_mean
from .utils import compute_ratio_and_resize, four_point_transform, reformat_input


@dataclass
class OCRForwardOutput:
    feature: torch.Tensor  # [B, T, D], attached to input autograd graph
    logits: torch.Tensor  # [B, T, C], including blank at index zero


@dataclass
class OCRDecodedOutput:
    text: str
    confidence: float


def configure_determinism():
    # Must be set before initializing CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class RecognitionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode(image)


class CharacterPredictionHead(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.linear = prediction

    def forward(self, feature):
        return self.linear(feature.contiguous())


class OCRRecognizer(nn.Module):
    """Frozen differentiable teacher for normalized [B, 1, H, W] tensors.

    Preprocessing NumPy/PIL crops is intentionally separate. This forward never
    calls no_grad(), inference_mode(), detach(), or a decoder.
    """
    def __init__(self, model):
        super().__init__()
        if isinstance(model, nn.DataParallel):
            model = model.module
        if not hasattr(model, "encode") or not isinstance(model.Prediction, nn.Linear):
            raise TypeError("A non-quantized built-in generation1/2 model is required")
        self.encoder = RecognitionEncoder(model)
        self.head = CharacterPredictionHead(model.Prediction)
        self.feature_dim = model.Prediction.in_features
        self.num_classes = model.Prediction.out_features
        self.requires_grad_(False)
        self.eval()

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected normalized [B, 1, H, W] image tensor")
        # cuDNN RNN backward is unavailable after an eval-mode cuDNN forward.
        # Native LSTM preserves eval semantics and input gradients on CUDA.
        if image.is_cuda and torch.is_grad_enabled() and image.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                feature = self.encoder(image)
        else:
            feature = self.encoder(image)
        return OCRForwardOutput(feature, self.head(feature))


class CTCDecoder:
    """Greedy CTC with the original EasyOCR language mask and custom_mean."""
    def __init__(self, converter, ignore_indices):
        self.converter = converter
        self.ignore_indices = list(ignore_indices)

    def __call__(self, logits):
        if logits.ndim != 3 or logits.shape[-1] != len(self.converter.character):
            raise ValueError("Expected [B, T, C] raw logits matching the charset")
        # Match recognizer_predict's softmax -> NumPy mask -> renormalize path.
        probs = logits.detach().float().softmax(-1).cpu().numpy()
        probs[:, :, self.ignore_indices] = 0.
        probs = probs / probs.sum(axis=2, keepdims=True)
        indices = probs.argmax(2)
        lengths = torch.IntTensor([logits.shape[1]] * logits.shape[0])
        texts = self.converter.decode_greedy(indices.reshape(-1), lengths)
        values = probs.max(2)
        outputs = []
        for text, value, index in zip(texts, values, indices):
            nonblank = value[index != 0]
            if not len(nonblank):
                nonblank = np.array([0])
            outputs.append(OCRDecodedOutput(text, float(custom_mean(nonblank))))
        return outputs


class TextDetector:
    """CRAFT's original quadrilaterals in image pixels, without line merging."""
    def __init__(self, reader, **thresholds):
        self.reader = reader
        self.config = dict(canvas_size=2560, mag_ratio=1.,
                           text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        unknown = set(thresholds) - set(self.config)
        if unknown:
            raise ValueError("Unknown CRAFT options: " + str(sorted(unknown)))
        self.config.update(thresholds)

    def __call__(self, image):
        color, _ = reformat_input(str(image) if isinstance(image, Path) else image)
        boxes, _ = test_net(net=self.reader.detector, image=color, poly=False,
                            device=self.reader.device, **self.config)
        polygons = [np.asarray(box, dtype=np.float32).reshape(4, 2)
                    for box in boxes[0]]
        # Stable region IDs; preserve CRAFT's point order and floating coordinates.
        polygons.sort(key=lambda p: (float(p[:, 1].min()), float(p[:, 0].min()),
                                     tuple(p.reshape(-1))))
        return [p.tolist() for p in polygons]


class OCRPipeline:
    def __init__(self, languages=None, gpu=True, model_storage_directory=None,
                 download_enabled=True, **detector_thresholds):
        languages = list(languages or ["ch_sim", "en"])
        if languages != ["ch_sim", "en"]:
            raise ValueError("Phase 1 teacher requires languages=['ch_sim', 'en']")
        configure_determinism()
        if gpu is False or str(gpu).lower() == "cpu":
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda" if gpu is True else str(gpu)
            if device.isdigit():
                device = "cuda:" + device
        else:
            device = "cpu"
        self.languages = languages
        self.reader = Reader(languages, gpu=device, detect_network="craft",
                             quantize=False, cudnn_benchmark=False,
                             model_storage_directory=model_storage_directory,
                             user_network_directory=(str(Path(model_storage_directory) / "user_network")
                                                     if model_storage_directory else None),
                             download_enabled=download_enabled)
        if isinstance(self.reader.detector, nn.DataParallel):
            self.reader.detector = self.reader.detector.module
        self.reader.detector.eval().requires_grad_(False)
        self.recognizer = OCRRecognizer(self.reader.recognizer)
        # Single-device teacher; avoid DataParallel's implicit cuda:0 device IDs.
        self.reader.recognizer = self.recognizer.encoder.model
        self.detector = TextDetector(self.reader, **detector_thresholds)
        ignore = [i + 1 for i, c in enumerate(self.reader.character)
                  if c not in set(self.reader.lang_char)]
        self.decoder = CTCDecoder(self.reader.converter, ignore)
        self.imgH = imgH

    @property
    def device(self):
        return next(self.recognizer.parameters()).device

    def detect(self, image):
        return self.detector(image)

    def crop(self, image, box):
        _, gray = reformat_input(str(image) if isinstance(image, Path) else image)
        polygon = np.asarray(box, dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError("bbox must be a finite [4, 2] polygon")
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1:
            raise ValueError("Degenerate/non-convex CRAFT polygon")
        crop = four_point_transform(gray, polygon)
        if crop.size == 0:
            raise ValueError("Empty rectified crop")
        return crop

    def prepare_crop(self, crop):
        """Match get_image_list's free-polygon path after four_point_transform.

        Width is derived ONLY from this crop, never other batch members.
        The intermediate OpenCV resize is retained for original-path parity.
        """
        if crop.ndim != 2 or crop.dtype != np.uint8 or not crop.size:
            raise ValueError("Expected a nonempty uint8 grayscale crop")
        height, width = crop.shape
        resized, ratio = compute_ratio_and_resize(crop, width, height, self.imgH)
        padded_width = math.ceil(max(1., ratio)) * self.imgH
        return resized, padded_width

    def preprocess(self, crop):
        resized, width = self.prepare_crop(crop)
        tensor = AlignCollate(imgH=self.imgH, imgW=width,
                              keep_ratio_with_pad=True, adjust_contrast=0.)(
                                  [Image.fromarray(resized)])
        return tensor.to(device=self.device,
                         dtype=next(self.recognizer.parameters()).dtype)

    def recognize_raw(self, crop):
        return self.recognizer(self.preprocess(crop))

    def recognize_many(self, crops, batch_size=16):
        """Yield (original_index, raw, input_size) using equal-width buckets.

        No cross-width padding; therefore region T and supervision do not depend
        on batch composition. Results carry original indices for safe assembly.
        Call under torch.no_grad() for offline annotation.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        buckets = {}
        for index, crop in enumerate(crops):
            resized, width = self.prepare_crop(crop)
            buckets.setdefault(width, []).append((index, resized))
        for width, items in sorted(buckets.items()):
            collate = AlignCollate(imgH=self.imgH, imgW=width,
                                   keep_ratio_with_pad=True, adjust_contrast=0.)
            for offset in range(0, len(items), batch_size):
                chunk = items[offset:offset + batch_size]
                tensor = collate([Image.fromarray(img) for _, img in chunk]).to(
                    device=self.device, dtype=next(self.recognizer.parameters()).dtype)
                raw = self.recognizer(tensor)
                for row, (index, _) in enumerate(chunk):
                    yield index, OCRForwardOutput(raw.feature[row:row+1],
                                                  raw.logits[row:row+1]), [self.imgH, width]

    def decode_batch(self, logits):
        return self.decoder(logits)

    def decode(self, logits):
        outputs = self.decode_batch(logits)
        if len(outputs) != 1:
            raise ValueError("Use decode_batch for B != 1")
        return outputs[0]
