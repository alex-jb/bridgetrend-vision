"""OpenCLIP encoder wrapper with lazy heavyweight imports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


class OpenCLIPEncoder:
    """Encode product images and text with a pretrained OpenCLIP model."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = f"open_clip:{model_name}:{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(
        self, image_paths: Iterable[str | Path], batch_size: int = 32
    ) -> np.ndarray:
        from PIL import Image

        paths = [Path(path) for path in image_paths]
        outputs: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            outputs.append(features.cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
