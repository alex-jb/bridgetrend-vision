"""OpenCV 5 preprocessing and product-image quality evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityEvidence:
    """Interpretable measurements produced before embedding an image."""

    width: int
    height: int
    blur_variance: float
    mean_luminance: float
    clipped_dark_fraction: float
    clipped_light_fraction: float

    @property
    def score(self) -> float:
        """Conservative 0-1 quality score used by the evidence agent."""

        resolution = min(1.0, min(self.width, self.height) / 384.0)
        sharpness = min(1.0, self.blur_variance / 250.0)
        exposure_penalty = min(
            1.0, self.clipped_dark_fraction + self.clipped_light_fraction
        )
        luminance_balance = 1.0 - min(1.0, abs(self.mean_luminance - 127.5) / 127.5)
        raw = (
            0.30 * resolution
            + 0.35 * sharpness
            + 0.20 * luminance_balance
            + 0.15 * (1.0 - exposure_penalty)
        )
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


def opencv_major_version() -> int:
    """Return the installed OpenCV major version using a lazy import."""

    import cv2

    return int(cv2.__version__.split(".", maxsplit=1)[0])


def require_opencv5() -> None:
    """Fail clearly when the competition runtime is not using OpenCV 5."""

    major = opencv_major_version()
    if major < 5:
        raise RuntimeError(f"OpenCV 5 or newer is required; found major version {major}")


def analyze_image(image_bgr: np.ndarray) -> QualityEvidence:
    """Measure sharpness, exposure, and resolution for a BGR image."""

    import cv2

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape [height, width, 3]")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must use uint8 pixels")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return QualityEvidence(
        width=int(width),
        height=int(height),
        blur_variance=blur_variance,
        mean_luminance=float(gray.mean()),
        clipped_dark_fraction=float(np.mean(gray <= 5)),
        clipped_light_fraction=float(np.mean(gray >= 250)),
    )


def load_and_prepare(
    path: str | Path,
    *,
    output_size: tuple[int, int] = (384, 384),
    apply_clahe: bool = True,
    enforce_opencv5: bool = True,
) -> tuple[np.ndarray, QualityEvidence]:
    """Load, normalize contrast, and letterbox a product image.

    The returned image is BGR so downstream OpenCV stages can consume it
    without another color conversion.  The evidence is computed on the source
    image so padding cannot inflate the quality score.
    """

    import cv2

    if enforce_opencv5:
        require_opencv5()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    image = _to_bgr(image)
    evidence = analyze_image(image)

    if apply_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        image = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
        )

    target_width, target_height = output_size
    if target_width < 1 or target_height < 1:
        raise ValueError("output_size values must be positive")
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )
    left = (target_width - resized_width) // 2
    right = target_width - resized_width - left
    top = (target_height - resized_height) // 2
    bottom = target_height - resized_height - top
    prepared = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return prepared, evidence


def _to_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("decoded image must have two or three dimensions")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        foreground = image[:, :, :3].astype(np.float32)
        white = np.full_like(foreground, 255.0)
        return np.rint(foreground * alpha + white * (1.0 - alpha)).astype(np.uint8)
    raise ValueError("decoded image must have 1, 3, or 4 channels")
