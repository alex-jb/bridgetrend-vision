import numpy as np
import pytest

from bridgetrend_vision.quality import (
    QualityEvidence,
    analyze_image,
    opencv_major_version,
)


def test_quality_score_rewards_resolution_and_sharpness():
    strong = QualityEvidence(
        width=800,
        height=800,
        blur_variance=500.0,
        mean_luminance=128.0,
        clipped_dark_fraction=0.01,
        clipped_light_fraction=0.01,
    )
    weak = QualityEvidence(
        width=120,
        height=120,
        blur_variance=5.0,
        mean_luminance=20.0,
        clipped_dark_fraction=0.50,
        clipped_light_fraction=0.0,
    )
    assert strong.score > weak.score
    assert strong.score == pytest.approx(0.997, abs=0.01)


def test_quality_evidence_serializes_computed_score():
    evidence = QualityEvidence(384, 384, 250.0, 127.5, 0.0, 0.0)
    assert evidence.to_dict()["score"] == pytest.approx(1.0)


def test_opencv5_analyzes_image_without_model_download():
    image = np.full((40, 60, 3), 127, dtype=np.uint8)
    image[10:30, 20:40] = 255
    evidence = analyze_image(image)

    assert opencv_major_version() >= 5
    assert evidence.width == 60
    assert evidence.height == 40
    assert evidence.blur_variance > 0
    assert 0.0 <= evidence.score <= 1.0
