import pandas as pd

from bridgetrend_vision.retrieval_protocol import (
    build_identity_judgments,
    iter_query_galleries,
)


def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "q1",
                "market": "GLOBAL",
                "category": "mouse",
                "split": "validation",
                "query_eligible": True,
                "evaluation_track": "retrieval_calibration",
                "product_family_id": "family-a",
            },
            {
                "image_id": "a2",
                "market": "GLOBAL",
                "category": "mouse",
                "split": "validation",
                "query_eligible": False,
                "evaluation_track": "retrieval_calibration",
                "product_family_id": "family-a",
            },
            {
                "image_id": "b1",
                "market": "GLOBAL",
                "category": "mouse",
                "split": "validation",
                "query_eligible": False,
                "evaluation_track": "retrieval_calibration",
                "product_family_id": "family-b",
            },
            {
                "image_id": "test1",
                "market": "GLOBAL",
                "category": "mouse",
                "split": "test",
                "query_eligible": False,
                "evaluation_track": "retrieval_calibration",
                "product_family_id": "family-c",
            },
        ]
    )


def test_global_calibration_gallery_stays_inside_split_and_excludes_query():
    pairs = iter_query_galleries(metadata(), mode="global_calibration")

    assert len(pairs) == 1
    query_index, gallery_indices = pairs[0]
    assert query_index == 0
    assert gallery_indices.tolist() == [1, 2]


def test_identity_judgments_assign_l3_only_to_same_product_family():
    judgments = build_identity_judgments(metadata())

    relevance = dict(zip(judgments["match_id"], judgments["relevance"]))
    assert relevance == {"a2": 3, "b1": 0}
