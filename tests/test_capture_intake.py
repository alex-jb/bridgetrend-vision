import csv
from pathlib import Path

import pytest
from PIL import Image

from bridgetrend_vision.capture_intake import (
    BAIDU_CAPTURE_INBOX_FIELDS,
    audit_baidu_capture_inbox,
    summarize_baidu_capture_audit,
    write_baidu_capture_audit,
)

CONCEPTS = frozenset({"BT-C001", "BT-C002"})


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "capture_id": "baidu-001",
        "received_on": "2026-09-17",
        "source_id": "baidu_index_screenshot",
        "market": "CN",
        "related_concept_id": "BT-C001",
        "query": "盲盒",
        "query_role": "broad_proxy",
        "surface": "search_index",
        "displayed_start": "2021-08-20",
        "displayed_end": "2026-09-13",
        "displayed_granularity": "week_visible",
        "query_geo": "CN_national",
        "device_scope": "PC+mobile",
        "raw_file_name": "capture.png",
        "mapping_status": "mapped_proxy",
        "exclusion_reason": "Screenshot is not a machine-readable series.",
        "notes": "Broad concept proxy.",
    }
    row.update(overrides)
    return row


def _write_inbox(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BAIDU_CAPTURE_INBOX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _create_image(path: Path) -> None:
    Image.new("RGB", (32, 24), "white").save(path)


def test_screenshot_audit_hashes_but_never_promotes_pixels(tmp_path: Path):
    capture = tmp_path / "capture.png"
    _create_image(capture)
    inbox = tmp_path / "inbox.csv"
    _write_inbox(inbox, [_row()])

    rows = audit_baidu_capture_inbox(inbox, tmp_path, allowed_concept_ids=CONCEPTS)
    report = summarize_baidu_capture_audit(rows)

    assert len(rows[0]["raw_file_sha256"]) == 64
    assert rows[0]["width_px"] == "32"
    assert rows[0]["height_px"] == "24"
    assert rows[0]["machine_readable_points"] == "false"
    assert rows[0]["history_eligible"] == "false"
    assert rows[0]["claim_eligible"] == "false"
    assert report["capture_count"] == 1
    assert report["claim_ready"] is False


def test_unregistered_discovery_must_not_claim_registry_mapping(tmp_path: Path):
    _create_image(tmp_path / "capture.png")
    inbox = tmp_path / "inbox.csv"
    _write_inbox(
        inbox,
        [
            _row(
                related_concept_id="BT-C001",
                query="捏捏",
                query_role="unregistered_discovery",
                mapping_status="unregistered",
            )
        ],
    )

    with pytest.raises(ValueError, match="cannot name a concept"):
        audit_baidu_capture_inbox(inbox, tmp_path, allowed_concept_ids=CONCEPTS)


def test_capture_file_name_cannot_escape_private_directory(tmp_path: Path):
    _create_image(tmp_path / "capture.png")
    inbox = tmp_path / "inbox.csv"
    _write_inbox(inbox, [_row(raw_file_name="../capture.png")])

    with pytest.raises(ValueError, match="must be a basename"):
        audit_baidu_capture_inbox(inbox, tmp_path, allowed_concept_ids=CONCEPTS)


def test_audit_writer_emits_sanitized_exact_contract(tmp_path: Path):
    _create_image(tmp_path / "capture.png")
    inbox = tmp_path / "inbox.csv"
    output = tmp_path / "audit.csv"
    _write_inbox(inbox, [_row()])
    rows = audit_baidu_capture_inbox(inbox, tmp_path, allowed_concept_ids=CONCEPTS)

    write_baidu_capture_audit(output, rows)

    text = output.read_text(encoding="utf-8")
    assert "private_raw_not_git" in text
    assert str(tmp_path) not in text
