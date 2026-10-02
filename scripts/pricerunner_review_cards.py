"""Private, title-only cards for the frozen PriceRunner variant audit.

This tool never samples offers, makes variant judgments, or contacts reviewers.
The curator must independently verify each external seal anchor before use.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from pricerunner_variant_sampler import canonical_json, priority, require_manifest_outside_repo


PROJECT = Path(__file__).resolve().parents[1]
LABELS = {"SAME_EXACT", "DIFFERENT_VARIANT", "INDETERMINATE"}
ATTRIBUTES = {"brand", "model/revision", "color", "size", "capacity",
              "bundle/pack count", "region", "other"}
REASONS = {"unspecified variant", "conflicting tokens",
           "non-product/garbled title", "other"}


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Missing UTC timestamp in lock record")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid UTC timestamp in lock record") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("Lock timestamp must be UTC")
    return parsed


def _read_json(path: Path) -> tuple[bytes, dict]:
    require_manifest_outside_repo(path, PROJECT)
    raw = path.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return raw, obj


def _anchor(record: dict, time_key: str) -> datetime:
    url = record.get("external_anchor_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Externally anchored lock record required")
    sealed_at = _utc(record.get(time_key))
    if sealed_at >= datetime.now(timezone.utc):
        raise ValueError("Lock timestamp must precede card generation")
    # The URL's content and timestamp need independent curator verification.
    return sealed_at


def load_sealed_manifest(manifest_path: Path, seal_path: Path) -> tuple[dict, str]:
    """Verify actual bytes against a separate recorded seal before exposing titles."""
    _, seal = _read_json(seal_path)
    _anchor(seal, "sealed_at_utc")
    expected = seal.get("manifest_sha256")
    if not _digest(expected):
        raise ValueError("Malformed manifest seal")
    require_manifest_outside_repo(manifest_path, PROJECT)
    raw = manifest_path.read_bytes()
    if _sha256(raw) != expected:
        raise ValueError("Manifest bytes differ from prior seal")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unexpected selected-pair manifest")
    if (seal.get("selection_commit") != manifest.get("selection_commit") or
            seal.get("protocol_sha256") != manifest.get("protocol_sha256")):
        raise ValueError("Manifest metadata differs from prior seal")
    selected = manifest.get("selected_pairs")
    if not isinstance(selected, list) or len(selected) > 100:
        raise ValueError("Unexpected selected-pair count")
    audit_ids = set()
    for pair in selected:
        if not isinstance(pair, dict):
            raise ValueError("Malformed selected pair")
        low, high = pair.get("product_low"), pair.get("product_high")
        if not all(isinstance(x, str) for x in (low, high)) or not low or not high:
            raise ValueError("Malformed selected pair IDs")
        if pair.get("audit_id") != priority("DISPLAY", low, high).hex():
            raise ValueError("Selected pair audit ID mismatch")
        if not all(isinstance(pair.get(k), str) for k in ("title_low", "title_high")):
            raise ValueError("Malformed selected titles")
        if pair["audit_id"] in audit_ids:
            raise ValueError("Repeated audit ID")
        audit_ids.add(pair["audit_id"])
    return manifest, expected


def cards_for(manifest: dict, reviewer: str, chosen: set[str] | None = None) -> list[dict[str, str]]:
    if reviewer not in {"R1", "R2", "R3"}:
        raise ValueError("Reviewer must be R1, R2, or R3")
    cards = []
    for pair in manifest["selected_pairs"]:
        aid = pair["audit_id"]
        if chosen is not None and aid not in chosen:
            continue
        left, right = pair["title_low"], pair["title_high"]
        if priority("ORIENT", reviewer, aid)[0] & 0x80:
            left, right = right, left
        cards.append({"audit_id": aid, "title_a": left, "title_b": right})
    return sorted(cards, key=lambda card: (priority("ORDER", reviewer, card["audit_id"]),
                                            card["audit_id"].encode("utf-8")))


def response_template(cards: list[dict[str, str]], reviewer: str,
                      manifest_sha: str) -> dict:
    return {"reviewer_id": reviewer, "manifest_sha256": manifest_sha,
            "cards_sha256": _sha256(canonical_json(cards)), "locked_at_utc": None,
            "responses": [
                {"audit_id": card["audit_id"], "label": None,
                 "concrete_attributes": [], "evidence": "", "ambiguity_reason": None,
                 "submitted_at_utc": None}
                for card in cards
            ]}


def _save_private_directory(output_dir: Path, cards: list[dict[str, str]],
                            template: dict) -> str:
    require_manifest_outside_repo(output_dir, PROJECT)
    if output_dir.exists():
        raise ValueError("Refusing to overwrite reviewer output")
    parent = output_dir.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".review_cards_", dir=parent))
    try:
        for name, data in (("cards.json", cards), ("responses.json", template)):
            path = temporary / name
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical_json(data))
                stream.flush()
                os.fsync(stream.fileno())
        if output_dir.exists():
            raise ValueError("Refusing to overwrite reviewer output")
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
    return template["cards_sha256"]


def prepare(manifest: dict, manifest_sha: str, reviewer: str,
            output_dir: Path, chosen: set[str] | None = None) -> str:
    cards = cards_for(manifest, reviewer, chosen)
    if chosen is not None and {c["audit_id"] for c in cards} != chosen:
        raise ValueError("Unknown audit ID in adjudication set")
    return _save_private_directory(output_dir, cards,
                                   response_template(cards, reviewer, manifest_sha))


def locked_responses(path: Path, lock_path: Path, manifest: dict,
                     manifest_sha: str, reviewer: str, seal_time: datetime) -> dict[str, str]:
    _, lock = _read_json(lock_path)
    locked_at = _anchor(lock, "locked_at_utc")
    if locked_at <= seal_time or lock.get("reviewer_id") != reviewer:
        raise ValueError("Response lock order or reviewer mismatch")
    if lock.get("manifest_sha256") != manifest_sha or not _digest(lock.get("responses_sha256")):
        raise ValueError("Response lock does not match selected manifest")
    raw, form = _read_json(path)
    if _sha256(raw) != lock["responses_sha256"]:
        raise ValueError("Response bytes differ from prior lock")
    cards = cards_for(manifest, reviewer)
    expect_ids = {item["audit_id"] for item in cards}
    if (form.get("reviewer_id") != reviewer or form.get("manifest_sha256") != manifest_sha or
            form.get("cards_sha256") != _sha256(canonical_json(cards)) or
            _utc(form.get("locked_at_utc")) != locked_at):
        raise ValueError("Response form metadata mismatch")
    responses = form.get("responses")
    if not isinstance(responses, list) or len(responses) != len(expect_ids):
        raise ValueError("Incomplete response form")
    labels = {}
    for answer in responses:
        if not isinstance(answer, dict) or answer.get("audit_id") not in expect_ids:
            raise ValueError("Unexpected response ID")
        aid = answer["audit_id"]
        if aid in labels or answer.get("label") not in LABELS:
            raise ValueError("Duplicate or incomplete response")
        attributes = answer.get("concrete_attributes")
        evidence = answer.get("evidence")
        if (not isinstance(attributes, list) or len(attributes) != len(set(attributes)) or
                any(attr not in ATTRIBUTES for attr in attributes) or
                not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Missing or invalid concrete title evidence")
        reason = answer.get("ambiguity_reason")
        if answer["label"] == "INDETERMINATE":
            if reason not in REASONS:
                raise ValueError("Indeterminate response needs ambiguity reason")
        elif reason is not None or not attributes:
            raise ValueError("Definite response needs an attribute and no ambiguity reason")
        submitted = _utc(answer.get("submitted_at_utc"))
        if not (seal_time < submitted <= locked_at):
            raise ValueError("Response timestamp outside sealed review window")
        labels[aid] = answer["label"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", required=True, type=Path)
    parser.add_argument("--manifest-seal-record", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--reviewer", choices=("R1", "R2", "R3"), required=True)
    parser.add_argument("--r1-responses", type=Path)
    parser.add_argument("--r1-lock-record", type=Path)
    parser.add_argument("--r2-responses", type=Path)
    parser.add_argument("--r2-lock-record", type=Path)
    args = parser.parse_args()
    require_manifest_outside_repo(args.private_output_dir, PROJECT)
    if args.private_output_dir.exists():
        raise ValueError("Refusing to overwrite reviewer output")
    manifest, seal_sha = load_sealed_manifest(args.private_manifest, args.manifest_seal_record)
    chosen = None
    if args.reviewer == "R3":
        if not all((args.r1_responses, args.r1_lock_record,
                    args.r2_responses, args.r2_lock_record)):
            raise ValueError("R3 requires two sealed and complete response forms")
        _, seal = _read_json(args.manifest_seal_record)
        seal_time = _utc(seal["sealed_at_utc"])
        r1 = locked_responses(args.r1_responses, args.r1_lock_record,
                              manifest, seal_sha, "R1", seal_time)
        r2 = locked_responses(args.r2_responses, args.r2_lock_record,
                              manifest, seal_sha, "R2", seal_time)
        chosen = {aid for aid in r1 if r1[aid] != r2[aid]}
    elif any((args.r1_responses, args.r1_lock_record,
              args.r2_responses, args.r2_lock_record)):
        raise ValueError("R1/R2 preparation must not read any prior decisions")
    cards_sha = prepare(manifest, seal_sha, args.reviewer, args.private_output_dir, chosen)
    print(json.dumps({"reviewer": args.reviewer, "cards": len(cards_for(manifest, args.reviewer, chosen)),
                      "cards_sha256": cards_sha, "manifest_sha256": seal_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
