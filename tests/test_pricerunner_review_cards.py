"""Synthetic-only checks for private PriceRunner title-evidence review cards."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import pricerunner_review_cards as review  # noqa: E402
import pricerunner_variant_sampler as sampler  # noqa: E402


def utc(hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def fixture(tmp_path):
    rows = [("a", "Sony Cam 32GB Blue", "1", "cluster-A"),
            ("b", "Sony Cam 32GB Red", "2", "cluster-A"),
            ("c", "Other Mic 2 Pack", "3", "cluster-B"),
            ("d", "Other Mic 3 Pack", "4", "cluster-B")]
    manifest = sampler.select(rows, source={"zip_sha256": "synthetic"},
                              code_sha256="synthetic", protocol_sha256="synthetic",
                              selection_commit="synthetic")
    path = tmp_path / "manifest.json"
    path.write_bytes(sampler.canonical_json(manifest))
    digest = sha256(path.read_bytes()).hexdigest()
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"manifest_sha256": digest,
                                "selection_commit": "synthetic",
                                "protocol_sha256": "synthetic",
                                "sealed_at_utc": utc(-4),
                                "external_anchor_url": "https://github.com/example/seal"}))
    return manifest, path, seal, digest


def test_cards_are_blind_deterministic_and_private(tmp_path):
    manifest, path, seal, digest = fixture(tmp_path)
    loaded, sealed_sha = review.load_sealed_manifest(path, seal)
    assert loaded == manifest and sealed_sha == digest
    for reviewer in ("R1", "R2"):
        cards = review.cards_for(manifest, reviewer)
        assert [c["audit_id"] for c in cards] == sorted(
            [p["audit_id"] for p in manifest["selected_pairs"]],
            key=lambda aid: (sampler.priority("ORDER", reviewer, aid), aid.encode()))
        for card in cards:
            assert set(card) == {"audit_id", "title_a", "title_b"}
            pair = next(p for p in manifest["selected_pairs"] if p["audit_id"] == card["audit_id"])
            titles = [pair["title_low"], pair["title_high"]]
            if sampler.priority("ORIENT", reviewer, card["audit_id"])[0] & 0x80:
                titles.reverse()
            assert [card["title_a"], card["title_b"]] == titles
    output = tmp_path / "private_r1"
    assert review.prepare(manifest, digest, "R1", output) == sha256(
        (output / "cards.json").read_bytes()
    ).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "cards.json").stat().st_mode & 0o777 == 0o600
    assert (output / "responses.json").stat().st_mode & 0o777 == 0o600
    assert "cluster-A" not in (output / "cards.json").read_text()
    assert all(x["label"] is None for x in json.loads(
        (output / "responses.json").read_text()
    )["responses"])
    with pytest.raises(ValueError, match="overwrite"):
        review.prepare(manifest, digest, "R1", output)


def test_tampered_seal_and_any_git_checkout_path_fail_closed(tmp_path):
    manifest, path, seal, digest = fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Manifest bytes"):
        review.load_sealed_manifest(path, seal)
    path.write_bytes(sampler.canonical_json(manifest))
    assert review.load_sealed_manifest(path, seal)[1] == digest
    with pytest.raises(ValueError, match="outside"):
        review.prepare(manifest, digest, "R1", review.PROJECT / "forbidden")
    other = tmp_path / "other-git"
    other.mkdir()
    (other / ".git").write_text("gitdir: /tmp/elsewhere\n")
    alias = tmp_path / "alias"
    alias.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="outside every Git repository"):
        review.prepare(manifest, digest, "R1", alias / "forbidden")


def _write_locked(tmp_path, manifest, digest, reviewer, labels):
    cards = review.cards_for(manifest, reviewer)
    form = review.response_template(cards, reviewer, digest)
    form["locked_at_utc"] = utc(-1)
    for item in form["responses"]:
        item["label"] = labels[item["audit_id"]]
        definite = item["label"] != "INDETERMINATE"
        item["concrete_attributes"] = ["color"] if definite else []
        item["evidence"] = "Title states a color" if definite else "Color omitted"
        item["ambiguity_reason"] = None if definite else "unspecified variant"
        item["submitted_at_utc"] = utc(-2)
    path = tmp_path / f"{reviewer}.json"
    path.write_bytes(sampler.canonical_json(form))
    lock = tmp_path / f"{reviewer}.lock.json"
    lock.write_text(json.dumps({"reviewer_id": reviewer, "manifest_sha256": digest,
                                "responses_sha256": sha256(path.read_bytes()).hexdigest(),
                                "locked_at_utc": form["locked_at_utc"],
                                "external_anchor_url": "https://github.com/example/locked"}))
    return path, lock


def test_r3_sees_only_disagreements_after_complete_locked_forms(tmp_path):
    manifest, _, seal, digest = fixture(tmp_path)
    ids = [p["audit_id"] for p in manifest["selected_pairs"]]
    assert len(ids) == 2
    p1, l1 = _write_locked(tmp_path, manifest, digest, "R1",
                           {ids[0]: "SAME_EXACT", ids[1]: "INDETERMINATE"})
    p2, l2 = _write_locked(tmp_path, manifest, digest, "R2",
                           {ids[0]: "SAME_EXACT", ids[1]: "DIFFERENT_VARIANT"})
    sealed_at = review._utc(json.loads(seal.read_text())["sealed_at_utc"])
    r1 = review.locked_responses(p1, l1, manifest, digest, "R1", sealed_at)
    r2 = review.locked_responses(p2, l2, manifest, digest, "R2", sealed_at)
    disputed = {aid for aid in r1 if r1[aid] != r2[aid]}
    assert disputed == {ids[1]}
    output = tmp_path / "private_r3"
    review.prepare(manifest, digest, "R3", output, disputed)
    cards = json.loads((output / "cards.json").read_text())
    assert [item["audit_id"] for item in cards] == [ids[1]]
    assert all(set(item) == {"audit_id", "title_a", "title_b"} for item in cards)
    assert "DIFFERENT_VARIANT" not in (output / "cards.json").read_text()
    p2.write_bytes(p2.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Response bytes"):
        review.locked_responses(p2, l2, manifest, digest, "R2", sealed_at)


def test_incomplete_or_missing_evidence_cannot_trigger_r3(tmp_path):
    manifest, _, seal, digest = fixture(tmp_path)
    labels = {p["audit_id"]: "DIFFERENT_VARIANT" for p in manifest["selected_pairs"]}
    path, lock_path = _write_locked(tmp_path, manifest, digest, "R1", labels)
    sealed_at = review._utc(json.loads(seal.read_text())["sealed_at_utc"])
    form = json.loads(path.read_text())
    form["responses"].pop()
    path.write_bytes(sampler.canonical_json(form))
    lock = json.loads(lock_path.read_text())
    lock["responses_sha256"] = sha256(path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Incomplete"):
        review.locked_responses(path, lock_path, manifest, digest, "R1", sealed_at)
    _, _ = _write_locked(tmp_path, manifest, digest, "R1", labels)
    lock = json.loads(lock_path.read_text())
    form = json.loads(path.read_text())
    form["responses"][0]["evidence"] = ""
    path.write_bytes(sampler.canonical_json(form))
    lock["responses_sha256"] = sha256(path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="evidence"):
        review.locked_responses(path, lock_path, manifest, digest, "R1", sealed_at)
