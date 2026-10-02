"""Synthetic-only checks of the prospective PriceRunner audit selection rule."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pricerunner_variant_sampler.py"
SPEC = importlib.util.spec_from_file_location("pricerunner_variant_sampler_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def run(rows):
    return audit.select(rows, source={"zip_sha256": "synthetic"},
                        code_sha256="synthetic", protocol_sha256="synthetic",
                        selection_commit="synthetic")


def test_exact_50_groups_each_and_reordered_input_is_identical():
    rows = []
    for i in range(60):
        rows += [(f"s{i}a", f"solo{i} x", "1", f"s-cluster-{i}"),
                 (f"s{i}b", f"solo{i} y", "2", f"s-cluster-{i}")]
        rows += [(f"d{i}a", f"brand{i} model{i}", "3", f"d-left-{i}"),
                 (f"d{i}b", f"brand{i} model{i}", "4", f"d-right-{i}")]
    result = run(rows)
    assert result == run(list(reversed(rows)))
    assert result["strata"] == {
        "S": {"eligible_pairs": 60, "eligible_groups": 60, "selected_groups": 50},
        "D": {"eligible_pairs": 60, "eligible_groups": 60, "selected_groups": 50},
    }
    chosen_s = [entry for entry in result["selected_pairs"] if entry["stratum"] == "S"]
    chosen_d = [entry for entry in result["selected_pairs"] if entry["stratum"] == "D"]
    expect_s = sorted(range(60), key=lambda i: (
        audit.priority("S_GROUP", f"s-cluster-{i}"),
        f"s-cluster-{i}".encode()))[:50]
    expect_d = sorted(range(60), key=lambda i: (
        audit.priority("D_GROUP", f"d-left-{i}", f"d-right-{i}"),
        f"d-left-{i}".encode(), f"d-right-{i}".encode()))[:50]
    assert [entry["group_low"] for entry in chosen_s] == [f"s-cluster-{i}" for i in expect_s]
    assert [entry["group_low"] for entry in chosen_d] == [f"d-left-{i}" for i in expect_d]
    assert len({entry["audit_id"] for entry in result["selected_pairs"]}) == 100


def test_exact_dedup_cross_merchant_and_group_minimum_pair():
    rows = [
        ("z", "Sony α Blue 64", "0001", "cluster-A"),
        ("é", "Sony α Blue 64", "1", "cluster-A"),
        ("a2", "Sony α Blue 64", "2", "cluster-A"),
        ("b1", "Sony α Blue 64", "3", "cluster-B"),
        ("b2", "Sony α Blue 64", "4", "cluster-B"),
    ]
    result = run(rows)
    # UTF-8 bytes: z < é, so only z survives the exact representative key.
    assert result["collapsed_rows"] == 1
    assert result["representative_rows"] == 4
    assert result["strata"]["S"]["eligible_pairs"] == 2
    assert result["strata"]["D"]["eligible_pairs"] == 4
    d = next(entry for entry in result["selected_pairs"] if entry["stratum"] == "D")
    expected = min((audit.sorted_ids(a, b) for a in ("z", "a2")
                    for b in ("b1", "b2")),
                   key=lambda pair: (audit.priority("D_PAIR", *pair),
                                     pair[0].encode(), pair[1].encode()))
    assert (d["product_low"], d["product_high"]) == expected
    assert d["group_priority_sha256"] == audit.priority("D_GROUP", "cluster-A", "cluster-B").hex()
    assert d["audit_id"] == audit.priority("DISPLAY", *expected).hex()
    assert result["selected_offer_overlap_count"] > 0


def test_d_requires_two_shared_tokens_and_at_least_three_fifths_jaccard():
    rows = [
        ("a", "a b c d", "1", "A"),
        ("b", "a b c e", "2", "B"),  # 3/5 exactly, eligible
        ("c", "a b x y", "3", "C"),  # 2/6 with A, ineligible
        ("d", "a q r s", "4", "D"),  # 1 shared, ineligible
    ]
    result = run(rows)
    assert result["strata"]["D"] == {
        "eligible_pairs": 1, "eligible_groups": 1, "selected_groups": 1}
    assert result["strata"]["S"]["eligible_groups"] == 0
    assert audit.title_tokens("ＡＢＣ _Colour Café") == frozenset({"abc", "colour", "café"})


@pytest.mark.parametrize("rows", [
    [("a", "title a", "1", "x"), ("a", "title b", "2", "y")],
    [("a", "title a", "1", "x"), ("b", " ", "2", "y")],
    [("a", "title a", "1", "x"), ("b", "title b", "２", "y")],
    [("a", "title a", "1", "x"), ("b", "title b", "2", "")],
])
def test_source_quality_fail_closed(rows):
    with pytest.raises(ValueError):
        run(rows)


def test_synthetic_archive_gate_and_manifest_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "SOURCE_ROWS", 2)
    archive = tmp_path / "product+classification+and+clustering.zip"
    data = (",".join(audit.SOURCE_COLUMNS) + "\n" +
            "1,Title One,1,C1,ignored,99,also ignored\n" +
            "2,Title Two,2,C1,ignored,99,also ignored\n").encode()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(audit.ZIP_MEMBER, data)
    rows = audit.read_official_snapshot(
        archive, sha256(archive.read_bytes()).hexdigest(), sha256(data).hexdigest())
    assert rows == [("1", "Title One", "1", "C1"), ("2", "Title Two", "2", "C1")]
    manifest = run(rows)
    destination = tmp_path / "private" / "selection.json"
    seal = audit.write_private_manifest(destination, manifest)
    assert seal == sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.read_text())["strata"]["S"]["selected_groups"] == 1
    assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="overwrite"):
        audit.write_private_manifest(destination, manifest)
    with pytest.raises(ValueError, match="ZIP SHA"):
        audit.read_official_snapshot(archive, "0" * 64, sha256(data).hexdigest())
    with pytest.raises(ValueError, match="CSV SHA"):
        audit.read_official_snapshot(archive, sha256(archive.read_bytes()).hexdigest(), "0" * 64)
    monkeypatch.setattr(audit, "SOURCE_ROWS", 3)
    with pytest.raises(ValueError, match="row count"):
        audit.read_official_snapshot(
            archive, sha256(archive.read_bytes()).hexdigest(), sha256(data).hexdigest())


def test_source_hash_constants_are_frozen_to_v11():
    assert audit.FROZEN_ZIP_SHA256 == "31a79e8b25ef223759c75f6ebd3319a9a519a2aefa55802e3f90187c937c0dbe"
    assert audit.FROZEN_CSV_SHA256 == "35b55e774a55da272c349d4bc18aa3eb243691340d8d95d8c199136daf35df10"


def test_selection_error_never_emits_partial_manifest(monkeypatch, tmp_path):
    def fail_after_first(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(audit, "_record_pair", fail_after_first)
    destination = tmp_path / "sealed.json"
    with pytest.raises(OSError, match="disk failure"):
        audit.write_private_manifest(destination, run([
            ("1", "Blue One", "1", "C"), ("2", "Blue Two", "2", "C")]))
    assert not destination.exists()


def test_private_manifest_must_resolve_outside_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "private" / "selection.json"
    audit.require_manifest_outside_repo(elsewhere, repo)
    with pytest.raises(ValueError, match="outside"):
        audit.require_manifest_outside_repo(repo / "selection.json", repo)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        audit.require_manifest_outside_repo(Path("repo") / ".." / "repo" / "selection.json", repo)
    link = tmp_path / "linked-repo"
    link.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        audit.require_manifest_outside_repo(link / "selection.json", repo)
    private = tmp_path / "private"
    private.mkdir()
    link_to_private = repo / "linked-private"
    link_to_private.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        audit.require_manifest_outside_repo(link_to_private / "selection.json", repo)
    with pytest.raises(ValueError, match="outside"):
        audit.write_private_manifest(
            Path(audit.__file__).resolve().parents[1] / "private_selection.json", {}
        )


def test_private_outputs_and_temp_reject_another_git_checkout(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    checkout = tmp_path / "another-checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /tmp/elsewhere\\n")
    with pytest.raises(ValueError, match="outside every Git repository"):
        audit.require_manifest_outside_repo(checkout / "private.json", project)
    alias = tmp_path / "linked-checkout"
    alias.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="outside every Git repository"):
        audit.require_manifest_outside_repo(alias / "private.json", project)
    monkeypatch.setattr(audit.tempfile, "gettempdir", lambda: str(checkout))
    with pytest.raises(ValueError, match="Temporary SQLite directory"):
        run([("a", "Brand Model", "1", "C"), ("b", "Brand Model", "2", "C")])
    assert sorted(path.name for path in checkout.iterdir()) == [".git"]


def test_temp_preflight_precedes_archive_read(tmp_path, monkeypatch):
    destination = tmp_path / "private.json"
    args = SimpleNamespace(private_manifest=destination, archive=tmp_path / "archive.zip",
                           freeze_attestation=tmp_path / "attestation.json",
                           expected_freeze_sha="a" * 40)
    monkeypatch.setattr(audit.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(audit, "private_temp_root", lambda project:
                        (_ for _ in ()).throw(ValueError("synthetic unsafe temp root")))
    monkeypatch.setattr(audit, "read_official_snapshot", lambda *a:
                        (_ for _ in ()).throw(AssertionError("source read")))
    with pytest.raises(ValueError, match="synthetic unsafe temp root"):
        audit.main()


def test_manual_freeze_fields_are_declared_not_remote_proof():
    commit, code_sha, protocol_sha = "a" * 40, "b" * 64, "c" * 64
    frozen_at = datetime.now(timezone.utc) - timedelta(hours=2)
    review = {
        "reviewer_github_login": "independent-reviewer",
        "review_record_url": "https://github.com/alex-jb/bridgetrend-vision/pull/12#issuecomment-12345",
        "reviewed_at_utc": (frozen_at + timedelta(hours=1)).isoformat(),
        "reviewed_selection_commit": commit,
        "reviewed_selection_code_sha256": code_sha,
        "reviewed_protocol_sha256": protocol_sha,
        "reviewed_zip_sha256": audit.FROZEN_ZIP_SHA256,
        "reviewed_csv_sha256": audit.FROZEN_CSV_SHA256,
        "reviewed_python_version": "3.11",
    }
    frozen = {
        "frozen_at_utc": frozen_at.isoformat(),
        "external_anchor_url": f"https://github.com/alex-jb/bridgetrend-vision/commit/{commit}",
        "manual_review": review,
    }
    result = audit.validate_manual_freeze_review(frozen, commit, code_sha, protocol_sha)
    assert result["freeze_review_status"] == "operator_declared_manual_review_not_machine_verified"
    frozen["external_anchor_url"] = "https://github.com/unrelated/project/commit/" + commit
    with pytest.raises(ValueError, match="exact commit permalink"):
        audit.validate_manual_freeze_review(frozen, commit, code_sha, protocol_sha)
    frozen["external_anchor_url"] = f"https://github.com/alex-jb/bridgetrend-vision/commit/{commit}"
    review["reviewer_github_login"] = "alex-jb"
    with pytest.raises(ValueError, match="separate GitHub reviewer"):
        audit.validate_manual_freeze_review(frozen, commit, code_sha, protocol_sha)
    review["reviewer_github_login"] = "independent-reviewer"
    review["reviewed_protocol_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="reviewed_protocol_sha256"):
        audit.validate_manual_freeze_review(frozen, commit, code_sha, protocol_sha)


def test_hash_serialization_is_protocol_exact():
    expected = sha256((audit.SEED + '\n' + '["D_GROUP","a,b","c\\"d"]').encode()).digest()
    assert audit.priority("D_GROUP", "a,b", 'c"d') == expected
