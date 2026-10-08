"""CPU contract tests for the offline Exp2 retrieval comparison."""

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from tower.world_builder import exp2_retrieval as E
from tower.world_builder import global_solve as GS
from tests.test_world_builder_solve_masks import colmap  # noqa: F401


def test_switch_is_exact_and_off_by_default(monkeypatch):
    for value in (None, "off", "ON", "dinov2_GeM", "dinov2_gem ", "garbage"):
        if value is None:
            monkeypatch.delenv(E.SWITCH, raising=False)
        else:
            monkeypatch.setenv(E.SWITCH, value)
        assert E.enabled() is False
    monkeypatch.setenv(E.SWITCH, "dinov2_gem")
    assert E.enabled() is True


def test_rank_cadence_budget_distance_and_tie():
    vectors = np.eye(60, dtype=np.float32)
    vectors[0] = vectors[59]
    vectors[1] = vectors[59]
    rows = E.rank_candidates([f"{i:03}.jpg" for i in range(60)], vectors)
    assert [r["query_index"] for r in rows] == list(range(0, 60, 10))
    assert all(len(r["targets"]) == 50 for r in rows)
    assert rows[0]["targets"][:2] == ["001.jpg", "059.jpg"]
    assert "000.jpg" not in rows[0]["targets"]


def test_gem_uses_patch_tokens_p3_and_l2():
    patches = np.array([[[1., 4.], [8., 2.]]], dtype=np.float32)
    got = E.gem_patch_tokens(patches)[0]
    raw = np.power(np.mean(np.power(patches[0], 3), axis=0), 1 / 3)
    np.testing.assert_allclose(got, raw / np.linalg.norm(raw), rtol=1e-6)


def test_gem_clamps_negative_patch_activations_before_pooling():
    patches = np.array([[[-8., 4.], [2., 4.]]], dtype=np.float32)
    got = E.gem_patch_tokens(patches)[0]
    assert got[0] > 0
    assert got[1] > got[0]


def test_cache_identity_rejects_mutated_image_and_model(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"first")
    calls = []
    def fake_model(paths):
        calls.append(list(paths))
        return np.array([[3., 4.]], dtype=np.float32)
    out = tmp_path / "cache.json"
    E.descriptor_cache([image], out, "a" * 64, fake_model)
    E.descriptor_cache([image], out, "a" * 64, fake_model)
    assert len(calls) == 1
    image.write_bytes(b"other")
    E.descriptor_cache([image], out, "a" * 64, fake_model)
    E.descriptor_cache([image], out, "b" * 64, fake_model)
    assert len(calls) == 3


def test_copy_provenance_and_one_byte_refusal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.jpg").write_bytes(b"image")
    copied = tmp_path / "copied"
    record = E.copy_inputs(source, copied)
    assert (copied / "x.jpg").read_bytes() == b"image"
    assert E.verify_copied_inputs(copied, record)
    (copied / "x.jpg").write_bytes(b"imAge")
    with pytest.raises(ValueError, match="digest"):
        E.verify_copied_inputs(copied, record)


def test_freeze_manifest_pins_options_tree_and_checkpoint(tmp_path):
    image = tmp_path / "a.jpg"
    tree = tmp_path / "tree.bin"
    weights = tmp_path / "weights.bin"
    for file, data in ((image, b"image"), (tree, b"tree"), (weights, b"weights")):
        file.write_bytes(data)
    manifest = E.freeze_manifest([image], tree, weights, "4.2.0", {"overlap": 20},
                                 labels_sha256="1" * 64)
    assert manifest["tree_sha256"] == hashlib.sha256(b"tree").hexdigest()
    assert manifest["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert manifest["sequential_pairing_options"] == {"overlap": 20}
    assert manifest["retrieval"]["k"] == 50


def test_freeze_writer_serializes_pycolmap_path_options(tmp_path):
    target = tmp_path / "freeze.json"
    E._json(target, {"sequential_pairing_options": {"vocab_tree_path": Path("tree.bin")}})
    assert json.loads(target.read_text())["sequential_pairing_options"]["vocab_tree_path"] == "tree.bin"


def test_candidate_manifest_deduplicates_and_pins_bytes(tmp_path):
    rows = [{"query_index": 0, "query": "a.jpg", "targets": ["b.jpg", "c.jpg", "b.jpg"]}]
    path = tmp_path / "pairs.txt"
    manifest = E.write_candidate_manifest(rows, path, overlap=1)
    assert path.read_bytes() == b"a.jpg b.jpg\na.jpg c.jpg\n"
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["nonsequential_unique"] == 1


def test_retrieval_match_keeps_sequential_and_imports_exact_candidate_file(tmp_path, monkeypatch):
    names = [f"{i:03}.jpg" for i in range(60)]
    images = tmp_path / "images"
    images.mkdir()
    for name in names:
        (images / name).write_bytes(b"x")
    monkeypatch.setattr(E, "checkpoint_file", lambda: tmp_path / "weights")
    (tmp_path / "weights").write_bytes(b"weights")
    monkeypatch.setattr(E, "descriptor_cache", lambda *a, **k: np.eye(60, dtype=np.float32))
    monkeypatch.setattr(E, "remove_nonsequential_pairs", lambda *a, **k: 0)
    calls = []
    class Imported:
        match_list_path = None
    fake = SimpleNamespace(ImportedPairingOptions=Imported,
                           match_image_pairs=lambda *a, **k: calls.append((a, k)))
    pairing = SimpleNamespace(loop_detection=True, overlap=20)
    E.match_retrieval(fake, tmp_path / "database.db", images, names, tmp_path,
                      "matching", pairing, None,
                      lambda *a: calls.append("sequential"))
    assert pairing.loop_detection is True
    assert calls[0] == "sequential"
    assert calls[1][1]["pairing_options"].match_list_path.endswith("exp2-candidates.txt")
    assert (tmp_path / "exp2-candidates.json").is_file()


def test_r_refuses_a_root_without_copy_provenance_before_solver_io(tmp_path, monkeypatch):
    monkeypatch.setenv(E.SWITCH, "dinov2_gem")
    monkeypatch.setattr(GS, "solver_available", lambda: (True, None))
    with pytest.raises(FileNotFoundError):
        GS.solve(SimpleNamespace(root=tmp_path), "w", "s", final=True)
    assert list(tmp_path.iterdir()) == []


def test_r_refuses_copied_root_without_frozen_manifest_before_solver_io(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.jpg").write_bytes(b"x")
    copied = tmp_path / "copied"
    E.copy_inputs(source, copied)
    monkeypatch.setenv(E.SWITCH, "dinov2_gem")
    monkeypatch.setattr(GS, "solver_available", lambda: (True, None))
    with pytest.raises(ValueError, match="freeze"):
        GS.solve(SimpleNamespace(root=copied), "w", "s", final=True)
    assert not (copied / "worlds").exists()


def test_r_requires_the_final_loop_detection_recipe(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    copied = tmp_path / "copied"
    E.copy_inputs(source, copied)
    (copied / "exp2-freeze.json").write_text("{}")
    monkeypatch.setenv(E.SWITCH, "dinov2_gem")
    monkeypatch.setattr(GS, "solver_available", lambda: (True, None))
    with pytest.raises(ValueError, match="loop_detection=True"):
        GS.solve(SimpleNamespace(root=copied), "w", "s", final=True, loop_detection=False)


def test_tree_logger_records_preverification_proposal_order(tmp_path):
    tree = tmp_path / "tree.bin"
    tree.write_bytes(b"tree")
    class DB:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_all_images(self):
            return [SimpleNamespace(name=f"{i:03}.jpg", image_id=i + 1) for i in range(20)]
    class Options:
        def todict(self): return {"num_images": self.num_images}
    class Generator:
        def __init__(self, options, database, query_ids):
            assert query_ids == [1, 11]
        def all_pairs(self): return [(1, 20), (1, 18), (11, 2)]
    fake = SimpleNamespace(__version__="4.2.0", VocabTreePairingOptions=Options,
                           Database=SimpleNamespace(open=lambda path: DB()),
                           VocabTreePairGenerator=Generator)
    out = tmp_path / "tree.json"
    E.log_tree_candidates(tmp_path / "db", tree, [f"{i:03}.jpg" for i in range(20)],
                          out, pycolmap_module=fake)
    rows = json.loads(out.read_text())["rows"]
    assert rows[0]["targets"] == ["019.jpg", "017.jpg"]
    assert rows[0]["target_indices"] == [19, 17]


def test_r_removes_only_prior_nonsequential_pairs_from_copied_db(tmp_path):
    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.executescript("create table images(image_id integer,name text);"
                      "create table matches(pair_id integer);"
                      "create table two_view_geometries(pair_id integer);")
    con.executemany("insert into images values (?,?)",
                    [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")])
    close = 1 * 2147483647 + 2
    far = 1 * 2147483647 + 3
    for table in ("matches", "two_view_geometries"):
        con.executemany(f"insert into {table} values (?)", [(close,), (far,)])
    con.commit()
    con.close()
    assert E.remove_nonsequential_pairs(db, ["a.jpg", "b.jpg", "c.jpg"], overlap=1) == 2
    con = sqlite3.connect(db)
    assert [row[0] for row in con.execute("select pair_id from matches")] == [close]
    assert [row[0] for row in con.execute("select pair_id from two_view_geometries")] == [close]
    con.close()


def test_off_spellings_preserve_pinned_s_golden_and_one_byte_mutant(tmp_path, colmap, monkeypatch):
    from tests import test_world_builder_final_scale_guard_off_identity as G
    for name in G.OTHER_SWITCHES + (G.SWITCH,):
        monkeypatch.delenv(name, raising=False)
    for index, value in enumerate((None, "off", "typo")):
        if value is None:
            monkeypatch.delenv(E.SWITCH, raising=False)
        else:
            monkeypatch.setenv(E.SWITCH, value)
        s = G._make_walk(tmp_path / f"off{index}", colmap, monkeypatch)
        returned = G._final(s, "gated")
        observed = G._observe(s, returned)
        G._check("gated", observed)
        assert not any("exp2-" in path for path in observed["files"])
        if index == 0:
            path = s.workspace.solution_path
            before = path.read_bytes()
            mutant = bytearray(before)
            at = mutant.index(b'"translation"') + len(b'"translation": [')
            while not chr(mutant[at]).isdigit():
                at += 1
            mutant[at] = ord("7") if mutant[at] != ord("7") else ord("3")
            path.write_bytes(mutant)
            with pytest.raises(AssertionError):
                G._check("gated", G._observe(s, returned))
            path.write_bytes(before)


def test_off_spellings_write_identical_raw_three_files_at_one_path(tmp_path, colmap, monkeypatch):
    from tests import test_world_builder_final_scale_guard_off_identity as G
    from tower.world_builder import solve_masks

    root = tmp_path / "replay"
    baseline = None
    with monkeypatch.context() as fixed:
        fixed.setattr(time, "time", lambda: G.CLOCK0 + 500.0)
        fixed.setattr(time, "perf_counter", lambda: 1000.0)
        fixed.setattr(solve_masks.uuid, "uuid4", lambda: uuid.UUID(int=0))
        for name in G.OTHER_SWITCHES + (G.SWITCH,):
            fixed.delenv(name, raising=False)
        for index, value in enumerate((None, "off", "misspelled")):
            if value is None:
                fixed.delenv(E.SWITCH, raising=False)
            else:
                fixed.setenv(E.SWITCH, value)
            s = G._make_walk(root, colmap, fixed)
            G._final(s, "gated")
            files = {
                "solution.json": s.workspace.solution_path.read_bytes(),
                "align.json": (s.store.world_dir(s.world_id) / "dense" / s.session_id /
                               "align.json").read_bytes(),
                "components.json": (s.workspace.root / "components.json").read_bytes(),
            }
            if baseline is None:
                baseline = files
            else:
                assert files == baseline
            archive = tmp_path / f"archive{index}"
            assert root.resolve().parent == tmp_path.resolve()
            assert archive.resolve().parent == tmp_path.resolve()
            root.rename(archive)
