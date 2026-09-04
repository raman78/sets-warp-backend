"""A collapsed model must not reach users.

This has happened. A model carrying 1592 of roughly 3000 classes was
published and served, because the crop download had silently lost most of the
dataset and nothing compared the result against what was already out there.
The run was green throughout — the trainer had no opinion about whether the
thing it produced was worse than the thing it replaced.

The thresholds are deliberately loose. This is not a quality gate; a model
that trains a little worse is still a model. It is a collapse detector, and
its job is to stay silent until something is badly wrong.

Run standalone:
    python -m pytest tests/test_publication_guard.py -v
"""
from __future__ import annotations

import admin_train as t


def _prev(n_classes: int, val_acc: float = 0.80) -> dict:
    return {'n_classes': n_classes, 'val_acc': val_acc}


# ── What must be refused ───────────────────────────────────────────────────

def test_a_collapsed_class_count_is_refused():
    """The incident: roughly half the dataset silently missing."""
    refusal = t._publication_refusal(1592, 0.80, _prev(3187))

    assert refusal
    assert '1592' in refusal and '3187' in refusal


def test_the_refusal_says_what_it_saw():
    """A guard that only says 'no' gets switched off the first time it fires."""
    refusal = t._publication_refusal(1592, 0.80, _prev(3187))

    assert 'download' in refusal.lower()


def test_a_large_accuracy_drop_is_refused():
    refusal = t._publication_refusal(3187, 0.55, _prev(3187, 0.80))

    assert refusal
    assert '55' in refusal and '80' in refusal


# ── What must still publish ────────────────────────────────────────────────

def test_a_normal_run_publishes():
    assert not t._publication_refusal(3187, 0.801, _prev(3187, 0.800))


def test_growth_publishes():
    """The usual case — the dataset gets bigger."""
    assert not t._publication_refusal(3300, 0.82, _prev(3187, 0.80))


def test_a_small_shrink_publishes():
    """Classes do leave: a crop is rejected, a label is corrected onto another.
    Only a collapse is a fault."""
    assert not t._publication_refusal(3100, 0.80, _prev(3187, 0.80))


def test_a_small_accuracy_dip_publishes():
    """Retraining is stochastic; a couple of points is noise, not a defect."""
    assert not t._publication_refusal(3187, 0.77, _prev(3187, 0.80))


def test_the_boundary_is_not_a_refusal():
    """Exactly at the floor still publishes — the guard fires below it, so a
    dataset sitting on the line does not flap in and out."""
    assert not t._publication_refusal(2868, 0.80, _prev(3187, 0.80))


# ── Nothing to compare against ─────────────────────────────────────────────

def test_the_first_ever_publication_proceeds():
    """Refusing it would deadlock a fresh repo: nothing could ever be first."""
    assert not t._publication_refusal(3187, 0.80, {})


def test_an_unreadable_previous_version_does_not_block():
    """`_published_model_version` returns {} when the fetch fails. A network
    blip must not stop a good model from shipping."""
    assert not t._publication_refusal(3187, 0.80, {'n_classes': 0})


def test_a_previous_version_with_no_accuracy_still_guards_the_class_count():
    """Older metadata carried no `val_acc`; the count check must survive it."""
    refusal = t._publication_refusal(1000, 0.80, {'n_classes': 3187})

    assert refusal


# ── Both trainers, one threshold ───────────────────────────────────────────
#
# The embedder was published unguarded until 2026-09-05, and it is the model
# that matters more: it is the primary matcher for BOFF abilities and carries
# the `__empty__` / `__inactive__` gallery classes. A collapsed one does not
# recognise less — it answers with the nearest class it still holds, which
# reads as confident recognition of the wrong item.


def _embedder_dir(tmp_path, n_classes: int, recall: float):
    """A models dir shaped as `_upload_embedder` expects to find it."""
    import json
    (tmp_path / 'icon_embedder.pt').write_bytes(b'weights')
    (tmp_path / 'embedder_label_map.json').write_text('{}')
    (tmp_path / 'embedding_index.npz').write_bytes(b'idx')
    (tmp_path / 'icon_embedder_meta.json').write_text(json.dumps(
        {'n_classes': n_classes, 'val_recall@1': recall}))
    return tmp_path


def test_a_collapsed_embedder_is_not_committed(tmp_path, monkeypatch):
    """Drives the real upload path: the guard must stop it before any commit,
    not merely exist in the file."""
    import admin_train_metric as m

    committed = []
    monkeypatch.setattr(m, '_create_commit_with_retry',
                        lambda *a, **k: committed.append(a) or True)
    monkeypatch.setattr(m, '_published_embedder_meta',
                        lambda: {'n_classes': 3187, 'val_acc': 0.88})

    ok = m._upload_embedder(_embedder_dir(tmp_path, 1592, 0.88))

    assert ok is False
    assert committed == []


def test_a_healthy_embedder_is_committed(tmp_path, monkeypatch):
    """The guard must not be the thing that stops normal publication."""
    import admin_train_metric as m

    committed = []
    monkeypatch.setattr(m, '_create_commit_with_retry',
                        lambda *a, **k: committed.append(a) or True)
    monkeypatch.setattr(m, '_published_embedder_meta',
                        lambda: {'n_classes': 3187, 'val_acc': 0.88})

    ok = m._upload_embedder(_embedder_dir(tmp_path, 3200, 0.89))

    assert ok is True
    assert len(committed) == 1


def test_the_embedder_recall_is_read_as_the_shared_accuracy(tmp_path, monkeypatch):
    """The embedder records `val_recall@1` where the classifier records
    `val_acc`. A drop in it must trigger the shared guard, which means the
    mapping has to happen — not just be written down."""
    import admin_train_metric as m

    committed = []
    monkeypatch.setattr(m, '_create_commit_with_retry',
                        lambda *a, **k: committed.append(a) or True)
    monkeypatch.setattr(m, '_published_embedder_meta',
                        lambda: {'n_classes': 3187, 'val_acc': 0.88})

    ok = m._upload_embedder(_embedder_dir(tmp_path, 3187, 0.60))

    assert ok is False
    assert committed == []
