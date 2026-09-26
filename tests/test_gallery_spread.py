"""A collapsed embedder gallery is not published.

The embedder trained 2026-09-26 05:59 UTC mapped every picture to almost the
same vector (random gallery pairs 0.990 cosine against 0.033 before), while
val_recall@1 still read 0.80, so the class-count/accuracy guard passed it.

Offline: no HF, no network; the upload is stubbed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import admin_train_metric as atm


def _healthy(n=400, d=64):
    return np.random.default_rng(3).normal(size=(n, d)).astype(np.float32)


def _collapsed(n=400, d=64):
    base = np.random.default_rng(4).normal(size=d)
    return (base + np.random.default_rng(5).normal(scale=0.05, size=(n, d))).astype(np.float32)


def test_the_measure_tells_the_two_apart():
    assert atm._gallery_spread(_healthy()) < 0.1
    assert atm._gallery_spread(_collapsed()) > atm.GALLERY_COLLAPSED_SIM


def _model_dir(tmp_path, emb):
    d = tmp_path / 'models'
    d.mkdir()
    (d / 'icon_embedder.pt').write_bytes(b'x')
    (d / 'embedder_label_map.json').write_text('{}')
    np.savez(d / 'embedding_index.npz', embeddings=emb, labels=np.zeros(len(emb), dtype=np.int32))
    (d / 'icon_embedder_meta.json').write_text(json.dumps({'n_classes': 3159, 'val_recall@1': 0.80}))
    return d


@pytest.fixture
def uploads(monkeypatch):
    sent = []
    monkeypatch.setattr(atm, '_published_embedder_meta',
                        lambda: {'n_classes': 3155, 'val_acc': 0.83})
    monkeypatch.setattr(atm, '_create_commit_with_retry',
                        lambda *a, **k: sent.append(a) or True)
    return sent


def test_a_collapsed_gallery_is_not_published(tmp_path, uploads, capsys):
    assert atm._upload_embedder(_model_dir(tmp_path, _collapsed())) is False
    assert uploads == []
    assert 'collapsed' in capsys.readouterr().err


def test_a_healthy_gallery_is_published(tmp_path, uploads):
    assert atm._upload_embedder(_model_dir(tmp_path, _healthy())) is True
    assert len(uploads) == 1


def test_the_measure_matches_the_clients():
    """Two copies of one measure must give one number; the client refuses
    on it too. Needs the sto-warp checkout beside this repo."""
    sibling = Path(__file__).resolve().parents[2] / 'sto-warp'
    if not (sibling / 'warp' / 'recognition' / 'icon_matcher.py').exists():
        pytest.skip('sto-warp checkout not beside this repo')
    sys.path.insert(0, str(sibling))
    try:
        from warp.recognition.icon_matcher import GALLERY_COLLAPSED_SIM, gallery_spread
    except Exception as e:
        pytest.skip(f'sto-warp not importable here: {e}')
    for emb in (_healthy(), _collapsed()):
        assert atm._gallery_spread(emb) == pytest.approx(gallery_spread(emb))
    assert atm.GALLERY_COLLAPSED_SIM == GALLERY_COLLAPSED_SIM
