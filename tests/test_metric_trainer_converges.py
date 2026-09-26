"""A projection that did not learn is trained again, and never saved.

On 2026-09-26 the nightly embedder run stalled at loss 5.3 from epoch 3; a
rerun on the same code and data converged to 0.19. The stalled run's
epoch-1 state — a collapsed gallery — was saved and published. The trainer
now checks the final loss, retries from the next seed on the features it
already has, and fails the run after TRAIN_ATTEMPTS without saving.

Offline: the EfficientNet backbone is replaced by a tiny pooling network so
nothing is downloaded; the loop, ArcFace head, sampler and save are real.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import admin_train_metric as atm


@pytest.fixture
def small(monkeypatch):
    import torch.nn as nn

    def _tiny(prev_model_pt=None):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten())
                self.proj = nn.Linear(48, atm.EMBED_DIM)
        return Tiny()

    monkeypatch.setattr(atm, '_build_embedder', _tiny)
    monkeypatch.setattr(atm, 'MAX_EPOCHS', 2)
    monkeypatch.setattr(atm, 'BATCHES_PER_EPOCH_MIN', 2)
    monkeypatch.setattr(atm, 'N_FEATURE_AUGS', 1)
    monkeypatch.setattr(atm, 'MODEL_IMG_SIZE', 32)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (40, 40, 3), dtype=np.uint8) for _ in range(32)]
    labels = [f'Item {i % 8}' for i in range(32)]
    return crops, labels


def test_a_converged_run_is_saved_with_its_seed(tmp_path, small, monkeypatch):
    monkeypatch.setattr(atm, 'CONVERGED_LOSS', float('inf'))
    atm._fit_metric(*small, models_dir=tmp_path, prev_model_pt=None, deadline=None, seed=41)
    meta = json.loads((tmp_path / 'icon_embedder_meta.json').read_text())

    assert (meta['seed'], meta['attempts']) == (41, 1)


def test_a_run_that_did_not_learn_is_retried_with_the_next_seed(tmp_path, small,
                                                                   monkeypatch, capsys):
    """With the threshold below any reachable loss, every attempt counts as
    stalled: the retry must run from the next seed and then give up."""
    seeds = []
    real = atm._seed_all
    monkeypatch.setattr(atm, '_seed_all', lambda s: (seeds.append(s), real(s)))
    monkeypatch.setattr(atm, 'CONVERGED_LOSS', -1.0)
    with pytest.raises(RuntimeError, match='did not converge in 2 attempts'):
        atm._fit_metric(*small, models_dir=tmp_path, prev_model_pt=None, deadline=None, seed=7)

    assert seeds[-2:] == [7, 8]
    assert 'Did not converge' in capsys.readouterr().out


def test_nothing_is_saved_when_no_attempt_learned(tmp_path, small, monkeypatch):
    monkeypatch.setattr(atm, 'CONVERGED_LOSS', -1.0)
    with pytest.raises(RuntimeError):
        atm._fit_metric(*small, models_dir=tmp_path, prev_model_pt=None, deadline=None, seed=3)

    assert not (tmp_path / 'icon_embedder.pt').exists()
