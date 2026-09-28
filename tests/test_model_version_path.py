"""`/model/version` tells a client which models path its version describes.

A 128-trained model fed at 224 — which every client before the input-size
change does — loses about 24 points in the embedder (measured 2026-09-28).
So the 128 set is published under `models/in128/` and `models/` keeps the
last 224 set. A client that can feed 128 asks with `?input=128` and
downloads from the `models_path` in the answer; one that asks nothing gets
the legacy set, exactly as before.

Offline: FastAPI's TestClient, `hf_hub_download` replaced by local files.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Files the fake HF repo holds, by path in repo."""
    files: dict[str, dict] = {}

    def _dl(repo_id, filename, repo_type, token=None):
        if filename not in files:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace('/', '__')
        p.write_text(json.dumps(files[filename]))
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _dl)
    monkeypatch.setattr(main, 'HF_REPO_ID', 'test/repo')
    monkeypatch.setattr(main, '_model_version_cache', {})
    return files


@pytest.fixture
def client():
    return TestClient(main.app)


LEGACY = {'trained_at': '2026-09-27T19:27:34Z', 'n_classes': 3160}
NEW = {'trained_at': '2026-09-29T10:00:00Z', 'n_classes': 3170}


def test_a_client_that_sends_nothing_gets_the_legacy_set(hub, client):
    hub['models/model_version.json'] = LEGACY
    hub['models/in128/model_version.json'] = NEW

    body = client.get('/model/version').json()

    assert body['trained_at'] == LEGACY['trained_at']
    assert body['models_path'] == 'models'


def test_a_128_client_gets_the_128_set(hub, client):
    hub['models/model_version.json'] = LEGACY
    hub['models/in128/model_version.json'] = NEW

    body = client.get('/model/version?input=128').json()

    assert body['trained_at'] == NEW['trained_at']
    assert body['models_path'] == 'models/in128'


def test_a_128_client_gets_the_legacy_set_until_128_is_published(hub, client):
    """The client ships before the first 128 run; it must keep updating."""
    hub['models/model_version.json'] = LEGACY

    body = client.get('/model/version?input=128').json()

    assert body['trained_at'] == LEGACY['trained_at']
    assert body['models_path'] == 'models'


def test_an_unknown_size_gets_the_legacy_set(hub, client):
    hub['models/model_version.json'] = LEGACY
    hub['models/in128/model_version.json'] = NEW

    body = client.get('/model/version?input=96').json()

    assert body['models_path'] == 'models'


def test_the_embedder_stamp_comes_from_the_same_path(hub, client):
    """The embedder has its own trainer and clock; its stamp must describe
    the embedder the client would download from that same path."""
    hub['models/model_version.json'] = LEGACY
    hub['models/icon_embedder_meta.json'] = {'trained_at': 'legacy-embedder'}
    hub['models/in128/model_version.json'] = NEW
    hub['models/in128/icon_embedder_meta.json'] = {'trained_at': 'embedder-128'}

    assert client.get('/model/version').json()['embedder_trained_at'] == 'legacy-embedder'
    assert client.get('/model/version?input=128').json()['embedder_trained_at'] == 'embedder-128'


def test_nothing_published_is_unavailable(hub, client):
    assert client.get('/model/version?input=128').json() == {'available': False}
