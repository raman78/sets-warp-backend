"""The icon models train at 128 and are published under their own folder.

Every client released before the input-size change feeds the models 224.
A 128 model fed 224 loses about 24 points in the embedder (measured
2026-09-28 on 1 564 held-out crops through the client's own matcher:
96.9 % → 72.6 %), so `models/` keeps the last 224 set and the 128 set goes
to `models/in128/`. The screen classifier, anchors and OCR corrections do
not depend on the size and stay under `models/`.

Offline: no HF — commits are captured, downloads come from a fake hub.
"""
from __future__ import annotations

import json

import pytest

import admin_train as t


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Files the fake HF repo holds, by path in repo; records what was asked."""
    files: dict[str, str] = {}
    asked: list[str] = []

    def _dl(repo_id=None, filename=None, repo_type=None, token=None, **kw):
        filename = filename or kw.get('filename')
        asked.append(filename)
        if filename not in files:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace('/', '__')
        p.write_text(files[filename])
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _dl)
    return files, asked


def test_the_folder_names_the_size_and_the_endpoint_agrees():
    """Changing one without the other would publish a size under a folder
    that promises another, or serve clients a folder nothing writes."""
    import main
    assert t.ICON_MODELS_PATH == f'models/in{t.MODEL_IMG_SIZE}'
    assert main.MODELS_PATH_BY_INPUT[t.MODEL_IMG_SIZE] == t.ICON_MODELS_PATH


def test_the_classifier_is_published_under_its_size_folder(tmp_path, monkeypatch):
    committed = []
    monkeypatch.setattr(t, '_create_commit_with_retry',
                        lambda api, repo, rtype, ops, msg: committed.append(ops) or True)
    monkeypatch.setattr(t, '_published_model_version', lambda: {})
    for name in ('icon_classifier.pt', 'label_map.json', 'icon_classifier_meta.json',
                 'training_manifest.json', 'screen_classifier.pt',
                 'screen_classifier_labels.json'):
        (tmp_path / name).write_text('x')

    assert t._upload_model(tmp_path, 3000, 0.88, 13000, 30, sc_val_acc=0.9)

    paths = {op.path_in_repo for op in committed[0]}
    assert paths == {
        'models/in128/icon_classifier.pt', 'models/in128/label_map.json',
        'models/in128/icon_classifier_meta.json', 'models/in128/model_version.json',
        'models/in128/training_manifest.json',
        'models/screen_classifier.pt', 'models/screen_classifier_labels.json',
    }


def test_the_embedder_is_published_under_its_size_folder(tmp_path, monkeypatch):
    import admin_train_metric as m
    committed = []
    monkeypatch.setattr(m, '_create_commit_with_retry',
                        lambda api, repo, rtype, ops, msg: committed.append(ops) or True)
    monkeypatch.setattr(m, '_published_embedder_meta', lambda: {})
    (tmp_path / 'icon_embedder.pt').write_bytes(b'w')
    (tmp_path / 'embedder_label_map.json').write_text('{}')
    (tmp_path / 'embedding_index.npz').write_bytes(b'i')
    (tmp_path / 'icon_embedder_meta.json').write_text(
        json.dumps({'n_classes': 3000, 'val_recall@1': 0.89}))

    assert m._upload_embedder(tmp_path)

    paths = {op.path_in_repo for op in committed[0]}
    assert paths == {'models/in128/icon_embedder.pt', 'models/in128/embedding_index.npz',
                     'models/in128/icon_embedder_meta.json',
                     'models/in128/embedder_label_map.json'}


def test_the_legacy_manifest_does_not_skip_the_first_run_at_a_new_size(hub):
    """The legacy manifest lists the crops of the last 224 run. Read as this
    size's, it would report "unchanged" and the first 128 model would never
    be trained until new crops arrived."""
    files, asked = hub
    files['models/training_manifest.json'] = json.dumps({'crop_shas': ['a', 'b']})

    shas, _digest = t._load_training_manifest()

    assert shas == set()
    assert 'models/training_manifest.json' not in asked


def test_warm_start_and_guard_read_this_size_first(hub):
    files, asked = hub
    files['models/in128/icon_classifier.pt'] = 'new'
    files['models/icon_classifier.pt'] = 'legacy'

    path = t._hf_icon_model_file('icon_classifier.pt')

    assert open(path).read() == 'new'


def test_until_this_size_exists_the_legacy_set_is_used(hub, capsys):
    """The first 128 run warm-starts from, and is guarded against, the model
    users have now — and says so."""
    files, _asked = hub
    files['models/icon_classifier.pt'] = 'legacy'

    path = t._hf_icon_model_file('icon_classifier.pt')

    assert open(path).read() == 'legacy'
    assert 'nothing under models/in128/' in capsys.readouterr().out


def test_neither_set_raises(hub):
    with pytest.raises(FileNotFoundError):
        t._hf_icon_model_file('icon_classifier.pt')
