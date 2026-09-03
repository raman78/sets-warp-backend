"""The trainers get their crops in one operation, and say when they cannot.

Fetching 12 274 crops one by one over the REST path draws `HTTP 429 Too Many
Requests`: one run finished with 4 520 of them and the embedder published
having trained on a third of the dataset — 1 637 classes against 2 867.
Recall went *up*, which is what training on an easier subset looks like.

Nothing said why, because the fetch swallowed every exception and reported a
count. Both halves are covered here: the crops come over the git protocol,
which the tarball builder and the client's cold start already use for exactly
this reason, and a crop that cannot be had is named rather than counted.

Offline: no HF, no network — these read the shipped source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SOURCES = ['admin_train_metric.py', 'admin_train.py']
ROOT = Path(__file__).resolve().parent.parent


def _fetch_body(name: str) -> str:
    src = (ROOT / name).read_text()
    body = src[src.index('def _fetch_crop'):]
    return body[:body.index('\n    all_shas')]


@pytest.mark.parametrize('name', SOURCES)
def test_crops_arrive_in_one_operation(name):
    src = (ROOT / name).read_text()

    assert 'git' in src and 'sparse-checkout' in src, (
        'the crops are being fetched per file again — that is what drew 429s')


@pytest.mark.parametrize('name', SOURCES)
def test_no_per_crop_http_request_remains(name):
    """The regression was a request inside the per-crop function."""
    body = _fetch_body(name)

    assert '_opener.open' not in body
    assert 'hf_hub_download' not in body


@pytest.mark.parametrize('name', SOURCES)
def test_a_missing_crop_is_named_not_only_counted(name):
    src = (ROOT / name).read_text()

    assert '_first_errors' in src
    assert 'first failures' in src
