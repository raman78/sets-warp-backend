"""The trainers fetch each crop once, from the path the repo actually has.

`data/crops/` is part sharded (`<ab>/<sha>.png`, because HF caps a directory
at 10 000 files) and part flat, from before the shards. Guessing both paths
per crop doubled the request count against 12 274 crops and lost 7 903 of
them in one run: the embedder trained on a third of the dataset — 1592
classes instead of 2867 — and shipped, because the failure was counted and
never explained.

Offline: no HF, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SOURCES = ['admin_train_metric.py', 'admin_train.py']
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize('name', SOURCES)
def test_the_crop_path_comes_from_a_listing_not_a_guess(name):
    src = (ROOT / name).read_text()

    assert '_crop_paths' in src, 'no path map — the fetch is guessing again'
    assert 'list_repo_files' in src


@pytest.mark.parametrize('name', SOURCES)
def test_only_one_request_is_made_per_crop(name):
    """The regression was a loop over two candidate paths inside the fetch."""
    src = (ROOT / name).read_text()
    body = src[src.index('def _fetch_crop'):]
    body = body[:body.index('\n    all_shas')]

    assert body.count('_opener.open') == 1, 'more than one request per crop'
    assert 'for _rel in' not in body


@pytest.mark.parametrize('name', SOURCES)
def test_a_failure_is_reported_not_only_counted(name):
    src = (ROOT / name).read_text()

    assert '_first_errors' in src
    assert re.search(r'first failures', src), 'failures never reach the log'
