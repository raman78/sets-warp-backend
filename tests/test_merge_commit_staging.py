"""The crop merge applies as ordered, chunked commits.

One commit carrying everything is what broke: once the backlog passed ~4600
operations, HF answered an opaque 400 — no message, only a Request ID — and
every scheduled run from 2026-07-16 died the same way. 2281 crops sat
un-promoted for seven weeks while the workflow reported success, because the
step piped its output through `tee` and bash returned tee's exit status.

Two defects, two guards: the chunked, ordered commit here, and `set -o
pipefail` in the workflow (covered by tests/test_workflow_exit_codes.py).

Offline: no HF, no network. The API is a recorder.
"""
from __future__ import annotations

import pytest

import democratic_merge_crops as merge
import hf_commit


class _Op:
    """Stands in for a CommitOperationAdd / Delete — only the path matters."""

    def __init__(self, path: str) -> None:
        self.path_in_repo = path

    def __repr__(self) -> str:
        return f'_Op({self.path_in_repo!r})'


class _Api:
    """Records every commit instead of making one."""

    def __init__(self) -> None:
        self.commits: list[dict] = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append({'ops': list(operations), 'msg': commit_message})


def _paths(commit) -> list[str]:
    return [o.path_in_repo for o in commit['ops']]


@pytest.fixture
def parts():
    crops = [_Op(f'data/crops/{i:02d}.png') for i in range(5)]
    anno = _Op('data/annotations.jsonl')
    drain = [_Op(f'staging/iid/crops/{i:02d}.png') for i in range(3)]
    return crops, anno, drain


# ── Order ──────────────────────────────────────────────────────────────────

def test_crops_are_committed_before_the_annotations_reference_them(parts):
    """An annotation pointing at a crop that is not there yet is a broken
    dataset; a crop nothing references yet is inert."""
    crops, anno, drain = parts
    api = _Api()
    merge._commit_in_stages(api, crops, anno, drain, 'msg', chunk=100)

    order = [_paths(c)[0] for c in api.commits]
    assert order.index('data/crops/00.png') < order.index('data/annotations.jsonl')


def test_staging_is_drained_only_after_the_data_is_authoritative(parts):
    """Deleting staging before `data/` carries the crop would lose it."""
    crops, anno, drain = parts
    api = _Api()
    merge._commit_in_stages(api, crops, anno, drain, 'msg', chunk=100)

    order = [_paths(c)[0] for c in api.commits]
    assert order.index('data/annotations.jsonl') < order.index('staging/iid/crops/00.png')


# ── Chunking ───────────────────────────────────────────────────────────────

def test_no_commit_exceeds_the_chunk_size(parts):
    crops, anno, drain = parts
    api = _Api()
    merge._commit_in_stages(api, crops, anno, drain, 'msg', chunk=2)

    assert max(len(c['ops']) for c in api.commits) <= 2


def test_every_operation_is_committed_exactly_once(parts):
    crops, anno, drain = parts
    api = _Api()
    merge._commit_in_stages(api, crops, anno, drain, 'msg', chunk=2)

    sent = [p for c in api.commits for p in _paths(c)]
    assert sorted(sent) == sorted(
        [o.path_in_repo for o in crops] + ['data/annotations.jsonl']
        + [o.path_in_repo for o in drain])


def test_an_empty_stage_produces_no_commit(parts):
    _crops, anno, _drain = parts
    api = _Api()
    merge._commit_in_stages(api, [], anno, [], 'msg', chunk=2)

    assert len(api.commits) == 1


# ── Validation: name the fault instead of sending a doomed commit ──────────

def test_duplicate_paths_are_refused_before_anything_is_written(parts):
    """Two operations on one path is one of the shapes that produces a 400,
    and the server does not say which."""
    _crops, anno, _drain = parts
    api = _Api()
    dupe = [_Op('data/crops/aa.png'), _Op('data/crops/aa.png')]

    with pytest.raises(SystemExit):
        merge._commit_in_stages(api, dupe, anno, [], 'msg')

    assert api.commits == []


def test_an_empty_path_is_refused(parts):
    _crops, anno, _drain = parts
    api = _Api()

    with pytest.raises(SystemExit):
        merge._commit_in_stages(api, [_Op('')], anno, [], 'msg')

    assert api.commits == []


@pytest.mark.parametrize('path', ['/data/crops/a.png', 'data//crops/a.png',
                                  'data/crops/'])
def test_a_malformed_path_is_refused(path, parts):
    _crops, anno, _drain = parts
    api = _Api()

    with pytest.raises(SystemExit):
        merge._commit_in_stages(api, [_Op(path)], anno, [], 'msg')

    assert api.commits == []


def test_a_clean_operation_set_reports_no_problem(parts):
    """The check itself lives in `hf_commit`, shared by every tool that
    commits — the crop merge was only the first to outgrow one commit."""
    crops, anno, drain = parts

    assert hf_commit.validate_ops(crops + [anno] + drain) == []
