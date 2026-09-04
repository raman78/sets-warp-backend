"""A staging crop with no annotation row is swept, not left to rot.

A crop is only ever tallied through its row in
`staging/<iid>/annotations.jsonl`. A PNG with no row can never be promoted,
drained, or seen again: it is dead weight that only a manual script removed,
and that script had not run since 2026-07-17. Ten such files were in staging
when this was written.

The sweep uses what the merge has already computed, so orphans stop being
possible rather than needing someone to remember them.

Offline: no HF, no network. The API is a recorder.
"""
from __future__ import annotations

import pytest

import democratic_merge_crops as merge


class _Api:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.added: list[str] = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        for op in operations:
            (self.deleted if type(op).__name__ == 'CommitOperationDelete'
             else self.added).append(op.path_in_repo)


@pytest.fixture
def apply_merge(monkeypatch):
    """Drive `_apply` with no network: nothing to copy, so only the drain runs."""
    def _run(repo_files, staging_records, promoted=frozenset()):
        api = _Api()
        merge._apply(
            api, 'token',
            merged={}, promoted_shas=set(promoted), existing={},
            crop_src={}, repo_files=set(repo_files),
            contributors_for_sha={}, staging_records=staging_records,
        )
        return api
    return _run


ANN = 'staging/iid/annotations.jsonl'


def _row(sha: str) -> dict:
    return {'crop_sha256': sha, 'name': 'Deflector Array', 'slot': 'Deflector'}


# ── What gets swept ────────────────────────────────────────────────────────

def test_a_crop_with_no_row_is_deleted(apply_merge):
    api = apply_merge(
        repo_files=[ANN, 'staging/iid/crops/aa.png'],
        staging_records={'iid': [_row('bb')]},
    )

    assert 'staging/iid/crops/aa.png' in api.deleted


def test_a_crop_that_still_has_a_row_is_kept(apply_merge):
    """It has a vote pending; deleting it would lose the only copy."""
    api = apply_merge(
        repo_files=[ANN, 'staging/iid/crops/bb.png'],
        staging_records={'iid': [_row('bb')]},
    )

    assert 'staging/iid/crops/bb.png' not in api.deleted


def test_a_row_belonging_to_another_install_does_not_protect_a_crop(apply_merge):
    """Rows are per install: a crop under one install is not kept alive by a
    row filed under a different one."""
    api = apply_merge(
        repo_files=[ANN, 'staging/other/crops/bb.png'],
        staging_records={'iid': [_row('bb')]},
    )

    assert 'staging/other/crops/bb.png' in api.deleted


def test_data_crops_are_never_swept(apply_merge):
    """The sweep is about staging. A crop in data/ is the dataset.

    Asserted as "no `data/` path is deleted" rather than "nothing is deleted".
    The two were the same until rows whose crop exists nowhere became
    sweepable, and this fixture's row is one: it names `bb`, which has no PNG
    in staging and no entry in data/. Deleting its staging annotations file is
    correct; the claim this test defends is about `data/`.
    """
    api = apply_merge(
        repo_files=[ANN, 'data/crops/ab/aa.png'],
        staging_records={'iid': [_row('bb')]},
    )

    assert [p for p in api.deleted if p.startswith('data/')] == []


def test_a_crop_promoted_in_this_batch_is_deleted_once(apply_merge):
    """Its row is being trimmed and its file dropped by the drain; the sweep
    must not queue a second delete for the same path — two operations on one
    path is refused outright."""
    api = apply_merge(
        repo_files=[ANN, 'staging/iid/crops/bb.png'],
        staging_records={'iid': [_row('bb')]},
        promoted={'bb'},
    )

    assert api.deleted.count('staging/iid/crops/bb.png') <= 1


def test_crops_from_an_install_with_no_rows_at_all_are_swept():
    """Uploads write PNGs and rows in one commit, so an install holding crops
    and no annotations file was written by something that is not the upload
    path. `staging/migration-sister/` is the case that occurred; without this
    the movement audit would report it for ever and only a manual script
    could clear it."""
    api = _Api()
    merge._apply(
        api, 'token', merged={}, promoted_shas=set(), existing={},
        crop_src={},
        repo_files={ANN, 'staging/migration-sister/crops/aa.png'},
        contributors_for_sha={}, staging_records={'iid': [_row('bb')]},
    )

    assert 'staging/migration-sister/crops/aa.png' in api.deleted
