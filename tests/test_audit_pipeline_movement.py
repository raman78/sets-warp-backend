"""The audit that would have caught the seven-week stall.

Every other audit here checks a state — how much staging has piled up, how
many crops are mislabelled. A pipeline that has stopped entirely looks
healthy to both of them until the pile is enormous. This one checks movement:
uploads arriving while `data/` never changes is a breach, whatever the cause.

Offline: no HF, no network. The API is a stub.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

import admin_audit_pipeline_movement as audit


class _Commit:
    def __init__(self, title: str, age_days: float) -> None:
        self.title = title
        self.commit_id = 'deadbeef'
        self.created_at = (datetime.now(timezone.utc)
                           - timedelta(days=age_days))


class _Api:
    def __init__(self, staging: int, commits: list) -> None:
        self._files = [f'staging/iid/crops/{i:04d}.png' for i in range(staging)]
        if staging:
            # Real staging always carries the rows that make its crops
            # tallyable; crops without them are the residue case, tested
            # separately below.
            self._files.append('staging/iid/annotations.jsonl')
        self._files.append('data/annotations.jsonl')
        self._commits = commits

    def list_repo_files(self, *a, **k):
        return list(self._files)

    def list_repo_commits(self, *a, **k):
        return list(self._commits)


@pytest.fixture
def run(monkeypatch):
    def _run(staging: int, commits: list, argv: list[str] | None = None) -> int:
        api = _Api(staging, commits)
        monkeypatch.setattr(audit, 'HfApi', lambda **k: api, raising=False)
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **k: api)
        monkeypatch.setattr(sys, 'argv',
                            ['admin_audit_pipeline_movement.py'] + (argv or []))
        return audit.main()
    return _run


PROMOTION = 'democratic_merge: 9971 entries (+39 new crops)'


# ── Breach ─────────────────────────────────────────────────────────────────

def test_work_waiting_and_a_stale_data_dir_is_a_breach(run):
    """The real failure: uploads arriving, nothing promoted for weeks."""
    assert run(2281, [_Commit(PROMOTION, 48)]) == 1


def test_work_waiting_and_nothing_ever_promoted_is_a_breach(run):
    assert run(50, [_Commit('WARP bulk: 50 crops', 0.1)]) == 1


def test_the_age_limit_is_the_boundary(run):
    """Just inside passes, just outside fails — otherwise the threshold is
    decoration."""
    assert run(100, [_Commit(PROMOTION, 1.9)]) == 0
    assert run(100, [_Commit(PROMOTION, 2.1)]) == 1


def test_the_limit_can_be_widened(run):
    assert run(100, [_Commit(PROMOTION, 4)], ['--max-age-days', '7']) == 0


# ── No breach ──────────────────────────────────────────────────────────────

def test_a_moving_pipeline_passes(run):
    assert run(120, [_Commit(PROMOTION, 0.2)]) == 0


def test_an_empty_staging_area_is_not_a_breach(run):
    """Nothing waiting means a quiet `data/` is the correct state, however
    long it has been quiet."""
    assert run(0, [_Commit(PROMOTION, 400)]) == 0


def test_other_commits_do_not_count_as_a_promotion(run):
    """Uploads and screen merges kept committing throughout the stall — only
    the crop promotion rewrites `data/annotations.jsonl`."""
    commits = [_Commit('WARP bulk: 50 crops + annotations', 0.1),
               _Commit('democratic_merge_screens: 361 screens', 0.5),
               _Commit(PROMOTION, 48)]

    assert run(2281, commits) == 1


# ── Residue: files no run can settle ───────────────────────────────────────

def test_crops_with_no_annotations_file_are_residue():
    """The shape that actually occurred: a one-off migration wrote crops
    under `staging/migration-sister/` and no rows to tally them, so every
    merge since has walked past them."""
    files = ['staging/migration-sister/crops/aa.png',
             'staging/migration-sister/crops/bb.png']

    assert len(audit._residue(files)) == 1
    assert 'migration-sister' in audit._residue(files)[0]


def test_an_install_with_rows_is_not_residue():
    files = ['staging/iid/crops/aa.png', 'staging/iid/annotations.jsonl']

    assert audit._residue(files) == []


def test_residue_fails_the_audit_even_while_the_pipeline_moves(run):
    """Movement is not enough: a healthy pipeline can flow around a pile
    nothing reads, which is exactly how it went unnoticed."""
    class _ApiWithResidue(_Api):
        def list_repo_files(self, *a, **k):
            return list(self._files) + ['staging/orphaned/crops/zz.png']

    import sys as _sys
    api = _ApiWithResidue(0, [_Commit(PROMOTION, 0.1)])
    import huggingface_hub, pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(huggingface_hub, 'HfApi', lambda **k: api)
        mp.setattr(_sys, 'argv', ['admin_audit_pipeline_movement.py'])
        assert audit.main() == 1
    finally:
        mp.undo()
