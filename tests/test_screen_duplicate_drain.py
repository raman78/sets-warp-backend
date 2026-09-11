"""One screenshot filed under two types still gets both copies drained.

A client can upload the same sha under two screen types. sto-warp's
`set_screen_type` did exactly that until 2026-09-11: it copied the screenshot
into the new type's folder and left the old copy, so the classifier's guess
and the user's correction of it both went up. Measured on the maintainer's
store: 110 screenshots filed under two or three mutually exclusive types.

The vote is still one per (install, sha) — that part was already right. What
was missing is that only the *voting* path was recorded, so the other copy was
never drained and sat in staging for ever. And because the vote goes to
whichever copy the file walk reaches first, every later run was a fresh coin
toss between the correction and the guess.

Offline: a staging tree under `tmp_path`, `clone_hf_shallow` monkeypatched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import democratic_merge_screens as m


def _staging(tmp_path, *entries: tuple[str, str, str]) -> Path:
    """Build `staging/<iid>/screen_types/<TYPE>/<sha>.png` for each entry."""
    snap = tmp_path / 'snap'
    for iid, stype, sha in entries:
        d = snap / 'staging' / iid / 'screen_types' / stype
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{sha}.png').write_bytes(b'\x89PNG' + sha.encode())
    return snap


@pytest.fixture
def collect(tmp_path, monkeypatch):
    def _run(*entries):
        snap = _staging(tmp_path, *entries)
        import hf_clone
        monkeypatch.setattr(hf_clone, 'clone_hf_shallow',
                            lambda *a, **k: str(snap))
        votes, src, _text, _rej, _unp = m._collect_votes('token')
        return votes, src
    return _run


SHA = 'a' * 40


# ── The vote ──────────────────────────────────────────────────────────────

def test_one_install_filing_two_types_casts_one_vote(collect):
    """Otherwise a client with a stale copy would outvote everybody else."""
    votes, _src = collect(('iid1', 'BOFFS', SHA), ('iid1', 'SPACE_BOFFS', SHA))
    assert sum(votes[SHA].values()) == 1


def test_two_installs_agreeing_cast_two_votes(collect):
    votes, _src = collect(('iid1', 'BOFFS', SHA), ('iid2', 'BOFFS', SHA))
    assert votes[SHA]['BOFFS'] == 2


def test_two_installs_disagreeing_are_counted_apart(collect):
    votes, _src = collect(('iid1', 'BOFFS', SHA), ('iid2', 'SPACE_BOFFS', SHA))
    assert votes[SHA] == {'BOFFS': 1, 'SPACE_BOFFS': 1}


# ── The drain ─────────────────────────────────────────────────────────────

def test_both_copies_are_recorded_so_both_can_be_drained(collect):
    """Recording only the voting path left the other in staging for ever, and
    a surviving copy makes the next run a coin toss over the label."""
    _votes, src = collect(('iid1', 'BOFFS', SHA), ('iid1', 'SPACE_BOFFS', SHA))
    assert sorted(src[SHA]['iid1']) == [
        f'staging/iid1/screen_types/BOFFS/{SHA}.png',
        f'staging/iid1/screen_types/SPACE_BOFFS/{SHA}.png',
    ]


def test_a_single_copy_is_recorded_as_a_list_of_one(collect):
    _votes, src = collect(('iid1', 'BOFFS', SHA))
    assert src[SHA]['iid1'] == [f'staging/iid1/screen_types/BOFFS/{SHA}.png']


def test_every_install_keeps_its_own_paths(collect):
    _votes, src = collect(('iid1', 'BOFFS', SHA), ('iid2', 'SPACE_BOFFS', SHA))
    assert set(src[SHA]) == {'iid1', 'iid2'}
    assert src[SHA]['iid2'] == [
        f'staging/iid2/screen_types/SPACE_BOFFS/{SHA}.png']


def test_a_type_outside_the_whitelist_is_not_recorded_as_a_source(collect):
    """It can never be promoted, so it can never be drained by the promotion
    path either — `_sweep_unpromotable` owns it instead."""
    _votes, src = collect(('iid1', 'UNKNOWN', SHA))
    assert SHA not in src or not src[SHA]
