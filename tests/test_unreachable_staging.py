"""Staging must empty itself, including what the tally can never reach.

The drain removes what was promoted. Anything the tally refuses to consider
is therefore invisible to it and stays for good — and three such classes were
sitting in production when this was written, none of which any scheduled job
would ever have cleared:

    29  screens typed `UNKNOWN`, which is not in the whitelist. The client
        stopped uploading them on 2026-08-13 and the endpoint refuses them,
        but what arrived before that stayed.
     2  annotation rows whose crop exists nowhere — not in staging, not in
        `data/`. The mirror of the orphan PNG the sweep already handled, and
        the only direction it did not cover.
     4  rows and 2 crops barred by the review ledger. The maintainer rejected
        them, so the tally skips them, so nothing drains them — the rejection
        would be re-litigated every run and the files kept for good.

Run standalone:
    python -m pytest tests/test_unreachable_staging.py -v
"""
from __future__ import annotations

import democratic_merge_crops as crops
import democratic_merge_screens as screens


# ── Screens: a type the merger will never accept ───────────────────────────

def test_unknown_is_not_a_mergeable_screen_type():
    """The premise of the sweep. If `UNKNOWN` were ever whitelisted the sweep
    would start deleting real votes, so this pins the assumption."""
    assert 'UNKNOWN' not in screens.SCREEN_TYPES


def test_every_whitelisted_type_stays_mergeable():
    """The sweep must key on the whitelist, not on a hardcoded name."""
    for stype in ('SPACE_EQ', 'BOFFS', 'DISCARD', 'SPACE_SKILLS'):
        assert stype in screens.SCREEN_TYPES


# ── Crops: rows and PNGs the tally cannot reach ────────────────────────────

def _records(*shas):
    return {'iid': [{'crop_sha256': s, 'name': '__inactive__'} for s in shas]}


def test_a_row_whose_crop_exists_nowhere_is_swept(monkeypatch):
    """Tallying reads the PNG's bytes, so a row without one can never be
    promoted, and the promotion drain never reaches it."""
    kept = crops._surviving_rows(
        records=_records('aa', 'bb'),
        staged_shas={'aa'},
        existing={},
        safe_promoted=set(),
        barred=set(),
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa']


def test_a_row_whose_crop_is_already_in_data_is_kept():
    """Promoted crops live in `data/`, not staging — that is not an orphan."""
    kept = crops._surviving_rows(
        records=_records('aa'),
        staged_shas=set(),
        existing={'aa': {'name': 'x'}},
        safe_promoted=set(),
        barred=set(),
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa']


def test_a_barred_row_is_swept_rather_than_re_litigated():
    kept = crops._surviving_rows(
        records=_records('aa', 'bb'),
        staged_shas={'aa', 'bb'},
        existing={},
        safe_promoted=set(),
        barred={'bb'},
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa']


def test_a_promoted_row_is_still_drained_as_before():
    """The behaviour that already existed must not change."""
    kept = crops._surviving_rows(
        records=_records('aa', 'bb'),
        staged_shas={'aa', 'bb'},
        existing={},
        safe_promoted={'bb'},
        barred=set(),
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa']


def test_a_row_with_no_sha_at_all_is_swept():
    kept = crops._surviving_rows(
        records={'iid': [{'name': '__empty__'}]},
        staged_shas=set(),
        existing={},
        safe_promoted=set(),
        barred=set(),
    )

    assert kept['iid'] == []


def test_a_healthy_row_survives_every_sweep():
    """The direction that matters most: a normal pending vote must stay."""
    kept = crops._surviving_rows(
        records=_records('aa'),
        staged_shas={'aa'},
        existing={},
        safe_promoted=set(),
        barred=set(),
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa']
