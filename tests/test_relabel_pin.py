"""A maintainer's correction must survive the next client upload.

The review ledger records RELABEL beside REJECT, and only REJECT was ever read
back. So a correction was written into `data/` and left unguarded: the merge
is a queue, every tallied entry is applied, and the next client to upload that
crop under its old name overwrote the maintainer with a single vote.

Measured on production, 2026-09-04: `Fleet Support Cruiser (T6)` was corrected
to `Fleet Yamaguchi Support Cruiser` at 10:24 and was back under the old name
by 16:28, six hours and one merge later. Every correction made that day had
the same lifetime.

Run standalone:
    python -m pytest tests/test_relabel_pin.py -v
"""
from __future__ import annotations

import json
from collections import Counter

import democratic_merge_crops as merge


SHA = 'a' * 64


def _run(votes: dict, existing: dict | None = None, pinned: dict | None = None):
    merged, report, _ = merge._merge(
        {SHA: Counter(votes)}, {SHA: Counter({'Devices': 1})},
        existing or {}, verbose=False, rejected=set(), relabelled=pinned or {},
    )
    return merged.get(SHA, {})


# ── The pin ────────────────────────────────────────────────────────────────

def test_a_pinned_name_beats_the_incoming_vote():
    """The case that was actually happening in production."""
    rec = _run({'Fleet Support Cruiser (T6)': 1},
               pinned={SHA: 'Fleet Yamaguchi Support Cruiser'})

    assert rec['name'] == 'Fleet Yamaguchi Support Cruiser'


def test_a_pinned_name_beats_several_clients_agreeing():
    """Four installs repeating a wrong convention is what produced one of the
    labels this was built to correct, so a majority must not overturn it."""
    rec = _run({'__empty__': 4}, pinned={SHA: '__inactive__'})

    assert rec['name'] == '__inactive__'


def test_the_overruled_votes_are_still_recorded():
    """The disagreement stays visible instead of being erased — an overturn
    has to remain auditable."""
    rec = _run({'Fleet Support Cruiser (T6)': 3},
               pinned={SHA: 'Fleet Yamaguchi Support Cruiser'})

    assert rec.get('losers', {}).get('Fleet Support Cruiser (T6)')


# ── What the pin must not do ───────────────────────────────────────────────

def test_an_unpinned_sha_is_tallied_normally():
    rec = _run({'Phaser Array': 2, 'Disruptor Array': 1})

    assert rec['name'] == 'Phaser Array'


def test_a_vote_agreeing_with_the_pin_is_not_disturbed():
    rec = _run({'Fleet Yamaguchi Support Cruiser': 2},
               pinned={SHA: 'Fleet Yamaguchi Support Cruiser'})

    assert rec['name'] == 'Fleet Yamaguchi Support Cruiser'


# ── Reading the ledger ─────────────────────────────────────────────────────

def _ledger(tmp_path, *rows):
    d = tmp_path / 'data'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'reviewed_virtual.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return tmp_path


def test_only_relabel_rows_are_read(tmp_path):
    root = _ledger(
        tmp_path,
        {'crop_sha256': 'aa', 'name': 'Real Item', 'decision': 'RELABEL'},
        {'crop_sha256': 'bb', 'name': '', 'decision': 'REJECT'},
        {'crop_sha256': 'cc', 'name': 'Kept Item', 'decision': 'KEEP'},
    )

    assert merge._load_relabelled(root) == {'aa': 'Real Item'}


def test_a_later_review_supersedes_an_earlier_one(tmp_path):
    """What makes a correction correctable — re-reviewing the same crop wins."""
    root = _ledger(
        tmp_path,
        {'crop_sha256': 'aa', 'name': 'First Guess', 'decision': 'RELABEL'},
        {'crop_sha256': 'aa', 'name': 'Second Look', 'decision': 'RELABEL'},
    )

    assert merge._load_relabelled(root) == {'aa': 'Second Look'}


def test_a_relabel_with_no_name_is_ignored(tmp_path):
    """Pinning an empty name would blank the record."""
    root = _ledger(tmp_path,
                   {'crop_sha256': 'aa', 'name': '  ', 'decision': 'RELABEL'})

    assert merge._load_relabelled(root) == {}


def test_a_missing_ledger_is_not_an_error(tmp_path):
    assert merge._load_relabelled(tmp_path) == {}


def test_a_corrupt_line_does_not_lose_the_rest(tmp_path):
    d = tmp_path / 'data'
    d.mkdir(parents=True)
    (d / 'reviewed_virtual.jsonl').write_text(
        '{ broken\n'
        + json.dumps({'crop_sha256': 'aa', 'name': 'Real Item',
                      'decision': 'RELABEL'}) + '\n',
        encoding='utf-8')

    assert merge._load_relabelled(tmp_path) == {'aa': 'Real Item'}


# ── Healing a correction that was already overwritten ──────────────────────
#
# The pin only bites on the next upload of that crop. A name overwritten
# before the pin existed would stay wrong until someone photographed the same
# slot again — for a rare item, never. So `data/` is reconciled against the
# ledger on every run.

def test_an_overwritten_correction_is_restored_without_a_new_vote():
    """No staging entry for this sha at all — the heal still fixes it."""
    existing = {SHA: {'crop_sha256': SHA, 'name': 'Fleet Support Cruiser (T6)',
                      'slot': 'Ship Type', 'votes': 1}}

    merged, _, _ = merge._merge(
        {}, {}, existing, verbose=False, rejected=set(),
        relabelled={SHA: 'Fleet Yamaguchi Support Cruiser'})

    assert merged[SHA]['name'] == 'Fleet Yamaguchi Support Cruiser'
    assert merged[SHA]['relabeled_at']


def test_an_entry_already_carrying_the_pinned_name_is_left_alone():
    """Idempotent: a run that changes nothing must not rewrite timestamps."""
    existing = {SHA: {'crop_sha256': SHA, 'name': 'Right Name', 'votes': 3}}

    merged, _, _ = merge._merge(
        {}, {}, existing, verbose=False, rejected=set(),
        relabelled={SHA: 'Right Name'})

    assert merged[SHA] == existing[SHA]


def test_the_heal_does_not_invent_entries_for_absent_crops():
    """A ledger row for a sha no longer in data/ must not resurrect it."""
    merged, _, _ = merge._merge(
        {}, {}, {}, verbose=False, rejected=set(),
        relabelled={SHA: 'Some Name'})

    assert SHA not in merged
