"""Rejecting a sample removes it; it does not blacklist the picture.

A REJECT used to bar its `crop_sha256` from `data/` for good — the tally
skipped the vote and the merge dropped the record, whatever a later upload
said. That conflated two statements, the same way the name and slot ballots
did before they were split: the rejection is keyed on the *picture*, but a
picture is not a claim. The claim is the picture plus the label plus the slot.

It bit on a real record. `Fleet Yamaguchi Support Cruiser` was published with
`slot: Ship Tier` because the client gave the tier row the class line's box, so
both rows shared one crop hash. Rejecting it would also have barred that class
line from ever being contributed as a `Ship Type` — a good sample, and the only
copy of it in the dataset.

So a REJECT now means what it says: the sample is gone as though it had never
been submitted. `admin_reject_crops.apply` deletes the record, the crop PNG and
the staging copies, so there is nothing left to re-tally. A later confirmation
of the same picture is fresh human input and counts like any other.

Offline: no HF, no network.
"""
from __future__ import annotations

import json
from collections import Counter

import democratic_merge_crops as merge


SHA = 'a' * 64


def _ledger(tmp_path, *rows):
    d = tmp_path / 'data'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'reviewed_virtual.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return tmp_path


# ── The tally no longer consults the ledger for rejections ─────────────────

def test_a_re_uploaded_crop_is_tallied_again():
    """The whole point: a fresh vote on a previously rejected picture is a
    fresh proposition, not a re-litigation."""
    merged, _report, promoted = merge._merge(
        {SHA: {'Devices': Counter({'Shields Battery': 1})}},
        {SHA: Counter({'Devices': 1})},
        {}, verbose=False)

    assert merged[SHA]['name'] == 'Shields Battery'
    assert SHA in promoted


def test_an_existing_record_is_not_dropped_by_a_past_rejection():
    """The self-heal used to re-drop rejected shas from data/ on every run,
    which would undo a legitimate later contribution of the same picture."""
    existing = {SHA: {'crop_sha256': SHA, 'name': 'Shields Battery',
                      'slot': 'Devices', 'votes': 3}}

    merged, _r, _p = merge._merge({}, {}, existing, verbose=False)

    assert SHA in merged


def test_poison_names_are_still_dropped_from_existing():
    """The other half of the self-heal is unrelated to rejections and stays."""
    existing = {SHA: {'crop_sha256': SHA, 'name': 'Test Item Name',
                      'slot': 'Devices', 'votes': 1}}

    merged, _r, _p = merge._merge({}, {}, existing, verbose=False)

    assert SHA not in merged


def test_the_bar_loader_is_gone():
    """If it comes back, the semantics have quietly reverted."""
    assert not hasattr(merge, '_load_rejected_shas')


# ── What the ledger is still for ──────────────────────────────────────────

def test_a_name_correction_is_still_pinned(tmp_path):
    """Removing the bar must not weaken the RELABEL pin — that one exists
    because corrections were being overwritten within hours."""
    root = _ledger(tmp_path, {'crop_sha256': 'aa', 'name': 'Real Item',
                              'decision': 'RELABEL'})

    assert merge._load_relabelled(root) == {'aa': 'Real Item'}


def test_a_rejection_is_still_recorded_for_review(tmp_path):
    """The ledger keeps every decision so the review tool never re-surfaces a
    crop somebody already looked at. It just no longer gates the tally."""
    root = _ledger(tmp_path, {'crop_sha256': 'bb', 'name': 'Whatever',
                              'decision': 'REJECT'})

    assert merge._load_relabelled(root) == {}
    assert (root / 'data' / 'reviewed_virtual.jsonl').exists()


# ── The staging sweep no longer has a "barred" class ───────────────────────

def _records(*shas):
    return {'iid': [{'crop_sha256': s, 'name': 'X', 'slot': 'Devices'}
                    for s in shas]}


def test_a_row_for_a_rejected_crop_survives_the_sweep():
    """It is an ordinary vote now, so it drains by being tallied rather than
    by being swept."""
    kept = merge._surviving_rows(
        records=_records('aa', 'bb'),
        staged_shas={'aa', 'bb'},
        existing={},
        safe_promoted=set(),
    )

    assert [r['crop_sha256'] for r in kept['iid']] == ['aa', 'bb']
