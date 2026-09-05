"""A crop's name is decided among the voters who agreed what slot it is.

The published record used to be assembled from two independent elections on
the same picture: `name` from the name ballot, `slot` from the slot ballot.
Nothing required the two to come from the same voters, so the pair could be
one nobody had submitted.

That is not hypothetical. Measured 2026-09-05 in the published
`data/annotations.jsonl`:

    {"crop_sha256": "0ca800ab...", "name": "Fleet Yamaguchi Support Cruiser",
     "slot": "Ship Tier", "votes": 5, "losers": {"T6-X2": 1}}

A ship's class name recorded as its *tier*, five votes strong, with the one
correct answer listed as the loser. The cause is upstream: when no tier badge
is found on screen, the client gives the `Ship Tier` row the same bounding box
as the `Ship Type` row, so both rows produce identical pixels, an identical
hash, and land in one ballot.

Offline: no HF, no network.
"""
from __future__ import annotations

from collections import Counter

import pytest

import democratic_merge_crops as merge


SHA = 'a' * 64


def _merge(name_votes, slot_votes, existing=None):
    return merge._merge(name_votes, slot_votes, existing or {},
                        verbose=False, rejected=set())


# ── The production case ────────────────────────────────────────────────────

@pytest.fixture
def shared_crop(capsys):
    """One picture, uploaded under two slots — the class line and the tier.

    The printed output is captured here rather than in the test bodies: the
    merge runs during fixture setup, so a later `readouterr` would see nothing
    and the "is it reported?" test would pass against an empty string.
    """
    name_votes = {SHA: {
        'Ship Type': Counter({'Fleet Yamaguchi Support Cruiser': 5}),
        'Ship Tier': Counter({'T6-X2': 1}),
    }}
    slot_votes = {SHA: Counter({'Ship Type': 5, 'Ship Tier': 1})}
    merged, report, _promoted = _merge(name_votes, slot_votes)
    return merged[SHA], report, capsys.readouterr().out


def test_the_name_and_the_slot_come_from_the_same_voters(shared_crop):
    rec, _report, _out = shared_crop
    assert (rec['name'], rec['slot']) == (
        'Fleet Yamaguchi Support Cruiser', 'Ship Type')


def test_a_class_name_is_never_recorded_as_a_tier(shared_crop):
    rec, _report, _out = shared_crop
    assert not (rec['slot'] == 'Ship Tier'
                and rec['name'] == 'Fleet Yamaguchi Support Cruiser')


def test_the_other_slots_votes_are_not_listed_as_losers(shared_crop):
    """`T6-X2` did not lose the class-line election — it was never in it."""
    rec, _report, _out = shared_crop
    assert 'T6-X2' not in (rec.get('losers') or {})


def test_the_minority_slot_is_reported_not_dropped_in_silence(shared_crop):
    """Rule: a rejection is surfaced. A crop claimed by two slots means a
    client is still sending one picture for two rows."""
    _rec, _report, out = shared_crop
    assert SHA[:12] in out and 'Ship Tier=1' in out


# ── The winning slot decides, whichever it is ──────────────────────────────

def test_the_tier_wins_when_the_tier_voters_are_the_majority():
    """Same collision, opposite majority: the record must follow the slot,
    not a name that happens to be more popular overall."""
    name_votes = {SHA: {
        'Ship Type': Counter({'Fleet Yamaguchi Support Cruiser': 2}),
        'Ship Tier': Counter({'T6-X2': 5}),
    }}
    slot_votes = {SHA: Counter({'Ship Type': 2, 'Ship Tier': 5})}
    merged, _r, _p = _merge(name_votes, slot_votes)
    assert (merged[SHA]['name'], merged[SHA]['slot']) == ('T6-X2', 'Ship Tier')


def test_votes_are_counted_within_the_slot_not_across_it():
    """Three voices for the losing slot must not out-vote two for the winner
    once the slot is settled."""
    name_votes = {SHA: {
        'Devices': Counter({'Shields Battery': 2}),
        'Fore Weapons': Counter({'Phaser Beam Array': 3}),
    }}
    slot_votes = {SHA: Counter({'Devices': 4, 'Fore Weapons': 3})}
    merged, _r, _p = _merge(name_votes, slot_votes)
    assert merged[SHA]['name'] == 'Shields Battery'
    assert merged[SHA]['votes'] == 2


# ── Entries that state no slot ─────────────────────────────────────────────

def test_an_entry_without_a_slot_still_counts():
    """Not stating a slot is not asserting a different one, so it counts
    toward whatever the crop turns out to be — dropping it would lose a vote
    for no reason."""
    name_votes = {SHA: {
        'Devices': Counter({'Shields Battery': 1}),
        '': Counter({'Shields Battery': 2}),
    }}
    slot_votes = {SHA: Counter({'Devices': 1})}
    merged, _r, _p = _merge(name_votes, slot_votes)
    assert merged[SHA]['votes'] == 3


def test_a_crop_with_no_slot_anywhere_is_still_merged():
    """Older staging entries carry no slot at all; they must not vanish."""
    name_votes = {SHA: {'': Counter({'Deflector Array': 2})}}
    merged, _r, _p = _merge(name_votes, {})
    assert merged[SHA]['name'] == 'Deflector Array'
    assert merged[SHA]['slot'] == ''


def test_a_slot_with_no_names_produces_no_record():
    """Guard against an empty ballot reaching most_common()."""
    name_votes = {SHA: {'Devices': Counter()}}
    merged, _r, _p = _merge(name_votes, {SHA: Counter({'Devices': 1})})
    assert SHA not in merged


# ── The split helper on its own ────────────────────────────────────────────

def test_ballot_for_folds_in_the_unstated_slot():
    ballot, dropped = merge._ballot_for(
        {'Devices': Counter({'X': 1}), '': Counter({'X': 2})}, 'Devices')
    assert ballot == Counter({'X': 3})
    assert dropped == {}


def test_ballot_for_sets_aside_the_other_slots():
    ballot, dropped = merge._ballot_for(
        {'Devices': Counter({'X': 1}), 'Ship Tier': Counter({'T6': 4})},
        'Devices')
    assert ballot == Counter({'X': 1})
    assert dropped == {'Ship Tier': Counter({'T6': 4})}
