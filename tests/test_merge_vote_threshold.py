"""Staging is a queue: tallying settles every entry, and it empties.

An entry in `staging/` has arrived from a client, has not been tallied, and is
not in the models. Tallying it settles it either way, so it is applied and
drained. Votes then express confidence *in* the record rather than gating
entry to it: they accumulate on agreement, and a superseded verdict keeps its
strength on the record so an overturn is auditable rather than silent.

The bar this replaced was "a second, independent voice to overturn" — sound
for a crowd, and equal to "never" for this project. Measured 2026-09-03: two
contributors with annotations, 4003 entries in staging of which 3897 merely
confirmed what `data/` already said and could never drain, and 102
corrections waiting indefinitely — among them a crop whose stored name,
`Attack Pattern Beta'`, no cargo row has ever matched.

Offline: no HF, no network.
"""
from __future__ import annotations

from collections import Counter

import pytest

import democratic_merge_crops as merge


def _existing(sha: str, name: str, votes: int = 5, losers: dict | None = None):
    rec = {'crop_sha256': sha, 'name': name, 'slot': 'Deflector',
           'votes': votes}
    if losers:
        rec['losers'] = losers
    return {sha: rec}


def _run(name_votes: dict, existing: dict):
    # Name votes are keyed by slot since the ballot was split per slot; these
    # cases are about the tally itself, not about slots, so they vote under the
    # "slot not stated" bucket, which counts toward whatever the crop is.
    by_slot = {sha: {'': c} for sha, c in name_votes.items()}
    return merge._merge(by_slot, {}, existing,
                        verbose=False, rejected=set())


def _action(report, sha: str) -> str:
    return next(r['action'] for r in report if r['sha'] == sha)


# ── Everything tallied is settled, so staging empties ──────────────────────

def test_a_new_crop_enters_on_one_vote():
    merged, report, promoted = _run({'aa': Counter({'Deflector Array': 1})}, {})

    assert _action(report, 'aa') == 'NEW'
    assert merged['aa']['name'] == 'Deflector Array'
    assert 'aa' in promoted


def test_a_lone_correction_is_applied():
    """The case that used to wait for a second voice that never came."""
    merged, report, promoted = _run({'aa': Counter({"Attack Pattern Beta": 1})},
                                    _existing('aa', "Attack Pattern Beta'"))

    assert _action(report, 'aa') == 'UPDATE'
    assert merged['aa']['name'] == 'Attack Pattern Beta'
    assert 'aa' in promoted


@pytest.mark.parametrize('votes,existing_name', [
    (Counter({'Deflector Array': 1}), 'Deflector Array'),      # confirms
    (Counter({'Something Else': 1}),  'Deflector Array'),      # overturns
    (Counter({'Deflector Array': 3}), None),                   # new
])
def test_every_tallied_entry_drains(votes, existing_name):
    existing = _existing('aa', existing_name) if existing_name else {}
    _merged, _report, promoted = _run({'aa': votes}, existing)

    assert 'aa' in promoted, 'left in staging, to be re-tallied for ever'


# ── Votes express confidence ───────────────────────────────────────────────

def test_agreement_accumulates():
    """Five confirmations must not read as one; the number is the whole
    signal the tail review sorts on."""
    merged, _r, _p = _run({'aa': Counter({'Deflector Array': 2})},
                          _existing('aa', 'Deflector Array', votes=5))

    assert merged['aa']['votes'] == 7


def test_an_overturned_verdict_starts_from_its_own_count():
    merged, _r, _p = _run({'aa': Counter({'Something Else': 1})},
                          _existing('aa', 'Deflector Array', votes=5))

    assert merged['aa']['votes'] == 1


def test_the_superseded_verdict_keeps_its_strength_on_the_record():
    """An overturn has to be reversible: without this, five votes of evidence
    vanish and nothing downstream can tell a contested entry from a settled
    one."""
    merged, _r, _p = _run({'aa': Counter({'Something Else': 1})},
                          _existing('aa', 'Deflector Array', votes=5))

    assert merged['aa']['losers']['Deflector Array'] == 5


def test_existing_dissent_survives_a_confirmation():
    merged, _r, _p = _run({'aa': Counter({'Deflector Array': 1})},
                          _existing('aa', 'Deflector Array', votes=5,
                                    losers={'Other Thing': 2}))

    assert merged['aa']['losers']['Other Thing'] == 2


def test_a_weak_entry_is_visibly_weak():
    """What the tail review is for: one vote and no corroboration."""
    merged, _r, _p = _run({'aa': Counter({'Deflector Array': 1})}, {})

    assert merged['aa']['votes'] == 1
    assert 'losers' not in merged['aa']
