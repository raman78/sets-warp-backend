"""The vote threshold guards changing a verdict, not confirming one.

The design: a crop nobody has seen before is taken on one voice; overturning
a verdict already in `data/` needs a second, independent one. Staging exists
to hold new evidence against the standing answer and decide whether it holds.

The threshold was applied to both, so a lone vote that *agreed* with `data/`
was refused, left in staging, and re-tallied every two hours for ever.
Measured 2026-09-03: 3897 of the 4003 entries in staging were exactly that,
`unchanged` was unreachable for a single-contributor install, and staging
could only grow.

Offline: no HF, no network.
"""
from __future__ import annotations

from collections import Counter

import pytest

import democratic_merge_crops as merge


def _existing(sha: str, name: str, votes: int = 5) -> dict:
    return {sha: {'crop_sha256': sha, 'name': name, 'slot': 'Deflector',
                  'votes': votes, 'losers': {'Other Thing': 2}}}


def _run(name_votes: dict, existing: dict, min_votes: int = 2):
    return merge._merge(name_votes, {}, existing,
                        min_votes=min_votes, verbose=False, rejected=set())


def _action(report, sha: str) -> str:
    return next(r['action'] for r in report if r['sha'] == sha)


# ── One voice is enough for something new ──────────────────────────────────

def test_a_crop_nobody_has_seen_is_taken_on_one_vote():
    merged, report, promoted = _run({'aa': Counter({'Deflector Array': 1})}, {})

    assert _action(report, 'aa') == 'NEW'
    assert merged['aa']['name'] == 'Deflector Array'
    assert 'aa' in promoted


# ── A confirming vote settles nothing, and must drain ──────────────────────

def test_a_lone_vote_that_agrees_is_accepted():
    merged, report, promoted = _run({'aa': Counter({'Deflector Array': 1})},
                                    _existing('aa', 'Deflector Array'))

    assert _action(report, 'aa') == 'unchanged'


def test_a_confirming_vote_drains_from_staging():
    """Left in staging it is re-tallied on every run, for ever — which is how
    4003 entries came to be waiting on a two-hourly job."""
    _merged, _report, promoted = _run({'aa': Counter({'Deflector Array': 1})},
                                      _existing('aa', 'Deflector Array'))

    assert 'aa' in promoted


def test_a_confirming_vote_does_not_rewrite_the_record():
    """The verdict has not changed, so neither has the evidence for it.
    Rewriting would report a consensus of 5 as a consensus of 1."""
    existing = _existing('aa', 'Deflector Array', votes=5)
    merged, _report, _promoted = _run({'aa': Counter({'Deflector Array': 1})},
                                      existing)

    assert merged['aa']['votes'] == 5
    assert merged['aa']['losers'] == {'Other Thing': 2}


# ── Overturning still needs corroboration ──────────────────────────────────

def test_a_lone_vote_cannot_overturn_a_verdict():
    merged, report, promoted = _run({'aa': Counter({'Something Else': 1})},
                                    _existing('aa', 'Deflector Array'))

    assert _action(report, 'aa') == 'SKIP'
    assert merged['aa']['name'] == 'Deflector Array'
    assert 'aa' not in promoted, 'a refused vote must stay for a second opinion'


def test_two_voices_overturn_it():
    merged, report, _promoted = _run({'aa': Counter({'Something Else': 2})},
                                     _existing('aa', 'Deflector Array'))

    assert _action(report, 'aa') == 'UPDATE'
    assert merged['aa']['name'] == 'Something Else'


@pytest.mark.parametrize('min_votes,expected', [(1, 'UPDATE'), (3, 'SKIP')])
def test_the_bar_for_overturning_follows_min_votes(min_votes, expected):
    _merged, report, _promoted = _run({'aa': Counter({'Something Else': 2})},
                                      _existing('aa', 'Deflector Array'),
                                      min_votes=min_votes)

    assert _action(report, 'aa') == expected
