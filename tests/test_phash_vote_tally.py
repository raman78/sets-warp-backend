"""A pHash vote, once counted, keeps counting.

`admin_merge` used to tally only the contributions that arrived since its last
run, then mark them processed. A dissenting vote that fell short was never
seen again, and overturning an entry needed two matching votes inside one run
— in practice never. Measured 2026-09-25 against the live repo: 205 processed
contributions disagreed with the table and had been forgotten.

A hash also does not identify one picture: one carried votes for five
different items. So the tally keeps every name it is given, and the
`knowledge` map carries only its leader.

Offline: no HF, no network.
"""
from __future__ import annotations

import admin_merge


def _c(ph: str, name: str, cid: str = '') -> dict:
    return {'phash': ph, 'item_name': name, 'confirmed': True,
            'contribution_id': cid or f'{ph}-{name}'}


def _run(contribs, existing, votes=None, min_votes=2):
    return admin_merge.merge(contribs, existing, min_votes=min_votes,
                             votes=votes)


# ── Votes are remembered ───────────────────────────────────────────────────

def test_a_vote_that_falls_short_is_kept():
    _, report, votes, _ = _run([_c('aa', 'Precision')], {'aa': 'D.O.M.I.N.O.'})

    assert report[0]['action'] == 'SKIP'
    assert votes['aa'] == {'D.O.M.I.N.O.': 1, 'Precision': 1}


def test_votes_from_separate_runs_add_up():
    """The reported case: one dissenting vote per run used to be forgotten
    each time, so the entry could never change."""
    existing = {'aa': 'D.O.M.I.N.O.'}
    _, _, votes, _ = _run([_c('aa', 'Precision', 'r1')], existing)
    merged, report, votes, _ = _run([_c('aa', 'Precision', 'r2')], existing, votes)

    assert votes['aa']['Precision'] == 2
    assert report[0]['action'] == 'UPDATE'
    assert merged['aa'] == 'Precision'


def test_agreeing_votes_strengthen_the_entry():
    _, report, votes, _ = _run([_c('aa', 'D.O.M.I.N.O.')], {'aa': 'D.O.M.I.N.O.'})

    assert report[0]['action'] == 'unchanged'
    assert votes['aa'] == {'D.O.M.I.N.O.': 2}


# ── What it takes to change an entry ───────────────────────────────────────

def test_a_tie_keeps_the_current_name():
    merged, report, _, _ = _run(
        [_c('aa', 'Precision')], {'aa': 'D.O.M.I.N.O.'},
        votes={'aa': {'D.O.M.I.N.O.': 1}}, min_votes=1)

    assert merged['aa'] == 'D.O.M.I.N.O.'
    assert report[0]['action'] == 'SKIP'   # a dissent that has not won yet


def test_a_challenger_needs_more_votes_than_the_current_name():
    merged, _, _, _ = _run(
        [_c('aa', 'Precision', f'p{i}') for i in range(3)], {'aa': 'D.O.M.I.N.O.'},
        votes={'aa': {'D.O.M.I.N.O.': 4}})

    assert merged['aa'] == 'D.O.M.I.N.O.'


def test_the_minimum_still_guards_an_update():
    """`--min` keeps its meaning: one vote alone never overturns an entry,
    even against an entry that also has one."""
    merged, report, _, _ = _run([_c('aa', 'Precision'), _c('aa', 'Precision', 'x')],
                                {'aa': 'D.O.M.I.N.O.'}, min_votes=3)

    assert report[0]['action'] == 'SKIP'
    assert merged['aa'] == 'D.O.M.I.N.O.'


def test_a_new_hash_enters_on_one_vote():
    merged, report, votes, _ = _run([_c('bb', 'Precision')], {})

    assert report[0]['action'] == 'NEW'
    assert merged['bb'] == 'Precision'
    assert votes['bb'] == {'Precision': 1}


# ── Starting from a table written before the tally ─────────────────────────

def test_an_entry_without_a_tally_starts_at_one_vote():
    """The votes that produced today's entries were not kept."""
    _, _, votes, _ = _run([_c('bb', 'Other')], {'aa': 'D.O.M.I.N.O.'})

    assert votes['aa'] == {'D.O.M.I.N.O.': 1}


def test_every_name_a_hash_was_given_stays_in_the_tally():
    """One hash, several pictures: the tally is what lets a client choose."""
    names = ['Complex Plasma Fires', 'Nanoenergy Cell', 'Aetherian Dual Beam Bank']
    _, _, votes, _ = _run([_c('cc', n) for n in names],
                          {'cc': 'Revolutionary Combat Impulse Engine'})

    assert set(votes['cc']) == set(names) | {'Revolutionary Combat Impulse Engine'}


def test_a_virtual_name_never_enters_the_tally():
    _, _, votes, _ = _run([_c('aa', '__empty__')], {'aa': 'D.O.M.I.N.O.'},
                          votes={'aa': {'D.O.M.I.N.O.': 1, '__inactive__': 9}})

    assert set(votes['aa']) == {'D.O.M.I.N.O.'}


# ── The scrub tool must not leave a removed name in the tally ──────────────

def test_a_scrubbed_name_leaves_the_tally(monkeypatch):
    """Otherwise its old votes would restore it the next time anyone voted
    on that hash."""
    import huggingface_hub
    import admin_scrub_knowledge as scrub

    sent = {}

    class _Api:
        def __init__(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj, **k):
            import json
            sent.update(json.loads(path_or_fileobj))

    monkeypatch.setattr(huggingface_hub, 'HfApi', _Api)
    envelope = {'knowledge': {'aa': 'Charged Particle Burst', 'bb': 'Precision'},
                'votes': {'aa': {'Charged Particle Burst': 20, 'Tachyon Beam': 1},
                          'bb': {'Precision': 3}}}

    assert scrub._save_cleaned(envelope, {'bb': 'Precision'})
    assert sent['votes'] == {'aa': {'Tachyon Beam': 1}, 'bb': {'Precision': 3}}
