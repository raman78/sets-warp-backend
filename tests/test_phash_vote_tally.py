"""A pHash vote, once counted, keeps counting — once per install.

`admin_merge` used to tally only the contributions that arrived since its last
run, then mark them processed. A dissenting vote that fell short was never
seen again, and overturning an entry needed two matching votes inside one run
— in practice never. Measured 2026-09-25 against the live repo: 205 processed
contributions disagreed with the table and had been forgotten.

The tally that replaced it counted contribution files, so one install voting
the same name 27 times for one hash counted 27. Since 2026-10-01 the record is
`voters`, phash → {install_id: [name, timestamp]}: one vote per install, its
latest.

A hash also does not identify one picture: one carried votes for five
different items. So the tally keeps every name it is given, and the
`knowledge` map carries only its leader.

Offline: no HF, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import admin_merge


def _c(ph: str, name: str, iid: str = 'i1', at: str = '2026-10-01T00:00:00Z',
       cid: str = '') -> dict:
    return {'phash': ph, 'item_name': name, 'confirmed': True,
            'install_id': iid, 'timestamp': at,
            'contribution_id': cid or f'{ph}-{name}-{iid}-{at}'}


def _run(contribs, existing, voters=None, min_votes=2):
    return admin_merge.merge(contribs, existing, min_votes=min_votes,
                             voters=voters)


def _votes(voters, merged):
    return admin_merge.votes_from_voters(voters, merged)


# ── Votes are remembered ───────────────────────────────────────────────────

def test_a_vote_that_falls_short_is_kept():
    merged, report, voters, _ = _run([_c('aa', 'Precision', 'i2')],
                                     {'aa': 'D.O.M.I.N.O.'},
                                     voters={'aa': {'i1': ['D.O.M.I.N.O.', '']}})

    assert report[0]['action'] == 'SKIP'
    assert _votes(voters, merged)['aa'] == {'D.O.M.I.N.O.': 1, 'Precision': 1}


def test_votes_from_separate_runs_add_up():
    """The reported case: one dissenting vote per run used to be forgotten
    each time, so the entry could never change."""
    existing = {'aa': 'D.O.M.I.N.O.'}
    _, _, voters, _ = _run([_c('aa', 'Precision', 'i1')], existing)
    merged, report, voters, _ = _run([_c('aa', 'Precision', 'i2')], existing, voters)

    assert _votes(voters, merged)['aa']['Precision'] == 2
    assert report[0]['action'] == 'UPDATE'
    assert merged['aa'] == 'Precision'


def test_agreeing_votes_strengthen_the_entry():
    merged, report, voters, _ = _run(
        [_c('aa', 'D.O.M.I.N.O.', 'i2')], {'aa': 'D.O.M.I.N.O.'},
        voters={'aa': {'i1': ['D.O.M.I.N.O.', '']}})

    assert report[0]['action'] == 'unchanged'
    assert _votes(voters, merged)['aa'] == {'D.O.M.I.N.O.': 2}


# ── One install, one vote ──────────────────────────────────────────────────

def test_one_install_voting_again_counts_once():
    """Measured 2026-10-01: one install had voted "Precision" 27 times for
    one hash. Counted per file, that alone overturned the entry."""
    repeats = [_c('aa', 'Precision', 'i1', f'2026-03-{d:02d}T00:00:00Z')
               for d in range(1, 28)]
    merged, report, voters, _ = _run(repeats, {'aa': 'D.O.M.I.N.O.'})

    assert merged['aa'] == 'D.O.M.I.N.O.'
    assert _votes(voters, merged)['aa'] == {'Precision': 1}


def test_an_install_s_latest_vote_replaces_its_earlier_one():
    merged, _, voters, _ = _run(
        [_c('aa', 'Chroniton Mine', 'i1', '2026-06-11T00:00:00Z'),
         _c('aa', 'Isomagnetic Console', 'i1', '2026-09-05T00:00:00Z')], {})

    assert voters['aa'] == {'i1': ['Isomagnetic Console', '2026-09-05T00:00:00Z']}
    assert merged['aa'] == 'Isomagnetic Console'


def test_an_older_vote_never_replaces_a_newer_one():
    """Runs may process contributions out of order, or the same one twice;
    neither may change the record."""
    newer = {'aa': {'i1': ['Isomagnetic Console', '2026-09-05T00:00:00Z']}}
    _, _, voters, _ = _run(
        [_c('aa', 'Chroniton Mine', 'i1', '2026-06-11T00:00:00Z')],
        {'aa': 'Isomagnetic Console'}, voters=newer)

    assert voters['aa'] == newer['aa']


def test_a_virtual_class_vote_withdraws_the_install_s_earlier_vote():
    """Measured 2026-10-01: an install voted "Charged Particle Burst" once,
    then `__inactive__`. Skipping the virtual vote kept the withdrawn one,
    which then entered the table as a NEW entry."""
    merged, _, voters, _ = _run(
        [_c('aa', 'Charged Particle Burst', 'i1', '2026-04-04T20:36:49Z'),
         _c('aa', '__inactive__',          'i1', '2026-04-04T20:36:50Z')], {})

    assert 'aa' not in merged
    assert 'aa' not in _votes(voters, merged)


# ── What it takes to change an entry ───────────────────────────────────────

def test_a_tie_keeps_the_current_name():
    merged, report, _, _ = _run(
        [_c('aa', 'Precision', 'i2')], {'aa': 'D.O.M.I.N.O.'},
        voters={'aa': {'i1': ['D.O.M.I.N.O.', '']}}, min_votes=1)

    assert merged['aa'] == 'D.O.M.I.N.O.'
    assert report[0]['action'] == 'SKIP'   # a dissent that has not won yet


def test_a_challenger_needs_more_installs_than_the_current_name():
    current = {'aa': {f'k{i}': ['D.O.M.I.N.O.', ''] for i in range(4)}}
    merged, _, _, _ = _run(
        [_c('aa', 'Precision', f'p{i}') for i in range(3)], {'aa': 'D.O.M.I.N.O.'},
        voters=current)

    assert merged['aa'] == 'D.O.M.I.N.O.'


def test_the_minimum_still_guards_an_update():
    """`--min` keeps its meaning: it counts installs, and one install alone
    never overturns an entry, even against an entry that also has one."""
    merged, report, _, _ = _run([_c('aa', 'Precision', 'i1'), _c('aa', 'Precision', 'i2')],
                                {'aa': 'D.O.M.I.N.O.'}, min_votes=3)

    assert report[0]['action'] == 'SKIP'
    assert merged['aa'] == 'D.O.M.I.N.O.'


def test_a_new_hash_enters_on_one_vote():
    merged, report, voters, _ = _run([_c('bb', 'Precision')], {})

    assert report[0]['action'] == 'NEW'
    assert merged['bb'] == 'Precision'
    assert _votes(voters, merged)['bb'] == {'Precision': 1}


# ── An entry nobody is recorded as voting for ──────────────────────────────

def test_an_entry_without_voters_counts_one_vote():
    _, _, voters, _ = _run([_c('bb', 'Other')], {'aa': 'D.O.M.I.N.O.'})

    assert _votes(voters, {'aa': 'D.O.M.I.N.O.'})['aa'] == {'D.O.M.I.N.O.': 1}


def test_every_name_a_hash_was_given_stays_in_the_tally():
    """One hash, several pictures: the tally is what lets a client choose."""
    names = ['Complex Plasma Fires', 'Nanoenergy Cell', 'Aetherian Dual Beam Bank']
    existing = {'cc': 'Revolutionary Combat Impulse Engine'}
    merged, _, voters, _ = _run(
        [_c('cc', n, f'i{k}') for k, n in enumerate(names)], existing,
        voters={'cc': {'x': ['Revolutionary Combat Impulse Engine', '']}})

    assert set(_votes(voters, merged)['cc']) == set(names) | {'Revolutionary Combat Impulse Engine'}


def test_a_virtual_name_never_counts_for_a_name():
    merged, _, voters, _ = _run(
        [_c('aa', '__empty__', 'i2')], {'aa': 'D.O.M.I.N.O.'},
        voters={'aa': {'i1': ['D.O.M.I.N.O.', ''], 'i3': ['__inactive__', '']}})

    assert set(_votes(voters, merged)['aa']) == {'D.O.M.I.N.O.'}


# ── What the drain may delete ──────────────────────────────────────────────

def test_a_skip_contribution_is_drained_too():
    """Its vote is in `voters`, and the file is never read again. Kept, it was
    an orphan to the staging audit: 671 of them on 2026-10-01."""
    contribs = [_c('aa', 'Precision', 'i2', cid='dissent'),
                _c('bb', 'Precision', 'i2', cid='new')]
    _, report, _, by_phash = _run(contribs, {'aa': 'D.O.M.I.N.O.'},
                                  voters={'aa': {'i1': ['D.O.M.I.N.O.', '']}})
    paths = [Path('contributions/2026-10-01/dissent.json'),
             Path('contributions/2026-10-01/new.json'),
             Path('contributions/2026-10-01/pending.json')]

    drained = admin_merge._contributions_to_drain(contribs, by_phash, paths)

    assert {r['phash']: r['action'] for r in report} == {'aa': 'SKIP', 'bb': 'NEW'}
    assert sorted(p.stem for p in drained) == ['dissent', 'new']


def test_a_contribution_without_install_is_not_counted(capsys):
    c = _c('aa', 'Precision', cid='orphan')
    c['install_id'] = ''
    merged, _, voters, by_phash = _run([c], {})

    assert merged == {} and voters == {} and by_phash == {}
    assert 'orphan' in capsys.readouterr().out


# ── Writing over a table without voters ────────────────────────────────────

def test_the_merger_refuses_a_table_without_voters(monkeypatch, capsys):
    """Schema 3 counted files: merging over it would keep counting repeats."""
    def _unreachable(*a, **k):
        raise AssertionError('merged past the refusal')

    monkeypatch.setattr(admin_merge, '_hf_load_state',
                        lambda: ({'aa': 'X'}, set(), '', None))
    # If the refusal regresses, the run must stop here, not at the real HF
    # listing and save (2026-10-01: it reached them and wrote production).
    monkeypatch.setattr(admin_merge, '_hf_list_contributions', _unreachable)
    monkeypatch.setattr(admin_merge, '_hf_save_state', _unreachable)
    monkeypatch.setattr('sys.argv', ['admin_merge.py', '--apply'])

    with pytest.raises(SystemExit) as exc:
        admin_merge.main()

    assert exc.value.code == 1
    assert 'admin_rebuild_votes.py' in capsys.readouterr().err


# ── The scrub tool must not leave a removed name behind ────────────────────

def test_a_scrubbed_name_leaves_the_tally_and_the_voters(monkeypatch):
    """Otherwise its old votes would restore it the next time anyone voted
    on that hash, or the next time voters were rebuilt."""
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
                'votes': {'aa': {'Charged Particle Burst': 2, 'Tachyon Beam': 1},
                          'bb': {'Precision': 1}},
                'voters': {'aa': {'i1': ['Charged Particle Burst', ''],
                                  'i2': ['Charged Particle Burst', ''],
                                  'i3': ['Tachyon Beam', '']},
                           'bb': {'i1': ['Precision', '']}}}

    assert scrub._save_cleaned(envelope, {'bb': 'Precision'})
    assert sent['votes'] == {'aa': {'Tachyon Beam': 1}, 'bb': {'Precision': 1}}
    assert sent['voters'] == {'aa': {'i3': ['Tachyon Beam', '']},
                              'bb': {'i1': ['Precision', '']}}
