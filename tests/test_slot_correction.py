"""A crop filed under the wrong slot is re-filed, not thrown away.

When no tier badge was found on screen the client gave the `Ship Tier` row the
same bounding box as the `Ship Type` row. Identical pixels give an identical
hash, so both rows landed in one ballot and a record could end up carrying a
class line's picture, the class's name, and `slot: Ship Tier`.

Only one field is wrong. REJECT would delete a good picture of a ship's class
line — often the only copy — so the review tool grew a fourth verb: `SLOT`.

Offline: `apply` is exercised through its parsing and validation, which is
where the decisions are refused; the HF commit itself is not reached.
"""
from __future__ import annotations

import pytest

import admin_reject_crops as rej


def _tsv(tmp_path, *lines):
    p = tmp_path / 'decisions.tsv'
    p.write_text('# decision\tidx\tsha\tlabel\tslot\n'
                 + '\n'.join(lines) + '\n', encoding='utf-8')
    return p


# ── Parsing ────────────────────────────────────────────────────────────────

def test_a_slot_decision_is_parsed(tmp_path):
    d = rej.read_decisions_tsv(_tsv(tmp_path, 'SLOT Ship Type\t1\taa\tname\tShip Tier'))
    assert d == [{'sha': 'aa', 'decision': 'SLOT',
                  'relabel_name': '', 'new_slot': 'Ship Type'}]


def test_a_slot_with_spaces_keeps_them(tmp_path):
    """Every real slot name has a space in it — `Fore Weapons`, `Ship Type`."""
    d = rej.read_decisions_tsv(_tsv(tmp_path, 'SLOT Engineering Consoles\t1\taa\tn\ts'))
    assert d[0]['new_slot'] == 'Engineering Consoles'


def test_a_slot_without_a_value_is_skipped(tmp_path):
    """A bare `SLOT` would otherwise blank the field."""
    assert rej.read_decisions_tsv(_tsv(tmp_path, 'SLOT\t1\taa\tn\ts')) == []


def test_a_relabel_without_a_name_is_still_skipped(tmp_path):
    assert rej.read_decisions_tsv(_tsv(tmp_path, 'RELABEL\t1\taa\tn\ts')) == []


def test_the_other_verbs_are_unchanged(tmp_path):
    d = rej.read_decisions_tsv(_tsv(
        tmp_path,
        'REJECT\t1\taa\tn\ts',
        'KEEP\t2\tbb\tn\ts',
        'RELABEL Console - Tactical - Phaser Relay\t3\tcc\tn\ts'))
    assert [x['decision'] for x in d] == ['REJECT', 'KEEP', 'RELABEL']
    assert d[2]['relabel_name'] == 'Console - Tactical - Phaser Relay'


def test_an_unknown_verb_is_skipped(tmp_path):
    assert rej.read_decisions_tsv(_tsv(tmp_path, 'MOVE Ship Type\t1\taa\tn\ts')) == []


# ── Validation refuses before it commits ───────────────────────────────────

class _Api:
    """Fails loudly if `apply` ever reaches the commit — these cases must not."""
    def create_commit(self, *a, **kw):        # pragma: no cover
        raise AssertionError('commit attempted despite an invalid decision')


@pytest.fixture
def dataset(tmp_path):
    """A shallow-clone lookalike holding one mis-filed record."""
    d = tmp_path / 'data'
    d.mkdir(parents=True)
    (d / 'annotations.jsonl').write_text(
        '\n'.join([
            '{"crop_sha256": "aa", "name": "Fleet Yamaguchi Support Cruiser",'
            ' "slot": "Ship Tier", "votes": 5}',
            '{"crop_sha256": "bb", "name": "Phaser Beam Array",'
            ' "slot": "Fore Weapons", "votes": 3}',
        ]) + '\n', encoding='utf-8')
    (d / 'reviewed_virtual.jsonl').write_text('', encoding='utf-8')
    return tmp_path


def _apply(dataset, decisions):
    return rej.apply(dataset, decisions, _Api(), set(), {'X'})


def test_a_slot_the_dataset_never_uses_is_refused(dataset, capsys):
    """Catches a typo without a hard-coded slot list that would drift."""
    ok = _apply(dataset, [{'sha': 'aa', 'decision': 'SLOT',
                           'relabel_name': '', 'new_slot': 'Ship Typ'}])
    assert ok is False
    assert 'not a slot this dataset uses' in capsys.readouterr().err


def test_a_slot_whose_vocabulary_rejects_the_name_is_refused(dataset, capsys):
    """Spelled fine and wrong anyway: a ship's name is not a weapon."""
    ok = _apply(dataset, [{'sha': 'aa', 'decision': 'SLOT',
                           'relabel_name': '', 'new_slot': 'Fore Weapons'}])
    assert ok is False
    assert 'does not resolve' in capsys.readouterr().err


def test_one_bad_decision_refuses_the_whole_run(dataset):
    """Same rule RELABEL already follows — no partial application."""
    ok = _apply(dataset, [
        {'sha': 'aa', 'decision': 'SLOT', 'relabel_name': '',
         'new_slot': 'Ship Type'},
        {'sha': 'bb', 'decision': 'SLOT', 'relabel_name': '',
         'new_slot': 'Nonsense Slot'},
    ])
    assert ok is False
