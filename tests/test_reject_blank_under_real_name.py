"""The review tool's second direction: a blank cell filed under an item name.

The tool was built for one contradiction — a colourful crop labelled
`__empty__` — and the mirror ran unnoticed for months. It is the more
damaging of the two: a blank cell under an item's name teaches the gallery
that the item is what nothing looks like, and the recogniser then answers
with that item on every blank cell it meets.

Measured on the published mirror 2026-09-03: 25 of 9227 real-named crops are
blank cells, and 20 of them carry one name — `Charged Particle Burst`, 20 of
the 29 crops that class has. An inactive BOFF cell sits at cosine 0.92 from
those 20 and at 0.45 from the 9 genuine ones, so the recogniser's residual
confusions on inactive cells traced back to the data, not to the model.

Offline: no HF, no network. Reads pixels from a temporary local mirror.
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')

import admin_reject_crops as tool

pytestmark = pytest.mark.skipif(
    tool._load_blank_check() is None,
    reason='sto-warp not importable — the blank-cell direction is off',
)


def _inactive_cell() -> 'np.ndarray':
    """A locked BOFF cell as the game draws it: even, dim, navy."""
    hsv = np.zeros((43, 33, 3), dtype=np.uint8)
    hsv[:, :, 0] = 110
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 70
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _real_icon() -> 'np.ndarray':
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (43, 33, 3), dtype=np.uint8)


@pytest.fixture
def mirror(tmp_path):
    """A crop mirror plus the annotation rows that point at it."""
    crops = tmp_path / 'crops'
    crops.mkdir()

    def _add(sha: str, img, name: str, slot: str) -> tuple[str, dict]:
        cv2.imwrite(str(crops / f'{sha}.png'), img)
        return sha, {'name': name, 'slot': slot, 'crop_sha256': sha}

    data = dict([
        _add('a' * 8, _inactive_cell(), 'Charged Particle Burst', 'Boff Science'),
        _add('b' * 8, _real_icon(), 'Deflector Array', 'Deflector'),
        _add('c' * 8, _inactive_cell(), '__inactive__', 'Boff Science'),
        _add('d' * 8, _inactive_cell(), 'U.S.S. Whatever', 'ship_type_band'),
    ])
    return data, crops


def _scan(data, crops, ledger=None):
    return tool._scan_blank_under_real_name(
        data, ledger or {}, '', crops, False)


# ── What it flags ──────────────────────────────────────────────────────────

def test_a_blank_cell_under_an_item_name_is_flagged(mirror):
    data, crops = mirror
    assert [e['sha'] for e in _scan(data, crops)] == ['a' * 8]


def test_a_real_icon_is_not_flagged(mirror):
    data, crops = mirror
    assert all(e['sha'] != 'b' * 8 for e in _scan(data, crops))


def test_a_blank_cell_under_a_virtual_name_is_not_flagged(mirror):
    """That one is correctly labelled, and the other direction owns it."""
    data, crops = mirror
    assert all(e['sha'] != 'c' * 8 for e in _scan(data, crops))


def test_a_text_band_is_not_judged_as_a_cell(mirror):
    """Ship name / class / tier crops are wide low-contrast strips, not slot
    cells — the blank judgement does not apply to them."""
    data, crops = mirror
    assert all(e['sha'] != 'd' * 8 for e in _scan(data, crops))


# ── How it feeds the existing workflow ─────────────────────────────────────

def test_the_finding_carries_its_reason(mirror):
    """`why` is what tells the two directions apart in the montage and TSV."""
    data, crops = mirror
    assert _scan(data, crops)[0]['why'] == 'blank-real'


def test_a_kept_crop_is_not_re_surfaced(mirror):
    """The ledger is shared with the other direction: a KEEP is final."""
    data, crops = mirror
    ledger = {'a' * 8: {'decision': 'KEEP', 'name': 'Charged Particle Burst'}}
    assert _scan(data, crops, ledger) == []


# ── The tail: the least-corroborated entries ───────────────────────────────

def _entry(sha, name, votes, losers=None, slot='Deflector'):
    rec = {'crop_sha256': sha, 'name': name, 'slot': slot, 'votes': votes}
    if losers:
        rec['losers'] = losers
    return rec


@pytest.fixture
def dataset(tmp_path):
    """A crop mirror plus entries with differing vote counts."""
    crops = tmp_path / 'crops'
    crops.mkdir()
    data = {}
    for sha, votes in (('a' * 8, 7), ('b' * 8, 1), ('c' * 8, 3)):
        cv2.imwrite(str(crops / f'{sha}.png'), _real_icon())
        data[sha] = _entry(sha, f'Item {sha[0]}', votes)
    return data, crops


def test_the_weakest_entries_come_first(dataset):
    data, crops = dataset

    out = tool._scan_weakest(data, {}, '', crops, False, limit=3)

    assert [e['votes'] for e in out] == [1, 3, 7]


def test_the_limit_is_respected(dataset):
    data, crops = dataset

    assert len(tool._scan_weakest(data, {}, '', crops, False, limit=2)) == 2


def test_a_superseded_verdict_is_shown_alongside(dataset):
    """An entry that overturned a stronger one is not just another single
    vote, and the reviewer has to be able to see that."""
    data, crops = dataset
    data['b' * 8] = _entry('b' * 8, 'Item b', 1, losers={'Item was': 5})

    out = tool._scan_weakest(data, {}, '', crops, False, limit=1)

    assert 'Item was' in out[0]['why']


def test_a_virtual_label_is_not_in_the_tail(dataset):
    """Empty and inactive slots are reviewed by the other two directions;
    a lone vote on one is normal and not evidence of anything."""
    data, crops = dataset
    cv2.imwrite(str(crops / f'{"d" * 8}.png'), _inactive_cell())
    data['d' * 8] = _entry('d' * 8, '__inactive__', 1)

    out = tool._scan_weakest(data, {}, '', crops, False, limit=10)

    assert all(e['name'] != '__inactive__' for e in out)


def test_a_kept_entry_is_not_re_surfaced(dataset):
    data, crops = dataset
    ledger = {'b' * 8: {'decision': 'KEEP', 'name': 'Item b'}}

    out = tool._scan_weakest(data, ledger, '', crops, False, limit=10)

    assert all(e['sha'] != 'b' * 8 for e in out)


def test_an_entry_that_overturned_a_stronger_one_ranks_first(dataset, monkeypatch):
    """A lone vote is normal. A lone vote that replaced five is the one case
    where it is doing damage if it is wrong."""
    data, crops = dataset
    monkeypatch.setattr(tool, 'load_canonical_names',
                        lambda: {f'Item {c}' for c in 'abc'})
    data['c' * 8] = _entry('c' * 8, 'Item c', 1, losers={'Item was': 5})

    out = tool._scan_weakest(data, {}, '', crops, False, limit=3)

    assert out[0]['sha'] == 'c' * 8
    assert 'overturned-stronger' in out[0]['why']


def test_a_name_cargo_does_not_know_is_flagged(dataset, monkeypatch):
    data, crops = dataset
    monkeypatch.setattr(tool, 'load_canonical_names', lambda: {'Item a', 'Item c'})

    out = tool._scan_weakest(data, {}, '', crops, False, limit=3)

    assert 'name-not-in-cargo' in out[0]['why']
    assert out[0]['name'] == 'Item b'


def test_a_placeholder_slot_is_not_a_signal(dataset, monkeypatch):
    """`slot='migrated'` sits on 2848 of 12274 entries, so it ranks nothing
    and buried the two signals that do. It is also not a defect: every one of
    those crops is icon-shaped, so none has leaked into the k-NN pool."""
    data, crops = dataset
    monkeypatch.setattr(tool, 'load_canonical_names',
                        lambda: {f'Item {c}' for c in 'abc'})
    data['a' * 8] = _entry('a' * 8, 'Item a', 7, slot='migrated')

    out = tool._scan_weakest(data, {}, '', crops, False, limit=3)

    assert all('no-real-slot' not in e['why'] for e in out)


def test_an_unremarkable_entry_carries_no_flag(dataset, monkeypatch):
    data, crops = dataset
    monkeypatch.setattr(tool, 'load_canonical_names',
                        lambda: {f'Item {c}' for c in 'abc'})

    out = tool._scan_weakest(data, {}, '', crops, False, limit=3)

    assert out[-1]['why'].startswith('votes=')
