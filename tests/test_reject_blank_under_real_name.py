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
