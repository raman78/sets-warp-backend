"""The maintainer console's one-at-a-time review pass.

The table alone is an overview: a 64 px thumbnail is too small to tell a dim
icon from an empty cell, which is the judgement the pass exists to make. The
selected crop is shown large beside the table and the decision keys advance,
so a review is one crop at a time.

These drive the real window offscreen — no HF, no network. The crops come
from a temporary local mirror.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')
pytest.importorskip('PySide6')

import admin_console as console


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    crops = tmp_path / 'crops'
    crops.mkdir()
    rng = np.random.default_rng(0)
    for sha in ('a' * 8, 'b' * 8):
        cv2.imwrite(str(crops / f'{sha}.png'),
                    rng.integers(0, 255, (43, 33, 3), dtype=np.uint8))
    monkeypatch.setattr(console, '_fetch_crop',
                        lambda sha, token, local_dir=None:
                        cv2.imread(str(crops / f'{sha}.png')))
    monkeypatch.setattr(console, 'load_canonical_names', lambda: {'Deflector Array'})

    QApplication.instance() or QApplication([])
    w = console.AdminConsole()
    w._populate([
        {'decision': 'REJECT', 'relabel': '', 'idx': '1', 'sha': 'a' * 8,
         'label': 'Charged Particle Burst', 'slot': 'Boff Science',
         'bright': '0.020', 'rich': '0.010', 'why': 'blank-real'},
        {'decision': 'REJECT', 'relabel': '', 'idx': '2', 'sha': 'b' * 8,
         'label': '__empty__', 'slot': 'Devices',
         'bright': '0.400', 'rich': '0.380', 'why': 'colourful-virtual'},
    ])
    yield w
    w.close()


# ── The preview ────────────────────────────────────────────────────────────

def test_the_first_row_is_selected_and_previewed(window):
    assert window.table.currentRow() == 0
    assert window.preview.pixmap() is not None
    assert not window.preview.pixmap().isNull()


def test_the_preview_names_what_flagged_the_crop(window):
    """Which contradiction it is decides which way a RELABEL should go."""
    assert 'blank-real' in window.preview_info.text()


def test_the_preview_is_larger_than_the_crop(window):
    """A 33x43 cell at its own size is unreadable; it is scaled up whole."""
    assert window.preview.pixmap().width() > 33


# ── The keys ───────────────────────────────────────────────────────────────

def test_a_decision_key_sets_the_row(window):
    window._decide('KEEP')
    assert window.table.cellWidget(0, console.COL_DECISION).currentText() == 'KEEP'


def test_a_decision_key_advances_to_the_next_crop(window):
    window._decide('KEEP')
    assert window.table.currentRow() == 1


def test_the_last_crop_does_not_advance_past_the_end(window):
    window.table.setCurrentCell(1, console.COL_SHA)
    window._decide('REJECT')
    assert window.table.currentRow() == 1


def test_choosing_relabel_switches_the_decision(window):
    window._focus_relabel()
    assert window.table.cellWidget(0, console.COL_DECISION).currentText() == 'RELABEL'


# ── The round trip through the decisions file ──────────────────────────────

def test_the_reason_survives_a_write_and_re_read(tmp_path):
    """The TSV is the handover to `--apply`; a column appended after the fact
    must not shift the ones the readers index by position."""
    path = tmp_path / 'decisions.tsv'
    rows = [{'decision': 'RELABEL', 'relabel': '__inactive__', 'idx': '1',
             'sha': 'a' * 8, 'label': 'Charged Particle Burst',
             'slot': 'Boff Science', 'bright': '0.020', 'rich': '0.010',
             'why': 'blank-real'}]
    console._write_tsv(path, rows)
    back = console._parse_tsv(path)

    assert back[0]['why'] == 'blank-real'
    assert back[0]['bright'] == '0.020'
    assert back[0]['relabel'] == '__inactive__'


def test_a_file_written_before_the_reason_existed_still_parses(tmp_path):
    path = tmp_path / 'old.tsv'
    path.write_text(
        '# decision\tidx\tsha\tlabel\tslot\tbright\trich\n'
        'REJECT\t1\taaaaaaaa\tSome Item\tDevices\t0.500\t0.400\n',
        encoding='utf-8')

    row = console._parse_tsv(path)[0]

    assert row['rich'] == '0.400'
    assert row['why'] == ''
