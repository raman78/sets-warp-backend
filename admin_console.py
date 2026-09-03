#!/usr/bin/env python3
"""
admin_console.py — maintainer GUI for WARP dataset review + cleanup
===================================================================
A thin PySide6 launcher over the existing admin_*.py CLIs. It does NOT
re-implement any mutation logic — every destructive action shells out to
the reviewed, testable command-line tool via QProcess, so the GUI and the
scripts can never drift.

Maintainer-only. Requires a HF write token (.env, same as the mergers) and
the optional `admin` extra:

    pip install -e ".[admin]"
    python admin_console.py

NEVER ship this with the client (pipx `sto-warp`): it drives write-token
operations on the server-owned dataset. It lives here, next to the admin
tools and the token.

Current tabs / actions
----------------------
- Virtual crops: run `admin_reject_crops.py --scan`, review each flagged
  crop with a thumbnail, set KEEP / REJECT / RELABEL, then Apply (runs
  `admin_reject_crops.py --apply`).
- Tools: run read-only checks — `democratic_merge_crops.py` (dry-run) and
  `admin_audit_staging.py` — streaming their output into the log pane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QCompleter, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# Reuse tool constants + the LFS-aware crop fetch + the cargo name source
# (single source of truth — RELABEL picks from sto-warp's live cargo list).
sys.path.insert(0, str(Path(__file__).parent))
from admin_reject_crops import (  # noqa: E402
    HF_TOKEN, REPO, DEFAULT_DECISIONS, DEFAULT_MONTAGE, _fetch_crop,
    load_canonical_names, local_mirror_crops_dir,
)

REPO_DIR = Path(__file__).parent
DECISIONS = ('REJECT', 'KEEP', 'RELABEL')

# Table columns.
(COL_THUMB, COL_SHA, COL_LABEL, COL_SLOT, COL_WHY, COL_STATS,
 COL_DECISION, COL_RELABEL) = range(8)
_THUMB_PX = 64
_PREVIEW_PX = 256


def _bgr_to_pixmap(bgr: np.ndarray, size: int = _THUMB_PX) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    big = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    qimg = QImage(big.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _parse_tsv(path: Path) -> list[dict]:
    """Parse the decisions TSV into full rows (decision + metadata)."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.lstrip().startswith('#'):
            continue
        p = line.split('\t')
        if len(p) < 7:
            continue
        tok = p[0].strip().split(None, 1)
        rows.append({
            'decision': tok[0].upper(),
            'relabel':  tok[1].strip() if len(tok) > 1 else '',
            'idx':      p[1].strip(),
            'sha':      p[2].strip(),
            'label':    p[3].strip(),
            'slot':     p[4].strip(),
            'bright':   p[5].strip(),
            'rich':     p[6].strip(),
            # Appended after the fact, so a TSV written before the second
            # scan direction existed still parses.
            'why':      p[7].strip() if len(p) > 7 else '',
        })
    return rows


def _write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ['# decision\tidx\tsha\tlabel\tslot\tbright\trich\twhy',
             '# decision ∈ {REJECT, KEEP, RELABEL <canonical name>}']
    for r in rows:
        verb = r['decision']
        first = f'{verb} {r["relabel"]}'.strip() if verb == 'RELABEL' else verb
        lines.append(f'{first}\t{r["idx"]}\t{r["sha"]}\t{r["label"]}\t'
                     f'{r["slot"]}\t{r["bright"]}\t{r["rich"]}\t'
                     f'{r.get("why", "")}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


class AdminConsole(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('WARP maintainer console')
        self.resize(1000, 720)
        self._proc: QProcess | None = None
        self._decisions_path = REPO_DIR / DEFAULT_DECISIONS
        self._rows: list[dict] = []
        self._images: dict[int, object] = {}
        # Cargo names for the RELABEL picker — from sto-warp's own loader.
        self._cargo = sorted(load_canonical_names())
        # Local crop mirror for thumbnails (avoids re-downloading from HF).
        self._local_crops = local_mirror_crops_dir()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── Action bar ──────────────────────────────────────────────────
        bar = QHBoxLayout()
        self.cmb_direction = QComboBox()
        self.cmb_direction.addItems(['both', 'virtual', 'real'])
        self.cmb_direction.setToolTip(
            'virtual — a colourful crop labelled __empty__/__inactive__\n'
            'real    — a blank cell labelled with an item name\n'
            'both    — one review pass over the two')
        self.btn_scan   = QPushButton('Scan mislabelled crops')
        self.btn_apply  = QPushButton('Apply decisions')
        self.btn_merge  = QPushButton('Merge (dry-run)')
        self.btn_audit  = QPushButton('Audit staging')
        self.btn_scan.clicked.connect(self.on_scan)
        self.btn_apply.clicked.connect(self.on_apply)
        self.btn_merge.clicked.connect(
            lambda: self._run(['democratic_merge_crops.py']))
        self.btn_audit.clicked.connect(
            lambda: self._run(['admin_audit_staging.py']))
        bar.addWidget(QLabel('direction:'))
        bar.addWidget(self.cmb_direction)
        for b in (self.btn_scan, self.btn_apply, self.btn_merge, self.btn_audit):
            bar.addWidget(b)
        bar.addStretch(1)
        self.status = QLabel('Ready. Run a scan to load flagged crops.')
        bar.addWidget(self.status)
        root.addLayout(bar)

        # ── Review pane ─────────────────────────────────────────────────
        # The table alone is an overview: a 64 px thumbnail is too small to
        # tell a dim icon from an empty cell, which is the whole judgement
        # being made here. The selected crop is shown large beside it, and
        # the decision keys advance to the next row, so the pass is one crop
        # at a time rather than a grid to scan.
        self.preview = QLabel('select a row')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(_PREVIEW_PX + 16, _PREVIEW_PX + 16)
        self.preview.setStyleSheet('background:#141414; border:1px solid #333;')
        self.preview_info = QLabel('')
        self.preview_info.setWordWrap(True)
        self.preview_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_hint = QLabel(
            'K keep · R reject · L relabel · ↑ ↓ move')
        self.preview_hint.setStyleSheet('color:#888;')

        side = QVBoxLayout()
        side.addWidget(self.preview)
        side.addWidget(self.preview_info)
        side.addWidget(self.preview_hint)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)

        # ── Table + log split ───────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ['', 'sha', 'label', 'slot', 'flagged as', 'bright/rich',
             'decision', 'relabel name'])
        self.table.verticalHeader().setDefaultSectionSize(_THUMB_PX + 4)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_RELABEL, QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(
            lambda row, *_: self._show_preview(row))

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.table)
        top.addWidget(side_w)
        top.setSizes([720, 280])
        split.addWidget(top)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText('Tool output appears here…')
        split.addWidget(self.log)
        split.setSizes([460, 240])
        root.addWidget(split, 1)
        self._install_shortcuts()

    # ── Process plumbing ────────────────────────────────────────────────

    def _busy(self, on: bool) -> None:
        for b in (self.btn_scan, self.btn_apply, self.btn_merge, self.btn_audit):
            b.setEnabled(not on)

    def _run(self, argv: list[str], on_done=None) -> None:
        if self._proc is not None:
            QMessageBox.information(self, 'Busy', 'A tool is already running.')
            return
        self._busy(True)
        self.log.appendPlainText(f'\n$ {sys.executable} {" ".join(argv)}\n')
        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_DIR))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(
                bytes(proc.readAllStandardOutput()).decode('utf-8', 'replace').rstrip()))

        def _finished(code, _status):
            self._proc = None
            self._busy(False)
            self.log.appendPlainText(f'[exit {code}]')
            if on_done:
                on_done(code)

        proc.finished.connect(_finished)
        self._proc = proc
        proc.start(sys.executable, argv)

    # ── One-at-a-time review ────────────────────────────────────────────

    def _show_preview(self, row: int) -> None:
        """Render the selected crop large, at its own pixels.

        Nearest-neighbour on purpose: smoothing a 33x43 cell invents detail
        that is not in the crop, and the decision is exactly whether there is
        detail in it.
        """
        if row < 0 or row >= len(getattr(self, '_rows', [])):
            return
        r = self._rows[row]
        img = self._images.get(row)
        if img is None:
            self.preview.setText('crop unavailable')
        else:
            h, w = img.shape[:2]
            scale = max(1, int(_PREVIEW_PX / max(h, w)))
            big = cv2.resize(img, (w * scale, h * scale),
                             interpolation=cv2.INTER_NEAREST)
            self.preview.setPixmap(_bgr_to_pixmap(big))
        self.preview_info.setText(
            f'<b>{r["label"]}</b><br>slot: {r["slot"]}<br>'
            f'flagged as: {r.get("why", "—")}<br>'
            f'{row + 1} of {len(self._rows)}')

    def _decide(self, verb: str) -> None:
        """Set the decision on the selected row and move to the next."""
        row = self.table.currentRow()
        if row < 0:
            return
        combo = self.table.cellWidget(row, COL_DECISION)
        if combo is not None:
            combo.setCurrentText(verb)
        if row + 1 < self.table.rowCount():
            self.table.setCurrentCell(row + 1, COL_SHA)

    def _focus_relabel(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        combo = self.table.cellWidget(row, COL_DECISION)
        if combo is not None:
            combo.setCurrentText('RELABEL')
        picker = self.table.cellWidget(row, COL_RELABEL)
        if picker is not None:
            picker.setFocus()

    def _install_shortcuts(self) -> None:
        for key, fn in (('K', lambda: self._decide('KEEP')),
                        ('R', lambda: self._decide('REJECT')),
                        ('L', self._focus_relabel)):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)

    # ── Actions ─────────────────────────────────────────────────────────

    def on_scan(self) -> None:
        self.status.setText('Scanning… (fetching crop pixels, ~1–3 min)')
        # Scan is the default (dry-run) mode — there is no --scan flag.
        self._run(['admin_reject_crops.py',
                   '--direction', self.cmb_direction.currentText(),
                   '--decisions', str(self._decisions_path),
                   '--montage', str(REPO_DIR / DEFAULT_MONTAGE)],
                  on_done=self._load_after_scan)

    def _load_after_scan(self, code: int) -> None:
        if code != 0:
            self.status.setText('Scan failed — see log.')
            return
        rows = _parse_tsv(self._decisions_path)
        self._populate(rows)
        self.status.setText(f'{len(rows)} crop(s) flagged. Set decisions, then Apply.')

    def _populate(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        self._rows = rows
        self._images = {}
        for i, r in enumerate(rows):
            thumb = QTableWidgetItem()
            img = _fetch_crop(r['sha'], HF_TOKEN, local_dir=self._local_crops)
            self._images[i] = img
            if img is not None:
                thumb.setData(Qt.ItemDataRole.DecorationRole, _bgr_to_pixmap(img))
            thumb.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, COL_THUMB, thumb)

            self.table.setItem(i, COL_SHA, QTableWidgetItem(r['sha'][:12]))
            self.table.setItem(i, COL_LABEL, QTableWidgetItem(r['label']))
            self.table.setItem(i, COL_SLOT, QTableWidgetItem(r['slot']))
            # Which contradiction flagged this crop. Worth seeing while
            # deciding: a colourful crop under `__empty__` usually wants
            # RELABEL to the item, a blank cell under an item's name usually
            # wants RELABEL to `__empty__` / `__inactive__`.
            self.table.setItem(i, COL_WHY, QTableWidgetItem(r.get('why', '')))
            self.table.setItem(
                i, COL_STATS,
                QTableWidgetItem(f'{float(r["bright"]):.0%} / {float(r["rich"]):.0%}'))

            combo = QComboBox()
            combo.addItems(DECISIONS)
            combo.setCurrentText(r['decision'] if r['decision'] in DECISIONS else 'REJECT')
            self.table.setCellWidget(i, COL_DECISION, combo)

            self.table.setCellWidget(i, COL_RELABEL, self._make_relabel_picker(r['relabel']))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            COL_RELABEL, QHeaderView.ResizeMode.Stretch)
        if rows:
            self.table.setCurrentCell(0, COL_SHA)
            self._show_preview(0)

    def _make_relabel_picker(self, current: str) -> QComboBox:
        """Editable combo restricted to the cargo list, with type-to-filter
        (contains-match). No free-text targets: the completer + apply-time
        validation keep RELABEL to real, current item names."""
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.addItem('')  # empty = no target chosen yet
        combo.addItems(self._cargo)
        comp = QCompleter(self._cargo, combo)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        combo.setCompleter(comp)
        combo.setCurrentText(current if current in self._cargo else '')
        combo.lineEdit().setPlaceholderText('type to search cargo (RELABEL only)')
        return combo

    def on_apply(self) -> None:
        if self.table.rowCount() == 0:
            QMessageBox.information(self, 'Nothing to apply', 'Run a scan first.')
            return
        # Rewrite the TSV from the freshly parsed file (full sha + stats) but
        # with the table's decisions overlaid — the table shows short shas, so
        # we merge on row order rather than trusting the truncated cell text.
        base = _parse_tsv(self._decisions_path)
        cargo = set(self._cargo)
        bad_names: list[str] = []
        n_relabel_missing = 0
        for i, r in enumerate(base):
            combo = self.table.cellWidget(i, COL_DECISION)
            picker = self.table.cellWidget(i, COL_RELABEL)
            r['decision'] = combo.currentText()
            r['relabel']  = picker.currentText().strip()
            if r['decision'] == 'RELABEL':
                if not r['relabel']:
                    n_relabel_missing += 1
                elif r['relabel'] not in cargo:
                    bad_names.append(f'{r["sha"][:10]}: {r["relabel"]!r}')
        if n_relabel_missing:
            QMessageBox.warning(
                self, 'Missing name',
                f'{n_relabel_missing} row(s) set to RELABEL have no name. '
                f'Pick one from the cargo list or change the decision.')
            return
        if bad_names:
            QMessageBox.warning(
                self, 'Name not in cargo',
                'These RELABEL names are not in the sto-warp cargo list '
                '(pick from the dropdown):\n\n' + '\n'.join(bad_names))
            return

        counts = {v: sum(1 for r in base if r['decision'] == v) for v in DECISIONS}
        ok = QMessageBox.question(
            self, 'Apply to HF dataset',
            f'Commit to {REPO}?\n\n'
            f'REJECT: {counts["REJECT"]}   KEEP: {counts["KEEP"]}   '
            f'RELABEL: {counts["RELABEL"]}\n\n'
            f'This deletes/relabels crops in data/ and cannot be trivially undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ok != QMessageBox.StandardButton.Yes:
            return

        _write_tsv(self._decisions_path, base)
        self.status.setText('Applying…')
        self._run(['admin_reject_crops.py', '--apply',
                   '--decisions', str(self._decisions_path)],
                  on_done=self._after_apply)

    def _after_apply(self, code: int) -> None:
        if code == 0:
            self.status.setText('Applied. Re-scan to confirm data/ is clean.')
            self.table.setRowCount(0)
        else:
            self.status.setText('Apply failed — see log.')


def main() -> int:
    if not HF_TOKEN:
        print('ERROR: HF_TOKEN not set (.env or shell).', file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv)
    win = AdminConsole()
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
