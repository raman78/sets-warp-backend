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
from PySide6.QtGui import QImage, QPixmap
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
COL_THUMB, COL_SHA, COL_LABEL, COL_SLOT, COL_STATS, COL_DECISION, COL_RELABEL = range(7)
_THUMB_PX = 64


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
        })
    return rows


def _write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ['# decision\tidx\tsha\tlabel\tslot\tbright\trich',
             '# decision ∈ {REJECT, KEEP, RELABEL <canonical name>}']
    for r in rows:
        verb = r['decision']
        first = f'{verb} {r["relabel"]}'.strip() if verb == 'RELABEL' else verb
        lines.append(f'{first}\t{r["idx"]}\t{r["sha"]}\t{r["label"]}\t'
                     f'{r["slot"]}\t{r["bright"]}\t{r["rich"]}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


class AdminConsole(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('WARP maintainer console')
        self.resize(1000, 720)
        self._proc: QProcess | None = None
        self._decisions_path = REPO_DIR / DEFAULT_DECISIONS
        # Cargo names for the RELABEL picker — from sto-warp's own loader.
        self._cargo = sorted(load_canonical_names())
        # Local crop mirror for thumbnails (avoids re-downloading from HF).
        self._local_crops = local_mirror_crops_dir()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── Action bar ──────────────────────────────────────────────────
        bar = QHBoxLayout()
        self.btn_scan   = QPushButton('Scan virtual crops')
        self.btn_apply  = QPushButton('Apply decisions')
        self.btn_merge  = QPushButton('Merge (dry-run)')
        self.btn_audit  = QPushButton('Audit staging')
        self.btn_scan.clicked.connect(self.on_scan)
        self.btn_apply.clicked.connect(self.on_apply)
        self.btn_merge.clicked.connect(
            lambda: self._run(['democratic_merge_crops.py']))
        self.btn_audit.clicked.connect(
            lambda: self._run(['admin_audit_staging.py']))
        for b in (self.btn_scan, self.btn_apply, self.btn_merge, self.btn_audit):
            bar.addWidget(b)
        bar.addStretch(1)
        self.status = QLabel('Ready. Run a scan to load flagged crops.')
        bar.addWidget(self.status)
        root.addLayout(bar)

        # ── Table + log split ───────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ['', 'sha', 'label', 'slot', 'bright/rich', 'decision', 'relabel name'])
        self.table.verticalHeader().setDefaultSectionSize(_THUMB_PX + 4)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_RELABEL, QHeaderView.ResizeMode.Stretch)
        split.addWidget(self.table)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText('Tool output appears here…')
        split.addWidget(self.log)
        split.setSizes([460, 240])
        root.addWidget(split, 1)

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

    # ── Actions ─────────────────────────────────────────────────────────

    def on_scan(self) -> None:
        self.status.setText('Scanning… (fetching crop pixels, ~1–3 min)')
        # Scan is the default (dry-run) mode — there is no --scan flag.
        self._run(['admin_reject_crops.py',
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
        for i, r in enumerate(rows):
            thumb = QTableWidgetItem()
            img = _fetch_crop(r['sha'], HF_TOKEN, local_dir=self._local_crops)
            if img is not None:
                thumb.setData(Qt.ItemDataRole.DecorationRole, _bgr_to_pixmap(img))
            thumb.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, COL_THUMB, thumb)

            self.table.setItem(i, COL_SHA, QTableWidgetItem(r['sha'][:12]))
            self.table.setItem(i, COL_LABEL, QTableWidgetItem(r['label']))
            self.table.setItem(i, COL_SLOT, QTableWidgetItem(r['slot']))
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
