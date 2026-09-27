"""A training run must stop in time to publish what it trained.

The icon classifier checked its deadline only before starting an epoch. An
epoch took 12-15 min on the CPU runner, so one that started just under the
deadline ran the job past its 60-min cap during the upload: on 2026-09-27
two runs trained both models and were killed before publishing either. The
next run then saw the same new crops and played the same lottery.

Run standalone:
    python -m pytest tests/test_training_budget.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import admin_train as t

WORKFLOW = (Path(__file__).resolve().parent.parent
            / '.github' / 'workflows' / 'train_central_model.yml')


def test_an_epoch_that_would_cross_the_deadline_is_not_started():
    """The incident: 5 min left, epochs take 13 — the old check said go."""
    assert not t._next_epoch_fits(deadline=1000.0, last_epoch_s=13 * 60,
                                  now=1000.0 - 5 * 60)


def test_an_epoch_that_fits_is_started():
    assert t._next_epoch_fits(deadline=1000.0, last_epoch_s=60.0, now=500.0)


def test_the_first_epoch_starts_while_the_budget_lasts():
    """No duration measured yet — only the deadline itself can stop it."""
    assert t._next_epoch_fits(deadline=1000.0, last_epoch_s=None, now=999.0)
    assert not t._next_epoch_fits(deadline=1000.0, last_epoch_s=None, now=1001.0)


def test_no_deadline_never_stops():
    assert t._next_epoch_fits(deadline=None, last_epoch_s=10_000.0, now=0.0)


def test_the_stop_message_says_why():
    """A budget stop must read as one, with the numbers that caused it."""
    import time
    msg = t._budget_stop_message('icon classifier', 4,
                                 deadline=time.monotonic() + 300,
                                 last_epoch_s=13 * 60 + 12)
    assert 'before epoch 4' in msg
    assert '13m12s' in msg


def _workflow_timeout_s() -> int:
    m = re.search(r'^\s*timeout-minutes:\s*(\d+)', WORKFLOW.read_text(), re.M)
    assert m, 'train_central_model.yml has no timeout-minutes'
    return int(m.group(1)) * 60


def test_the_budgets_fit_inside_the_job_timeout():
    """Budget + one overrunning epoch of each model + upload < job cap.

    The overrun allowance is the largest epoch seen (15 min) for the icon
    classifier; the guard stops before it, so this is the safety margin
    for an epoch slower than the one before it.
    """
    worst = (t.ICON_TRAIN_BUDGET_S + 15 * 60     # icon + one slow epoch
             + t.SC_TRAIN_BUDGET_S + 2 * 60      # screen + one epoch
             + 10 * 60)                          # setup, clone, upload
    assert worst <= _workflow_timeout_s()


def test_the_job_timeout_is_within_githubs_cap():
    """GitHub-hosted runners kill any job at 6 h regardless of the YAML."""
    assert _workflow_timeout_s() <= 360 * 60


def test_the_trainer_runs_unbuffered():
    """Buffered stdout put the traceback above the line that failed."""
    assert 'python -u admin_train.py' in WORKFLOW.read_text()
