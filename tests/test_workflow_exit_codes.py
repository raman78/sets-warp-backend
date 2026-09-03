"""A scheduled job must not report success after a step crashed.

`democratic_merge_crops.py` failed on every run from 2026-07-16, and the
workflow was green each time. The step is

    python democratic_merge_crops.py --apply --min "$MIN_VOTES" | tee ...

and in a pipeline bash returns the status of the *last* command — `tee`,
which succeeds. GitHub's default shell is `bash -e {0}`: errexit, no
pipefail. Seven weeks of promotions were lost behind a green tick.

This locks the fix across every workflow, not just the one that failed:
all seven steps in this repo pipe into `tee`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

WORKFLOWS = Path(__file__).resolve().parent.parent / '.github' / 'workflows'


def _steps():
    for f in sorted(WORKFLOWS.glob('*.yml')):
        doc = yaml.safe_load(f.read_text(encoding='utf-8'))
        for job_name, job in (doc.get('jobs') or {}).items():
            for step in (job.get('steps') or []):
                if step.get('run'):
                    yield f.name, job_name, step.get('name', '<unnamed>'), step['run']


def test_there_are_workflows_to_check():
    """A glob that matches nothing would make every test below vacuous."""
    assert list(_steps())


def test_no_step_pipes_a_command_without_pipefail():
    offenders = [
        f'{wf}:{step}'
        for wf, _job, step, run in _steps()
        if '|' in run and 'pipefail' not in run
        and any(line.strip().count('|') and not line.strip().startswith('#')
                for line in run.splitlines())
    ]
    assert not offenders, (
        'these steps would report success even if the command crashed: '
        + ', '.join(offenders))


def test_the_crop_merge_step_is_guarded():
    """The exact step that hid the failure for seven weeks."""
    runs = [run for wf, _j, _s, run in _steps()
            if wf == 'merge_staging.yml' and 'democratic_merge_crops' in run]

    assert runs, 'the crop merge step disappeared from merge_staging.yml'
    assert all('set -o pipefail' in r for r in runs)


def test_pipefail_is_set_before_the_command_it_guards():
    """`set -o pipefail` after the pipeline would guard nothing."""
    for wf, _job, step, run in _steps():
        if 'pipefail' not in run:
            continue
        lines = [l.strip() for l in run.splitlines() if l.strip()]
        first_pipe = next((i for i, l in enumerate(lines)
                           if '|' in l and not l.startswith('#')), None)
        guard = next(i for i, l in enumerate(lines) if 'pipefail' in l)
        assert first_pipe is None or guard < first_pipe, f'{wf}:{step}'
