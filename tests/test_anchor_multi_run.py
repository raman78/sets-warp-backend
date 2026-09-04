"""A multi-run anchor grid must merge, and a flat one must stay flat.

BOFF panels split a seat's abilities across several columns, so the client
emits the slot as a list of `runs` — the row's y/w/h at the slot level and an
x origin, step and count per run. `_aggregate_group` required `x0_rel` at the
slot level, so every slot of such a grid failed the mandatory-coord check, the
group aggregated to nothing, and the grids sat in staging.

Measured 2026-09-04 on the published mirror: 2 grids stuck in staging, and 0
of the 1255 slot entries in `data/anchors` multi-run — no such grid had ever
merged.

The flat direction matters just as much. The client sizes a flat row from the
ship's profile and takes a `runs` count as authoritative, so emitting `runs`
for what has always been flat would silently stop honouring the profile.

Run standalone:
    python -m pytest tests/test_anchor_multi_run.py -v
"""
from __future__ import annotations

import democratic_merge_anchors as m


def _flat(x0=0.36, y=0.46, w=0.067, h=0.076, step=0.234, count=3) -> dict:
    return {'x0_rel': x0, 'y_rel': y, 'w_rel': w, 'h_rel': h,
            'step_rel': step, 'count': count}


def _multi(runs, y=0.46, w=0.067, h=0.076) -> dict:
    return {'y_rel': y, 'w_rel': w, 'h_rel': h, 'runs': runs}


def _run(x0, step=0.093, count=2) -> dict:
    return {'x0_rel': x0, 'step_rel': step, 'count': count}


def _group(*grids):
    return m._aggregate_group({'iid': list(grids)},
                              ['620x733'] * len(grids), ['iid'])


# ── The grids that were stuck ──────────────────────────────────────────────

def test_a_multi_run_slot_aggregates_instead_of_being_dropped():
    out = _group({'slots': {'Boff Tactical': _multi([_run(0.36), _run(0.60)])}})

    assert out is not None
    assert len(out['slots']['Boff Tactical']['runs']) == 2


def test_the_row_coords_are_kept_at_the_slot_level():
    """A multi-run slot has no slot-level x, and must not be required to."""
    out = _group({'slots': {'Boff Science': _multi([_run(0.36), _run(0.60)])}})
    geo = out['slots']['Boff Science']

    assert set(geo) == {'y_rel', 'w_rel', 'h_rel', 'runs'}


def test_runs_are_medianed_position_by_position():
    a = _multi([_run(0.30), _run(0.60)])
    b = _multi([_run(0.40), _run(0.70)])
    c = _multi([_run(0.50), _run(0.80)])

    runs = _group({'slots': {'S': a}}, {'slots': {'S': b}},
                  {'slots': {'S': c}})['slots']['S']['runs']

    assert [r['x0_rel'] for r in runs] == [0.4, 0.7]


def test_the_number_of_runs_is_itself_a_median():
    """One contributor splitting a row differently cannot add a column."""
    two   = _multi([_run(0.30), _run(0.60)])
    three = _multi([_run(0.30), _run(0.60), _run(0.90)])

    runs = _group({'slots': {'S': two}}, {'slots': {'S': two}},
                  {'slots': {'S': three}})['slots']['S']['runs']

    assert len(runs) == 2


# ── The flat direction must not change ─────────────────────────────────────

def test_a_flat_slot_stays_flat():
    """Emitting `runs` here would stop the client honouring the ship profile."""
    geo = _group({'slots': {'Fore Weapons': _flat()}})['slots']['Fore Weapons']

    assert 'runs' not in geo
    assert geo['x0_rel'] == 0.36


def test_a_single_run_slot_is_written_flat_too():
    """One run is the flat shape by another spelling; write the shape whose
    client behaviour matches what a one-row slot has always had."""
    geo = _group({'slots': {'S': _multi([_run(0.36)])}})['slots']['S']

    assert 'runs' not in geo
    assert geo['x0_rel'] == 0.36


def test_a_flat_and_a_multi_run_contributor_still_aggregate():
    """Mixed clients: the flat one counts as a single run."""
    out = _group({'slots': {'S': _flat(x0=0.30)}},
                 {'slots': {'S': _multi([_run(0.40), _run(0.70)])}})

    assert out is not None
    assert out['slots']['S']['runs'][0]['x0_rel'] == 0.35


# ── Still refused ──────────────────────────────────────────────────────────

def test_a_slot_with_no_x_origin_anywhere_is_dropped():
    """Nothing can place an icon from a row alone."""
    out = _group({'slots': {'S': {'y_rel': 0.4, 'w_rel': 0.06, 'h_rel': 0.07}}})

    assert out is None


def test_a_run_without_an_x_origin_is_dropped_not_guessed():
    out = _group({'slots': {'S': _multi([_run(0.36), {'step_rel': 0.09}])}})

    assert len(out['slots']['S']['runs']) == 1


def test_a_slot_missing_a_row_coord_is_dropped():
    out = _group({'slots': {'S': {'x0_rel': 0.3, 'y_rel': 0.4, 'w_rel': 0.06}}})

    assert out is None
