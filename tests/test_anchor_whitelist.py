"""The /upload/anchors whitelist gate.

Anchor grids are keyed by the client's *build type*, which is a different
vocabulary from the screen types WARP CORE learns them on: it folds SPACE_EQ
and SPACE_MIXED into SPACE, GROUND_EQ and GROUND_MIXED into GROUND, TRAITS
into SPACE_TRAITS. The gate used to validate one against the other, so every
SPACE and GROUND grid was refused with "not in whitelist" and re-offered on
every sync, for good — 112 of the 176 grids in one maintainer's store. BOFFS
and SPACE_TRAITS went through only because the two vocabularies spell those
the same.

Offline: no backend process, no HF, no network. These call the same helpers
the endpoint calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import main

# The client drops these before sending — OCR-only text slots
# (`warp/trainer/training_data.py:NON_ICON_SLOTS` in sto-warp).
NON_ICON_SLOTS = {'Ship Type', 'Ship Tier'}


@pytest.fixture
def whitelist() -> dict[str, set[str]]:
    """Build type -> allowed slot names, from the bundled labels.json."""
    return main._anchor_whitelist(main._load_labels_bundled())


def accepts(whitelist: dict[str, set[str]], build_type: str,
            slots: list[str]) -> bool:
    """The endpoint's D-G.10 gate, in isolation."""
    if whitelist and build_type not in whitelist:
        return False
    allowed = whitelist.get(build_type) or set()
    if build_type in whitelist and not allowed:
        return False
    if allowed:
        return not [k for k in slots
                    if k not in allowed and not main._ANCHOR_SEAT_RE.match(k)]
    return True


# ── The vocabulary itself ──────────────────────────────────────────────

def test_build_types_are_not_screen_types():
    """The bug in one assertion: the two lists are not interchangeable."""
    labels = main._load_labels_bundled()
    screen_types = set(labels['screen_types'])
    build_types = set(labels['anchor_build_types'])

    assert build_types - screen_types == {'SPACE', 'GROUND', 'SPEC'}


def test_every_build_type_inherits_slots(whitelist):
    """A build type with no slots would refuse every grid for it."""
    assert whitelist
    for build_type, slots in whitelist.items():
        assert slots, f'{build_type} inherited no slot names'


def test_a_missing_key_falls_back_to_the_bundled_map(monkeypatch):
    """HF serves labels.json until the dataset copy is reseeded, and that
    copy predates the key. Falling back keeps the gate on."""
    stale = {k: v for k, v in main._load_labels_bundled().items()
             if k != 'anchor_build_types'}

    assert main._anchor_whitelist(stale) == main._anchor_whitelist(
        main._load_labels_bundled())


def test_no_map_anywhere_fails_open(monkeypatch):
    """Fail-open matches every other ingestion gate: a transient outage must
    not black-hole production traffic."""
    monkeypatch.setattr(main, '_load_labels_bundled',
                        lambda: {'screen_types': [], 'slots': {}})

    assert main._anchor_whitelist({'slots': {}}) == {}


# ── Grids the client really sends ──────────────────────────────────────

@pytest.mark.parametrize('build_type, slots, why', [
    ('SPACE', ['Fore Weapons', 'Deflector', 'Engines', 'Tactical Consoles'],
     'the fold that used to be refused outright'),
    ('GROUND', ['Body Armor', 'Kit Modules', 'Weapons', 'Ground Devices'],
     'the other fold'),
    ('SPACE_TRAITS', ['Personal Space Traits', 'Starship Traits',
                      'Personal Ground Traits', 'Ground Reputation'],
     'a TRAITS screen mixes both environments; the build type covers it'),
    ('BOFFS', ['Boff Tactical', 'Boff Engineering', 'Boff Temporal'],
     'spelled the same in both vocabularies, so this passed before too'),
])
def test_accepted(whitelist, build_type, slots, why):
    assert accepts(whitelist, build_type, slots), why


@pytest.mark.parametrize('slots, why', [
    (['Fore Weapons', 'Boff Seat L[T]_304', 'Boff Seat R[E+O]_484'],
     'marker-keyed seats carry a pixel offset and cannot be enumerated'),
    (['Fore Weapons', 'Boff Seat L_483', 'Deflector'],
     'legacy seat key, no seat code'),
])
def test_marker_keyed_seats_are_accepted_by_shape(whitelist, slots, why):
    assert accepts(whitelist, 'SPACE', slots), why


@pytest.mark.parametrize('build_type, slots, why', [
    ('SPACE_EQ', ['Fore Weapons', 'Deflector', 'Engines'],
     'a screen type is not a build type — anchors never carry one'),
    ('SPACE', ['Fore Weapons', 'Deflector', 'Not A Slot At All'],
     'an unknown slot name is still refused'),
    ('SPACE', ['Fore Weapons', 'Deflector', 'Boff Seat X[T]_1'],
     'and so is something merely shaped like a seat key'),
])
def test_refused(whitelist, build_type, slots, why):
    assert not accepts(whitelist, build_type, slots), why


# ── Against a real client's store ──────────────────────────────────────

def _anchors_path() -> Path | None:
    env = os.environ.get('WARP_ANCHORS_JSON')
    if env:
        return Path(env).expanduser()
    default = Path.home() / '.local/share/warp/training_data/anchors.json'
    return default if default.exists() else None


@pytest.mark.skipif(_anchors_path() is None,
                    reason='no anchors.json; set WARP_ANCHORS_JSON to point at one')
def test_a_real_store_is_fully_accepted():
    """Samples can agree with the code and still miss what clients send."""
    whitelist = main._anchor_whitelist(main._load_labels_bundled())
    learned = json.loads(_anchors_path().read_text(encoding='utf-8'))['learned']

    refused = []
    for entry in learned:
        build_type = entry.get('type', '')
        slots = [k for k in entry.get('slots', {}) if k not in NON_ICON_SLOTS]
        if not build_type or len(slots) < 3:
            continue
        if not accepts(whitelist, build_type, slots):
            refused.append((build_type, sorted(set(slots) - whitelist.get(build_type, set()))[:3]))

    assert not refused, f'grids the client would send and we refuse: {refused[:5]}'
