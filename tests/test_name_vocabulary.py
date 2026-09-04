"""`name-not-in-cargo` must check the vocabulary the slot actually allows.

The flag exists to surface labels nothing downstream can resolve. Checking
every label against the *item* cargo flags all of them — a `Ship Type` band
holds a ship name and a `Ship Tier` band holds `T6`, neither of which is an
item. Measured 2026-09-04 on the 12274-entry production mirror: 151 entries
flagged, 137 of them those two slots. 91% noise, burying 14 real hits, which
is how a flag stops ranking anything.

Run standalone:
    python -m pytest tests/test_name_vocabulary.py -v
"""
from __future__ import annotations

import admin_reject_crops as tool


VOCAB = {
    'items': {'Console - Universal - Assimilated Module', 'Fire on my Mark'},
    'ships': {'Legendary Bortasqu\' Command Battlecruiser'},
    'tiers': {'T5', 'T6', 'T6-X'},
}


def _nowhere(name: str, slot: str, vocab=VOCAB) -> bool:
    return tool._name_resolves_nowhere(name, slot, vocab)


# ── The noise the fix removes ──────────────────────────────────────────────

def test_a_ship_name_in_a_ship_type_band_is_not_flagged():
    assert not _nowhere("Legendary Bortasqu' Command Battlecruiser", 'Ship Type')


def test_a_tier_token_in_a_ship_tier_band_is_not_flagged():
    assert not _nowhere('T6-X', 'Ship Tier')


def test_the_lowercase_slot_spelling_is_recognised_too():
    """The client writes `ship_type_*`; the merged record writes `Ship Type`."""
    assert not _nowhere("Legendary Bortasqu' Command Battlecruiser", 'ship_type_0')
    assert not _nowhere('T6', 'ship_tier_0')


# ── The signal it must keep ────────────────────────────────────────────────

def test_a_misread_ship_name_is_still_flagged():
    """One character off is exactly what this hunts."""
    assert _nowhere("Legondary Bortasqu' Command Battlecruiser", 'Ship Type')


def test_an_item_name_nothing_resolves_is_still_flagged():
    assert _nowhere('Console - Universal - Sequential Warhead Loader',
                    'Tactical Consoles')


def test_a_junk_fragment_in_an_icon_slot_is_still_flagged():
    assert _nowhere('mart', 'Engineering Consoles')


def test_a_qualified_variant_of_a_real_item_is_still_flagged():
    """`Fire on my Mark (Ground)` must not pass just because the base does —
    the exporter cannot write the qualified form."""
    assert _nowhere('Fire on my Mark (Ground)', 'Boff Tactical')


# ── Cross-checks ───────────────────────────────────────────────────────────

def test_a_ship_name_is_not_accepted_in_an_icon_slot():
    """Widening the pool for text bands must not widen it everywhere."""
    assert _nowhere("Legendary Bortasqu' Command Battlecruiser",
                    'Tactical Consoles')


def test_a_real_item_sitting_in_a_text_slot_is_not_reported_as_unresolvable():
    """The slot comes from the detector, so a mislabelled row is possible.
    That is a slot problem, not an unknown name — this flag must not claim it."""
    assert not _nowhere('Fire on my Mark', 'Ship Type')


def test_nothing_is_flagged_when_the_vocabulary_could_not_be_loaded():
    """An empty pool means 'cannot check'. Flagging then marks the whole
    dataset and the review becomes useless."""
    empty = {'items': set(), 'ships': set(), 'tiers': set()}

    assert not tool._name_resolves_nowhere('anything at all', 'Devices', empty)
    assert not tool._name_resolves_nowhere('anything at all', 'Ship Type', empty)
