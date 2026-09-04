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

import pytest

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


# ── A ship has two names, and the game prints the other one ────────────────
#
# `cargo.ships()` is keyed on `Page`, the wiki article title. A `Ship Type`
# crop is OCR of what the game prints, which is the row's `name`, and the two
# differ for 84 of the 797 ships. Checking against article titles alone flags
# every one of those as unresolvable — the same wrong-vocabulary mistake as
# checking a ship name against the item cargo, one level further in.

SHIP_ROWS = {
    # Page (the wiki article title)          row as cargo returns it
    'Galaxy Exploration Cruiser Retrofit': {'name': 'Exploration Cruiser Retrofit'},
    'Fleet Yamaguchi Support Cruiser':     {'name': 'Fleet Yamaguchi Support Cruiser'},
    'Some Ship With No Display Name':      {},
}


@pytest.fixture
def ship_vocab(monkeypatch):
    """The real `load_vocabularies`, with cargo's ship rows stubbed.

    Driving the actual function matters here: building the same set inside
    the test would pass whether or not the source still does it, which is
    worse than no test.

    `warp` is not installed in this venv — the tool reaches the sibling
    checkout through `sys.path` at call time — so the stub is injected as the
    module rather than patched onto an import. That also keeps the test from
    depending on whether some earlier test happened to put sto-warp on the
    path first, which it silently did before.
    """
    import sys, types

    cargo = types.ModuleType('warp.data.cargo')
    cargo.ships = lambda: dict(SHIP_ROWS)
    pkg_warp = types.ModuleType('warp')
    pkg_data = types.ModuleType('warp.data')
    pkg_data.cargo = cargo
    pkg_warp.data = pkg_data
    for name, mod in (('warp', pkg_warp), ('warp.data', pkg_data),
                      ('warp.data.cargo', cargo)):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(tool, 'load_canonical_names', lambda: {'Some Item'})
    return tool.load_vocabularies()


def test_the_in_game_name_is_accepted(ship_vocab):
    """The label OCR actually produces — `ships()` is keyed on the article
    title, and the two differ for 84 of the 797 ships."""
    assert not tool._name_resolves_nowhere(
        'Exploration Cruiser Retrofit', 'Ship Type', ship_vocab)


def test_the_wiki_article_title_is_still_accepted(ship_vocab):
    """Older labels were written under the article title; both must resolve."""
    assert not tool._name_resolves_nowhere(
        'Galaxy Exploration Cruiser Retrofit', 'Ship Type', ship_vocab)


def test_a_ship_that_is_neither_is_still_flagged(ship_vocab):
    """Widening to two names must not widen to anything."""
    assert tool._name_resolves_nowhere(
        'Exploration Cruiser Refit', 'Ship Type', ship_vocab)


def test_a_row_with_no_display_name_still_contributes_its_title(ship_vocab):
    assert not tool._name_resolves_nowhere(
        'Some Ship With No Display Name', 'Ship Type', ship_vocab)


# ── The virtual classes are labels, not unresolvable names ─────────────────
#
# `__empty__` / `__inactive__` are gallery classes the embedder needs, and no
# cargo table carries them. An item-name check refuses them, which would have
# made the commonest correction of all — a blank cell filed under an item's
# name — impossible to apply: the RELABEL guard aborts the whole commit on a
# target it cannot resolve.

def test_a_virtual_label_is_an_acceptable_target():
    for v in ('__empty__', '__inactive__'):
        assert not _nowhere(v, 'Boff Science')
        assert not _nowhere(v, 'Tactical Consoles')


def test_a_virtual_label_is_acceptable_in_a_text_slot_too():
    assert not _nowhere('__empty__', 'Ship Type')


def test_an_invented_dunder_name_is_still_refused():
    """Accepting the two known classes must not accept anything shaped alike."""
    assert _nowhere('__nonsense__', 'Boff Science')
