"""Putting this machine's store beside the published dataset.

Every upload fault found so far was silent in the same way: the client
believed it had shared something the dataset does not have, and nothing put
the two side by side. Each was found by hand, months late, because a count
looked wrong.

Run standalone:
    python -m pytest tests/test_reconcile_local.py -v
"""
from __future__ import annotations

import json

import admin_reconcile_local as rec


# ── Why they differ, not whether ───────────────────────────────────────────
#
# The published dataset is the reference. A local store holding a different
# label is normally the tally working, not a fault — so the split is drawn on
# whether this install ever *sent* that label, which its own upload cache
# records. An earlier version of this tool scored both alike and concluded the
# dataset should be corrected to match one machine.

def test_a_matching_pair_is_not_reported():
    v = rec.compare({'aa': 'BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'BOFFS'})

    assert v == {'unsent': [], 'outvoted': [], 'absent': []}


def test_a_label_never_sent_is_a_transport_fault():
    """The screen-type bug: the correction stayed on the machine, so the
    dataset never had the chance to weigh it."""
    v = rec.compare({'aa': 'SPACE_BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'BOFFS'})

    assert v['unsent'] == [('aa', 'SPACE_BOFFS')]
    assert v['outvoted'] == []


def test_a_label_that_was_sent_and_lost_is_not_a_fault():
    """The tally weighed it and other contributors disagreed. Reported for
    review, never scored as an error."""
    v = rec.compare({'aa': 'SPACE_BOFFS'}, {'aa': 'BOFFS'}, {'aa': 'SPACE_BOFFS'})

    assert v['outvoted'] == [('aa', 'SPACE_BOFFS', 'BOFFS')]
    assert v['unsent'] == []


def test_something_never_sent_and_absent_there_is_unsent():
    v = rec.compare({'aa': 'DISCARD'}, {}, {})

    assert v['unsent'] == [('aa', 'DISCARD')]


def test_something_sent_and_then_dropped_is_not_a_transport_fault():
    """Sent, accepted, and later removed — a maintainer rejection reads this
    way and is legitimate."""
    v = rec.compare({'aa': 'DISCARD'}, {}, {'aa': 'DISCARD'})

    assert v['unsent'] == []
    assert v['outvoted'] == [('aa', 'DISCARD', '<dropped>')]


def test_something_in_the_dataset_and_not_here_is_absent():
    v = rec.compare({}, {'aa': 'BOFFS'}, {})

    assert v['absent'] == ['aa']


def test_a_legacy_cache_cannot_support_an_outvoted_claim(tmp_path):
    """The screen cache was a bare list of shas and recorded no label, so
    nothing in it proves which label was sent."""
    (tmp_path / '.sync_uploaded_screen_hashes.json').write_text(
        json.dumps(['aa']), encoding='utf-8')

    sent = rec.sent_labels(tmp_path, 'screens')

    assert sent == {'aa': ''}
    assert rec.compare({'aa': 'BOFFS'}, {'aa': 'TRAITS'}, sent)['unsent']


def test_a_missing_cache_leaves_everything_unproven(tmp_path):
    """No cache means no claim can be made about what was sent, so a
    difference falls back to the fault reading rather than being excused."""
    assert rec.sent_labels(tmp_path, 'crops') == {}


# ── Reading each side ──────────────────────────────────────────────────────

def test_the_unclassified_folder_is_not_compared(tmp_path):
    """`UNKNOWN` is a sentinel the backend refuses by design, so counting it
    would report a backlog nobody can clear."""
    d = tmp_path / 'screen_types' / rec.UNCLASSIFIED
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 50)

    assert rec.local_screens(tmp_path) == {}


def test_a_classified_screenshot_is_read_with_its_type(tmp_path):
    d = tmp_path / 'screen_types' / 'SPACE_BOFFS'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 50)

    assert list(rec.local_screens(tmp_path).values()) == ['SPACE_BOFFS']


def test_the_local_hash_is_truncated_like_the_client(tmp_path):
    """The client truncates to 32 hex chars and the dataset is keyed on that.
    Comparing full digests finds nothing at all — and an empty result reads
    as 'everything agrees', which is the worst way for this to fail."""
    d = tmp_path / 'screen_types' / 'BOFFS'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG')

    sha = next(iter(rec.local_screens(tmp_path)))

    # 32 literally, not `rec.SHA_LEN` — asserting against the constant the
    # code uses compares it with itself and passes whatever it is set to.
    assert len(sha) == 32


def test_published_screens_are_read_from_the_merged_metadata(tmp_path):
    p = tmp_path / 'data' / 'screen_types'
    p.mkdir(parents=True)
    (p / 'metadata.jsonl').write_text(
        json.dumps({'sha': 'aa', 'type': 'BOFFS'}) + '\n'
        + '{ broken\n'
        + json.dumps({'sha': 'bb', 'type': 'TRAITS'}) + '\n',
        encoding='utf-8')

    assert rec.published_screens(tmp_path) == {'aa': 'BOFFS', 'bb': 'TRAITS'}


def test_a_missing_dataset_file_is_not_an_error(tmp_path):
    assert rec.published_screens(tmp_path) == {}
    assert rec.published_crops(tmp_path) == {}


def test_a_crop_label_comes_from_the_annotations_not_the_filename(tmp_path):
    """The filename carries the label the file had when it was written, so a
    correction made later would be invisible exactly where it matters."""
    (tmp_path / 'crops').mkdir()
    png = tmp_path / 'crops' / 'boff_tactical__old_name__abc123def456.png'
    png.write_bytes(b'\x89PNG' + b'\x00' * 50)
    (tmp_path / 'annotations.json').write_text(json.dumps({
        'key': {'filename': 's.png', 'annotations': [
            {'ann_id': 'abc123def456', 'slot': 'Boff Tactical',
             'name': 'Corrected Name'}]}}), encoding='utf-8')

    assert list(rec.local_crops(tmp_path).values()) == \
        ['Boff Tactical|Corrected Name']
