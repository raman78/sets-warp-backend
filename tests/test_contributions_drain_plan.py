"""What the staging audit counts, and the drain deletes, in contributions/.

admin_merge treats a contribution as processed when its ID is in
`processed_contributions` or its date folder is older than `watermark_date`
— compaction moves old IDs out of the list. A planner that read only the list
could not see the second kind: 2,500 such files were on disk on 2026-10-01,
uncounted by the audit and untouched by the drain.

Offline: no HF, no network.
"""
from __future__ import annotations

import admin_drain_stale_staging as drain

FILES = [
    'contributions/2026-03-20/old.json', 'contributions/2026-03-20/old.png',
    'contributions/2026-09-01/listed.json', 'contributions/2026-09-01/listed.png',
    'contributions/2026-09-30/pending.json', 'contributions/2026-09-30/pending.png',
    'knowledge.json',
]


def test_a_file_older_than_the_watermark_is_planned():
    plan = drain._plan_contributions_drain(FILES, {'listed'}, '2026-04-04')

    assert 'contributions/2026-03-20/old.json' in plan
    assert 'contributions/2026-03-20/old.png' in plan


def test_a_listed_file_is_planned_and_a_pending_one_is_not():
    plan = drain._plan_contributions_drain(FILES, {'listed'}, '2026-04-04')

    assert 'contributions/2026-09-01/listed.json' in plan
    assert not any('pending' in p for p in plan)
    assert 'knowledge.json' not in plan


def test_without_a_watermark_only_the_list_counts():
    plan = drain._plan_contributions_drain(FILES, {'listed'})

    assert not any('old' in p for p in plan)
