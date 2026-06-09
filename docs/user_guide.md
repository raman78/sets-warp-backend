# User Guide — sets-warp-backend

This guide is for operators (server admins running the backend, not end
users of the sto-warp client). It covers the public client endpoints,
the admin commands for the four mergers + the audit, and the manual
recovery procedures.

For component-level detail see [`technical_overview.md`](technical_overview.md);
for the end-to-end data flow see [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

---

## 1. Client-facing endpoints

### Reading

| Endpoint | Use |
|---|---|
| `GET /health` | Liveness check, returns `{"ok": true}` |
| `GET /knowledge` | Merged pHash → item-name table |
| `GET /model/version` | Metadata for the latest trained model |
| `GET /config/labels` | Backend-side label map (screen types) |

### Writing (bulk — preferred)

These accept JSON batches and produce **one HF commit per batch**.
Used by sto-warp client v1.0.5+.

| Endpoint | Batch cap | Per-item cap |
|---|---|---|
| `POST /contribute/bulk-crops` | 50 crops | 150 000 bytes / crop |
| `POST /upload/screen-types` | 20 screens | 2 500 000 bytes / screen |
| `POST /upload/anchors` | 20 grids | — |

Per-call body fields are documented in the FastAPI auto-docs at
`/docs` (Swagger UI) when the service is running.

### Writing (legacy)

| Endpoint | Use |
|---|---|
| `POST /contribute` | Single pHash contribution. Kept for backward compatibility; new clients use the bulk endpoints. |

### Internal

| Endpoint | Use |
|---|---|
| `POST /webhooks/hf-dataset` | HF webhook → triggers `train_central_model.yml`. Not for client use. |
| `POST /admin/merge` | Retired (returns HTTP 410). Run `admin_merge.py` via the merger workflow instead. |

---

## 2. Admin commands

All four mergers run under `merge_staging.yml` every 2 hours. They can
also be invoked locally for debugging. Each one supports a dry-run by
default — pass `--apply` to commit.

### Promote crops → `data/crops/`

```sh
.venv/bin/python democratic_merge_crops.py                  # dry-run
.venv/bin/python democratic_merge_crops.py --apply          # commit
.venv/bin/python democratic_merge_crops.py --apply --min 1  # NEW threshold for UPDATEs too
.venv/bin/python democratic_merge_crops.py --since 2026-03-01
```

### Promote anchors → `data/anchors/`

```sh
.venv/bin/python democratic_merge_anchors.py                # dry-run
.venv/bin/python democratic_merge_anchors.py --apply
.venv/bin/python democratic_merge_anchors.py --apply --min 1
```

### Promote screen types + OCR text corrections → `data/`

```sh
.venv/bin/python democratic_merge_screens.py                # dry-run
.venv/bin/python democratic_merge_screens.py --apply
```

### Fold pHash contributions → `knowledge.json`

```sh
.venv/bin/python admin_merge.py                             # dry-run
.venv/bin/python admin_merge.py --apply
```

### Train icon + screen classifiers

```sh
.venv/bin/python admin_train.py                       # dry-run, no upload
.venv/bin/python admin_train.py --train --min 1       # full run + upload
.venv/bin/python admin_train.py --train --skip-if-unchanged
```

### Train ArcFace embedder

```sh
.venv/bin/python admin_train_metric.py --train
```

---

## 3. Health checks and audit

### Read-only orphan audit

`admin_audit_staging.py` runs on the 1st of each month at 04:00 UTC
(`audit_staging_health.yml`) and counts staging files whose key is
already in `data/` (i.e. files the next merger run will never look at).
Failure mails the repo owner.

To run locally:

```sh
.venv/bin/python admin_audit_staging.py                    # default thresholds
.venv/bin/python admin_audit_staging.py --crops-max 50     # tighter
.venv/bin/python admin_audit_staging.py --include-anchors  # add anchors
```

Exit code `0` means all domains are under their threshold; `1` means
at least one breach was recorded. The script prints both a
human-readable summary and machine-readable `AUDIT:` lines for grepping.

### When the audit fails

1. **Read the audit log.** Identify which domain breached and by how
   many orphans.
2. **Read the corresponding merger.** The breach means the merger
   either skipped the promoted entries' drain or refused to promote
   them at all. Both are bugs; fix the merger.
3. **Mop up the leaked orphans.** Only after the cause is patched,
   manually dispatch `drain_stale_staging.yml`. It runs
   `admin_drain_stale_staging.py`, which uses the same orphan-detection
   logic as the audit but actually deletes.

> Warning: `drain_stale_staging.yml` is `workflow_dispatch` only — no
> cron. Do not add a schedule. A scheduled drain would silently mask
> recurring merger bugs.

### Removing bad knowledge or labels

| Symptom | Tool |
|---|---|
| A wrong pHash override is poisoning client recognition | `admin_scrub_knowledge.py` |
| A wrong label leaked into `data/annotations.jsonl` | `admin_clean_labels.py` |

Both tools default to dry-run. Inspect the diff first, then re-run
with `--apply`.

---

## 4. Rate limits and abuse handling

Server-side daily caps:

| Limit | Default | Env override |
|---|---|---|
| Requests / IP / UTC day | 500 | `MAX_REQ_PER_IP` |
| Requests / install_id / UTC day | 500 | `MAX_REQ_PER_INSTALL` |

These are abuse gates, not polite ceilings — the client has its own
cap (`MAX_DAILY_UPLOADS = 1000` in `warp/trainer/sync.py`) below the
server cap so well-behaved clients never hit the 500 threshold.

If a single `install_id` is misbehaving:

1. Check the FastAPI logs on Render for that ID's request rate.
2. The democratic mergers naturally limit damage — one install =
   one vote per key, and `Z3` requires ≥ 2 votes to overwrite an
   existing entry. A single bad actor cannot rewrite community truth.
3. For persistent abuse, raise `MAX_REQ_PER_INSTALL` *down* (it's a
   global setting, not per-install). Per-install blocklists are not
   implemented today — the design relies on the staging quarantine
   plus democratic vote instead.

---

## 5. Backups and recovery

| Data | Where it lives | How to recover |
|---|---|---|
| `staging/<iid>/` | `sets-sto/sto-icon-dataset` | HF dataset commit history |
| `data/` | same repo | HF dataset commit history |
| `models/` | `sets-sto/warp-knowledge` | Previous training run's artefacts on HF |
| `knowledge.json` | `sets-sto/warp-knowledge` | HF dataset commit history |

HF commit history is the source of truth — there is no separate
backup. To roll back a bad model, upload the previous
`icon_classifier.pt` **with a fresh `trained_at` timestamp** in
`model_version.json`. The client's ModelUpdater only installs models
whose remote `trained_at` is strictly later than the local one —
silent backdated rollbacks are rejected as "no update available", so
the rollback must look like a normal forward-step. The next
ModelUpdater tick on every install (≤ 15 min) will then pull the
restored model.

---

## 6. Related documentation

- [Technical Overview](technical_overview.md)
- [Data Lifecycle](DATA_LIFECYCLE.md)
- [Main README](../README.md)
- [Agent Guidelines](../CLAUDE.md)
- [Changelog](../CHANGELOG.md)
