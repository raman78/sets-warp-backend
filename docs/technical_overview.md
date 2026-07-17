# Technical Overview — sets-warp-backend

The backend is the only component in the WARP ecosystem that holds a
write-scoped HuggingFace token. All client uploads flow through it, all
training jobs run from it, and all promotions to the public `data/`
folder originate from one of its mergers. This document is the
technical reference for that machinery.

For the end-to-end data-flow diagram (user → backend → staging →
mergers → data/ → training → models → user) see
[`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md). This file covers the
*components* and their contracts; the lifecycle doc covers how they
chain together.

---

## 1. Components

```
┌───────────────────────────────────────────────────────────────────┐
│  FastAPI service (main.py)            — HF Space (Docker)        │
│    • receives client uploads (bulk + legacy single-shot)         │
│    • serves /knowledge, /model/version, /config/labels           │
│    • holds the HF write token (env var, never on clients)        │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  HF Dataset: sets-sto/sto-icon-dataset
                          (staging + data + raw contributions)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  Democratic mergers                — GitHub Actions, 2-hourly    │
│    • democratic_merge_crops.py     → data/crops + annotations    │
│    • democratic_merge_anchors.py   → data/anchors                │
│    • democratic_merge_screens.py   → data/screen_types + text    │
│    • admin_merge.py                → knowledge.json (pHash map)  │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  Trainers                          — GitHub Actions               │
│    • admin_train.py        (hourly)  EfficientNet + MobileNetV3   │
│    • admin_train_metric.py (daily)   ArcFace embedder             │
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  HF Dataset: sets-sto/warp-knowledge
                          (models, label maps, knowledge.json)

┌───────────────────────────────────────────────────────────────────┐
│  Audit safety net                  — GitHub Actions, monthly      │
│    • admin_audit_staging.py        read-only orphan count         │
│    • admin_drain_stale_staging.py  manual dispatch only           │
└───────────────────────────────────────────────────────────────────┘
```

| Component | Role | File |
|---|---|---|
| FastAPI service | Receives client uploads, serves models + knowledge | `main.py` |
| Crops merger | Promotes icon crops staging → `data/crops/` | `democratic_merge_crops.py` |
| Anchors merger | Promotes anchor grids staging → `data/anchors/` | `democratic_merge_anchors.py` |
| Screens merger | Promotes screen-type PNGs + OCR text corrections | `democratic_merge_screens.py` |
| Knowledge merger | Folds pHash contributions into `knowledge.json` | `admin_merge.py` |
| Icon + screen trainer | EfficientNet-B0 + MobileNetV3-Small | `admin_train.py` |
| Embedder trainer | ArcFace metric model + gallery | `admin_train_metric.py` |
| Staging audit | Read-only orphan check, monthly cron | `admin_audit_staging.py` |
| One-shot drain | Manual cleanup when audit breaches | `admin_drain_stale_staging.py` |
| Knowledge scrubber | Removes confirmed-bad pHash entries | `admin_scrub_knowledge.py` |
| Label scrubber | Removes confirmed-bad crop labels | `admin_clean_labels.py` |
| Virtual-crop review | Reject/relabel colourful `__empty__` crops + GUI | `admin_reject_crops.py`, `admin_console.py` |
| Virtual-poison audit | Read-only unreviewed-poison count, monthly cron | `admin_audit_virtual_poison.py` |

---

## 2. FastAPI service (`main.py`)

Deployed as a HuggingFace Space (Docker, `sets-sto/warp-backend`,
`sets-sto-warp-backend.hf.space`), auto-updated by `deploy_space.py` /
the Deploy Space workflow on every push to `main`. Holds the HF write
token as a server-side Space secret; no client can reach HF directly
for writes.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check, returns `{"ok": true}` |
| `GET` | `/model/version` | Latest centrally-trained model metadata |
| `GET` | `/config/labels` | Backend-side label map (e.g. screen types) |
| `GET` | `/knowledge` | Merged pHash → item-name lookup table |
| `POST` | `/contribute` | Legacy single-shot pHash contribution |
| `POST` | `/contribute/bulk-crops` | Up to 50 confirmed icon crops per call |
| `POST` | `/upload/screen-types` | Up to 20 screen-type screenshots per call |
| `POST` | `/upload/anchors` | Up to 20 anchor grids per call |
| `POST` | `/webhooks/hf-dataset` | HF webhook → triggers GH training workflow |
| `POST` | `/admin/merge` | Retired (returns HTTP 410) — use the GH merger workflow |

### Bulk endpoints (Phase 1, added in `[Unreleased]`)

The three bulk endpoints exist so clients can drop their write-scoped HF
token (Phase 2). Each one accepts a JSON batch, validates every item,
and produces **a single HF commit** per batch — N crops cost
`ceil(N / batch_size)` commits, not N. Per-item caps live in `main.py`:

| Constant | Value | Purpose |
|---|---|---|
| `MAX_BULK_CROPS` | 50 | Per `/contribute/bulk-crops` call |
| `MAX_BULK_SCREEN_TYPES` | 20 | Per `/upload/screen-types` call |
| `MAX_BULK_ANCHOR_GRIDS` | 20 | Per `/upload/anchors` call |
| `MAX_CROP_PNG_BYTES` | 150 000 | Per icon crop |
| `MAX_SCREEN_PNG_BYTES` | 2 500 000 | Per full screenshot |
| `MIN_CROP_PX` | 16 | Reject pathological tiny crops |

### Rate limits

Per UTC day, enforced server-side:

| Limit | Default | Env override |
|---|---|---|
| Requests / IP | 500 | `MAX_REQ_PER_IP` |
| Requests / install_id | 500 | `MAX_REQ_PER_INSTALL` |

These are independent of the client-side `MAX_DAILY_UPLOADS = 1000`
counter in `warp/trainer/sync.py` (the client cap is a polite ceiling;
the server cap is the abuse gate).

### Validation

`install_id` matches `^[a-zA-Z0-9_-]{8,64}$`; `screen_type` matches
`^[a-zA-Z0-9_-]{1,40}$`; anchor coords must be relative (0.0–1.0);
aspect ratio bounded to plausible monitor ratios (~0.5 – 3.56). Item
names truncated at `MAX_NAME_LEN = 120`. Anything that fails validation
is rejected with HTTP 4xx — never silently written to HF.

### Webhook trigger

`POST /webhooks/hf-dataset` is invoked by HuggingFace when the dataset
changes. It calls the GitHub Actions REST API (`GH_TOKEN`, `GH_REPO`)
to dispatch `train_central_model.yml`. Used as a fast-path on top of
the hourly cron.

---

## 3. The four democratic mergers

All four follow the same pattern: read every `staging/<install_id>/`
prefix, tally votes per semantic key, promote winners to `data/`, and
**drain** the promoted entries from staging in the same HF commit.
The mergers differ only in their semantic key.

| Merger | Semantic key | Target |
|---|---|---|
| `democratic_merge_crops.py` | `crop_sha256` | `data/crops/<sha>.png` + `data/annotations.jsonl` |
| `democratic_merge_anchors.py` | `(build_type, aspect_bucket)` | `data/anchors/<bt>_<bucket>.json` |
| `democratic_merge_screens.py` | screen-type `<sha>` + text-correction `<key>` | `data/screen_types/<T>/<sha>.png` + `data/text_corrections.jsonl` |
| `admin_merge.py` | `phash` | `knowledge.json` (different repo: `sets-sto/warp-knowledge`) |

### Z3 — asymmetric thresholds

NEW keys require 1 vote. UPDATEs to an existing key require ≥ N votes
(default `--min 2`, configurable). Rationale:

- A first contributor adding a previously-unknown item is almost
  always right — gating that behind "wait for two users to see the
  same exotic console" stalls coverage.
- Overwriting an established entry is the dangerous case — that's
  where one malicious or careless contributor could flip a label. Two
  matching votes is the minimum quorum for a write that overrides
  existing community truth.

The threshold lives in each merger as a `--min` CLI flag and is
applied per key in `_merge()`:

```python
threshold = min_votes if key in existing else 1
accepted  = count >= threshold
```

### Drain on promote

Every merger deletes the staging entries it promoted **in the same
HF commit** as the promotion. The drain operation set is computed
inside `_apply()`:

```
ops = [
    CommitOperationAdd(data/crops/<sha>.png),       # promote
    CommitOperationAdd(data/annotations.jsonl),     # merged
    CommitOperationDelete(staging/<iid>/crops/<sha>.png),  # drain
    CommitOperationDelete(staging/<iid>/annotations.jsonl) # drain (if empty)
    …
]
HfApi().create_commit(operations=ops, …)
```

This keeps steady-state staging size bounded by `per_install_uploads ×
2 h` — the merger cadence. Lifetime contributions accumulate in
`data/`, not in staging.

### Poison filter

`_is_poison_name()` blocks names that should never reach `data/`:

- Names starting with `__` (virtual classes: `__empty__`,
  `__inactive__`, `__boff_*`).
- `Test Item Name` (debug placeholder).

This is **defence in depth** — the client also strips these names in
`warp/knowledge/sync_client.py` before upload (`_poison_filter_enabled`
flag, commit `91cd30d` in sto-warp), but the server-side filter is the
authoritative gate. If the client poison filter regresses, the merger
catches it; if the merger regresses, the audit (§5) flags it.

### Atomic commits

Every promotion + drain set goes out as a single
`HfApi.create_commit()` call. The dataset is never observed
half-applied: either the promotion + drain happens together, or the
HF commit fails and both stay pending for the next cycle.

---

## 4. Training pipelines

### Icon + screen classifier (`admin_train.py`)

| Aspect | Value |
|---|---|
| Cron | Hourly (`0 * * * *` in `train_central_model.yml`) |
| Runner | `ubuntu-latest`, CPU-only PyTorch wheels |
| Hard timeout | 60 min |
| In-script deadline | 50 min (`time.monotonic() + 50 * 60`) |
| Skip condition | `--skip-if-unchanged` (compares against `training_manifest.json`) |
| Min new crops | 10 (`MIN_NEW_CROPS`) |

Architecture: EfficientNet-B0 (icon classifier) + MobileNetV3-Small
(screen classifier). Both fine-tune from the previous baseline pulled
from `sets-sto/warp-knowledge/models/`; the classifier head is replaced
to match the new `n_classes`. Loss: focal. Schedule: cosine annealing
with early stopping.

The trainer also runs `collect_text_corrections()` which builds
`ship_type_corrections.json` from `Ship Type` / `Ship Tier` annotations
with non-empty `ml_name` — uploaded alongside the models.

Output files uploaded to `sets-sto/warp-knowledge/models/`:

```
icon_classifier.pt          + label_map.json + icon_classifier_meta.json
screen_classifier.pt        + screen_classifier_labels.json
model_version.json          (trained_at, n_classes, val_acc, …)
ship_type_corrections.json  (optional, only if any text corrections exist)
training_manifest.json      (set of crop SHAs in this run)
```

### ArcFace embedder (`admin_train_metric.py`)

Separate workflow `train_metric_model.yml`, daily at 00:45 UTC (off
the hourly classifier window). Trains the gallery model used as a
cross-check in the matcher priority chain. Outputs:
`icon_embedder.pt`, `embedder_label_map.json`,
`icon_embedder_meta.json`, `embedding_index.npz`.

Once the metric model is stable the cadence may move to hourly; for
now it stays daily so a bad batch can't churn the gallery faster than
it can be reviewed.

### Download strategy (post-mortem-driven)

`admin_train.py` downloads only the exact (install_id, sha) crops the
voting step selected, in parallel with `ThreadPoolExecutor(16)` over
`urllib.request`. `urllib` is used instead of `huggingface_hub`'s
`snapshot_download` because the latter uses `httpx` async I/O, which
ignores `socket.setdefaulttimeout` — a single stalled TCP read can
hang the entire training job indefinitely. With urllib + 120 s socket
timeout, a stalled file aborts after 2 min and is logged as a skip.

---

## 5. Audit safety net

### Monthly read-only audit (`admin_audit_staging.py`)

Runs on the 1st of each month at 04:00 UTC
(`audit_staging_health.yml`). Counts **orphans** — staging files whose
semantic key already appears in `data/`. An orphan should not exist:
either the merger drained it (then the count is 0) or the merger
skipped it for a reason the audit needs to surface.

| Domain | Threshold | Override input |
|---|---|---|
| Crops | 100 orphans | `crops-max` |
| Screens | 50 orphans | `screens-max` |
| Processed contributions | 50 orphans | `contributions-max` |
| Anchors (opt-in) | — | `include-anchors` |

Exit code 1 if any threshold is breached; the workflow then fails the
scheduled run and GitHub emails the repo owner. No auto-fix —
surfacing the anomaly forces a root-cause look.

### One-shot drain (`admin_drain_stale_staging.py`)

`drain_stale_staging.yml` is **workflow_dispatch only — no cron**. It
is reserved for the case where the audit flagged a breach, the cause
has been understood and patched, and the leaked orphans need to be
mopped up by hand. Scheduling it would silently paper over merger
bugs; the manual gate is deliberate.

### Virtual-poison audit (`admin_audit_virtual_poison.py`)

`audit_virtual_poison.yml` (monthly cron, 1st 05:00 UTC) counts colourful
`__empty__`/`__inactive__` crops in `data/` not yet resolved in the review
ledger `data/reviewed_virtual.jsonl` — real icons mislabeled as empty slots,
which the client logs as `CommunitySeed: POISON skip`. Breach exits 1 and
emails the owner. Cleanup is the manual review in `admin_reject_crops.py`
(`--scan` → montage + decisions TSV → `--apply`: reject / relabel / KEEP),
also driveable from the `admin_console.py` GUI. RELABEL names are validated
against sto-warp's `warp.data.cargo.canonical_names()`. Rejected shas are
barred from re-promotion by the denylist read in `democratic_merge_crops.py`.

---

## 6. Deployment (HF Space)

Production runs as a HuggingFace Space (`sets-sto/warp-backend`). A push
to `main` that touches a runtime file triggers `.github/workflows/
deploy_space.yml`, which runs `deploy_space.py` to upload the four runtime
files (`main.py`, `requirements.txt`, `space/Dockerfile`, `space/README.md`)
to the Space in one commit → one rebuild. `render.yaml` is a legacy fallback
target, not the live host.

| Aspect | Value |
|---|---|
| Host | HF Space, Docker SDK, `cpu-basic` (sleeps after ~48h idle) |
| Python | 3.12 (from `space/Dockerfile`) |
| Start | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/health` |
| Deploy | auto via `deploy_space.py` on push to `main` (secret `HF_TOKEN`) |
| Pinning | All `requirements.txt` entries pinned with `==` |

### Required env vars

| Var | Purpose |
|---|---|
| `HF_TOKEN` | HF write token (single shared backend token) |
| `HF_REPO_ID` | Model + knowledge repo (default `sets-sto/warp-knowledge`) |
| `HF_ICONS_REPO_ID` | Icon dataset repo (default `sets-sto/sto-icon-dataset`) |
| `ADMIN_KEY` | Retained for `/admin/merge` 410 response only |
| `GH_TOKEN` | GitHub PAT with `workflow` scope, used by `/webhooks/hf-dataset` |
| `GH_REPO` | `owner/name` of this repository |
| `MAX_REQ_PER_IP` | Optional override, default 500/day |
| `MAX_REQ_PER_INSTALL` | Optional override, default 500/day |

### Known pitfalls

| Issue | Root cause | Fix |
|---|---|---|
| `Port scan timeout reached, no open ports detected` | `starlette-1.0.0` major release installed via unpinned `starlette>=…` | Pin all deps with `==` in `requirements.txt` |
| Deploy picks up wrong package versions | No upper bounds → new releases break compatibility | Use `==` pins, update deliberately |
| `AssertionError: Torch not compiled with CUDA enabled` | Nested `torch.device()` in condition was always truthy | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `AttributeError: 'RepoFolder' object has no attribute 'type'` | `list_repo_tree()` returns `RepoFolder` with no `.type` | `isinstance(entry, RepoFolder)` |
| Training run >1 h on CI | Per-contributor `snapshot_download` loop did N full metadata scans | Single bulk `snapshot_download` with all patterns |
| CPU training exceeds 60 min CI limit | EfficientNet-B0 × 30 epochs on CPU | `deadline = monotonic() + 50 * 60`, check before each epoch |
| `snapshot_download` hangs ~1 h on a single file | `httpx` async I/O ignores socket timeout | `urllib.request` + `ThreadPoolExecutor(16)` + 120 s socket timeout |

---

## 7. Related documentation

- [Data Lifecycle](DATA_LIFECYCLE.md) — end-to-end data flow, staging
  vs data/ contract, Z3 thresholds, drain-on-promote, audit safety net.
- [User Guide](user_guide.md) — endpoint reference and admin commands.
- [Main README](../README.md) — setup, env vars, endpoint summary.
- [Agent Guidelines](../CLAUDE.md) — AI-assisted-dev rules.
- [Changelog](../CHANGELOG.md) — release notes.
