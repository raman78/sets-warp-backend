# Data Lifecycle — server-side view

How a single confirmed icon crop travels from a client's WARP CORE
window, through this backend, into the `data/` folder that trains every
model, and back out as a model file the client downloads. This file is
the server-side counterpart to
[`DATA_LIFECYCLE.md`](https://github.com/sets-sto/sto-warp/blob/develop/docs/DATA_LIFECYCLE.md)
in the sto-warp client repo — both ends of the same pipeline,
documented from each side.

For the component breakdown (FastAPI service, the four mergers, the
trainers, the audit) see [`technical_overview.md`](technical_overview.md).

---

## 1. End-to-end flow

```
┌──────────────────────────────────────────────────────────────────┐
│  CLIENT (sto-warp WARP CORE)                                     │
│                                                                  │
│  user confirms a bbox + label                                    │
│         │                                                        │
│         ▼                                                        │
│  ~/.local/share/warp/training_data/                              │
│    crops/<sha>.png                                               │
│    annotations.json                                              │
│         │                                                        │
│         │  client-side poison filter                             │
│         │  (drops __empty__, __inactive__, __boff_*,             │
│         │   "Test Item Name" — never uploaded)                   │
│         │                                                        │
│         │  SyncWorker (every 10 min)                             │
│         │  batched 50 crops / 20 screens / 20 grids per request  │
│         ▼                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │  HTTPS, no HF token on client
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  BACKEND (FastAPI on HF Space)                                   │
│                                                                  │
│  /contribute/bulk-crops    /upload/screen-types  /upload/anchors │
│         │                                                        │
│         │  per-item validation                                   │
│         │  (size, regex, byte caps; see main.py constants)       │
│         │                                                        │
│         │  rate-limit (500/day per IP, 500/day per install_id)   │
│         │                                                        │
│         │  one HF commit per batch — server-side HF token        │
│         ▼                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
   HF Dataset: sets-sto/sto-icon-dataset
   staging/<install_id>/crops/<sha>.png
   staging/<install_id>/annotations.jsonl
   staging/<install_id>/screen_types/<T>/<sha>.png
   staging/<install_id>/anchors_grid_<sha8>.json
          │
          │  GitHub Actions: merge_staging.yml
          │  cron: 22 */2 * * *  (every 2 h, off-peak offset)
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  THE FOUR DEMOCRATIC MERGERS                                     │
│                                                                  │
│   democratic_merge_crops.py    sha256 → data/crops + annotations │
│   democratic_merge_anchors.py  (bt, aspect) → data/anchors       │
│   democratic_merge_screens.py  sha → data/screen_types + text    │
│   admin_merge.py               phash → knowledge.json (other repo│
│                                       sets-sto/warp-knowledge)   │
│                                                                  │
│  per key:                                                        │
│    threshold = 1            if key ∉ data/   (NEW   — Z3)        │
│    threshold = --min (=2)   if key ∈ data/   (UPDATE — Z3)       │
│                                                                  │
│  promote + drain in ONE HF commit:                               │
│    + CommitOperationAdd    (data/…)                              │
│    + CommitOperationDelete (staging/<iid>/…)                     │
│                                                                  │
│  poison filter: drops __* and "Test Item Name" again,            │
│                 even if the client sent them                     │
└─────────┼────────────────────────────────────────────────────────┘
          │
          ▼
   HF Dataset: sets-sto/sto-icon-dataset
   data/crops/<sha>.png                  (winning crops, by sha)
   data/annotations.jsonl                (one line per winning sha)
   data/screen_types/<T>/<sha>.png       (winning screen-type imgs)
   data/text_corrections.jsonl           (OCR corrections, by key)
   data/anchors/<build_type>_<bucket>.json  (winning anchor grids)
          │
          │  GitHub Actions: train_central_model.yml
          │  cron: 0 * * * *  (hourly)
          │  + train_metric_model.yml (daily, 00:45 UTC)
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  TRAINING                                                        │
│                                                                  │
│  admin_train.py        EfficientNet-B0 (icon) + MobileNetV3      │
│                        (screen) — fine-tune from previous .pt    │
│                        Builds ship_type_corrections.json from    │
│                        Ship Type/Tier text corrections           │
│                                                                  │
│  admin_train_metric.py ArcFace embedder + gallery (.npz)         │
│                                                                  │
│  Hard caps on CI:                                                │
│    GH Actions step:   60 min                                     │
│    In-script deadline: 50 min  (leaves 10 min for upload)        │
│                                                                  │
│  Skip if unchanged: training_manifest.json compares SHA set      │
│  against the previous run; <10 new crops → fast-exit (~60 s).    │
└─────────┼────────────────────────────────────────────────────────┘
          │  one HF commit per training run, atomic upload
          ▼
   HF Dataset: sets-sto/warp-knowledge
   models/icon_classifier.pt        + label_map.json + …meta.json
   models/screen_classifier.pt      + screen_classifier_labels.json
   models/icon_embedder.pt          + embedder_label_map.json + .npz
   models/model_version.json        (trained_at, n_classes, val_acc)
   models/ship_type_corrections.json
   knowledge.json                   (pHash overrides)
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  CLIENT (ModelUpdater)                                           │
│                                                                  │
│  GET /model/version            every 15 min (_CHECK_INTERVAL)    │
│       └─ remote.trained_at > local.trained_at?                   │
│            ├─ yes → hf_hub_download every required file          │
│            │        copy to warp/models/ atomically              │
│            │        reset_ml_session(), reload immediately       │
│            └─ no  → skip                                         │
│                                                                  │
│  Demotion guard: install only when remote is strictly newer.     │
└──────────────────────────────────────────────────────────────────┘
```

End-to-end best case: roughly 3.5 h from client confirmation to model
update on every install (10 min sync delay + ≤2 h merge wait + ≤1 h
train + ≤15 min model-version poll on the recipient).

---

## 2. The two HF repos

The data and the models live in **separate datasets**. This split is
the load-bearing simplification of the whole system:

| Repo | Owns |
|---|---|
| `sets-sto/sto-icon-dataset` | Raw uploads (`staging/`) and curated data (`data/`). Receives every client upload; sourced by the mergers; sourced by the trainers. |
| `sets-sto/warp-knowledge` | Trained model artefacts (`models/`) and the pHash override table (`knowledge.json`). Downloaded by every client. |

- The mergers only touch `sto-icon-dataset` (and `admin_merge.py` only
  touches `warp-knowledge` for the pHash table).
- The trainers read `sto-icon-dataset/data/` and write to
  `warp-knowledge/models/`.
- The clients read both — `warp-knowledge` for models, the icon
  tarball release of `sto-icon-dataset` for cold-start crops.

Writing the training output to a separate repo means the icon dataset
can churn freely without affecting model delivery, and a bad model
upload can be rolled back without touching the dataset.

---

## 3. The staging vs data contract

`staging/<install_id>/` is **per-install raw votes**, ephemeral. Each
install_id writes to its own subtree; no two installs can collide.
Files here are pending democratic review.

`data/` is the **community-confirmed** content, the only thing that
trains models. One entry per semantic key (one crop per SHA, one anchor
per `(build_type, aspect_bucket)`, one screen per SHA).

| Property | `staging/<install_id>/` | `data/` |
|---|---|---|
| Per-install | yes | no — one canonical entry per key |
| Receives client uploads | yes | no |
| Read by mergers | yes (input) | yes (existing baseline) |
| Read by trainers | no | yes |
| Drained on promote | yes (atomic with promote) | no |
| Audited monthly | yes (orphan check) | no |

The mergers are the only writers to `data/`. Clients never write to
`data/` directly, never observe `data/` mid-promote (atomic commit),
and never need to know which install_id contributed which key.

---

## 4. Z3 — asymmetric thresholds

| Action | Required votes |
|---|---|
| NEW (key not in `data/` yet) | 1 |
| UPDATE (key already in `data/`) | ≥ N, default 2 (`--min` flag) |

The threshold is computed per key in the merger's `_merge()`:

```python
threshold = min_votes if key in existing else 1
accepted  = count >= threshold
```

The override flag is plumbed through `merge_staging.yml` as
`workflow_dispatch.inputs.min_votes` — operators can raise it for
adversarial periods without code changes.

The asymmetry is intentional. NEW is the bottleneck for coverage; the
first contributor to confirm a new exotic console is almost always
right. UPDATE is the bottleneck for safety; overwriting an established
label is where a careless or hostile vote could do damage.

---

## 5. Drain on promote

Every promotion to `data/` is paired with deletion of the
corresponding staging entries, **in the same HF commit**. Steady-state
staging size therefore stays bounded by `per_install_uploads × 2 h`
(the merger cadence), not by total lifetime contributions.

Look for `DRAIN:` lines in the merger output for the per-cycle count:

```
DRAIN: 47 staging crops, 12 staging annotations
```

The drain is the reason the audit works (§6 in
`technical_overview.md`): orphans should be impossible. When the
audit counts non-zero orphans, the merger is leaking and the cause
must be patched, not papered over.

---

## 6. Where to look when something goes wrong

| Symptom | Where to look first |
|---|---|
| Client says upload succeeded, but the new label never appears in `data/` | Merger run logs in `merge_staging.yml` — is the key showing up in votes? Did the threshold block it? |
| Audit failed, repo owner got an email | `audit_staging_health.yml` run log + the `AUDIT:` lines in stdout. Identify which domain breached, then read the corresponding merger to find the drain bug. |
| A bad label leaked into `data/` | Confirm the poison filter status (`_is_poison_name` in the merger; `_poison_filter_enabled` on the client). Use `admin_clean_labels.py` to remove the bad entries; do not edit `annotations.jsonl` by hand. |
| Model regressed after retrain | Inspect `model_version.json` history in `warp-knowledge` — `trained_at`, `val_acc`, `n_classes`. Roll back by uploading an older snapshot from a previous training run's artefacts. |
| Merger fails halfway through a commit | Atomic commits — half-applied state is impossible. The next run picks up where the previous one left off; no manual recovery needed. |
| Staging files leak (audit breached, cause patched) | Manually dispatch `drain_stale_staging.yml`. Never schedule it. |

---

## 7. Cadence summary

| Job | Cadence | Workflow |
|---|---|---|
| Merge staging → data/ | every 2 h, `22 */2 * * *` | `merge_staging.yml` |
| Train icon + screen classifier | hourly, `0 * * * *` | `train_central_model.yml` |
| Train ArcFace embedder | daily, `45 0 * * *` | `train_metric_model.yml` |
| Audit staging health | monthly, `0 4 1 * *` | `audit_staging_health.yml` |
| Drain stale staging | manual only | `drain_stale_staging.yml` |

The two-hour merge cadence is deliberately offset (`:22`) from the
classifier hour boundary so the trainer never starts mid-merge — by
the time `0 * * * *` fires, the most recent `:22` merger is already
done or already an hour stale.

---

## 8. Related documentation

- [Technical Overview](technical_overview.md) — components and their
  contracts.
- [User Guide](user_guide.md) — endpoints + admin commands.
- [sto-warp DATA_LIFECYCLE](https://github.com/sets-sto/sto-warp/blob/develop/docs/DATA_LIFECYCLE.md)
  — client-side counterpart of this document.
