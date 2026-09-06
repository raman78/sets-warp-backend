# sets-warp-backend

Backend service for the **WARP** (Weapon & Armor Recognition Program)
ecosystem. Receives community-contributed icon crops, screen-type
screenshots, anchor grids, and pHash overrides from sto-warp clients;
runs democratic voting; trains the EfficientNet-B0 icon classifier,
MobileNetV3-Small screen classifier, and ArcFace embedder; publishes
the trained models back out for every install.

This backend holds the only HuggingFace write token in the system —
clients never touch HF directly for writes.

---

## Architecture

- **FastAPI on a Hugging Face Space** (`sets-sto-warp-backend.hf.space`,
  Docker SDK) — receives client uploads, serves `/knowledge`,
  `/model/version` and `/quota`. A push to `main` touching a runtime file
  deploys it via `.github/workflows/deploy_space.yml`. `render.yaml` is a
  legacy fallback and is not the live host; anything here that still reads
  as "on Render" is stale, and one such line — the proxy assumption in
  `_get_client_ip` — is why `/quota` exists.
- **HuggingFace Datasets** — `sets-sto/sto-icon-dataset` (raw + curated
  data), `sets-sto/warp-knowledge` (models + pHash overrides).
- **GitHub Actions** — runs the four democratic mergers every 2 h,
  trains the classifiers hourly, trains the embedder daily, runs the
  staging audit monthly.

For the end-to-end data flow see
[docs/DATA_LIFECYCLE.md](docs/DATA_LIFECYCLE.md). For component-level
detail see [docs/technical_overview.md](docs/technical_overview.md).

---

## Setup

### 1. HuggingFace

- Create two Dataset repos: `sets-sto/sto-icon-dataset` (data) and
  `sets-sto/warp-knowledge` (models).
- Generate one **Write Token** in HF Settings → Access Tokens.

### 2. GitHub Actions

In **Settings → Secrets and variables → Actions**, add:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace write token |

Optionally add **Variables** for non-default repo names:

| Variable | Default |
|---|---|
| `HF_DATASET` | `sets-sto/sto-icon-dataset` |
| `HF_REPO_ID` | `sets-sto/warp-knowledge` |

### 3. Deployment (HF Space)

Set these as **Space secrets / variables**:

| Var | Purpose |
|---|---|
| `HF_TOKEN` | HF write token |
| `HF_REPO_ID` | Model + knowledge repo (default `sets-sto/warp-knowledge`) |
| `HF_ICONS_REPO_ID` | Icon dataset repo (default `sets-sto/sto-icon-dataset`) |
| `ADMIN_KEY` | Retained for `/admin/merge` 410 response |
| `GH_TOKEN` | GitHub PAT with `workflow` scope, used by `/webhooks/hf-dataset` |
| `GH_REPO` | This repo, e.g. `sets-sto/sets-warp-backend` |
| `MAX_REQ_PER_IP` | Optional, default 500/day |
| `MAX_REQ_PER_INSTALL` | Optional, default 500/day |

`requirements.txt` pins every dependency with `==` and `render.yaml` pins
Python to 3.12 for the legacy fallback. Do not relax the pins without a
deliberate test against the Space's Docker runtime, which is what
production runs on.

---

## Endpoints

### Client-facing

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check + whether ingestion validation is enforcing |
| `GET` | `/knowledge` | Merged pHash → item-name table |
| `GET` | `/model/version` | Latest trained model metadata |
| `GET` | `/config/labels` | Backend-side label map |
| `POST` | `/contribute` | Single pHash contribution (legacy) |
| `POST` | `/contribute/bulk-crops` | Up to 50 confirmed crops per batch |
| `POST` | `/upload/screen-types` | Up to 20 screen-type screenshots per batch |
| `POST` | `/upload/anchors` | Up to 20 anchor grids per batch |

### Internal

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/webhooks/hf-dataset` | HF webhook → triggers GH training |
| `POST` | `/admin/merge` | Retired — returns HTTP 410 |

---

## Scripts

| Script | Role |
|---|---|
| `democratic_merge_crops.py` | Promote icon crops staging → `data/crops/` |
| `democratic_merge_anchors.py` | Promote anchor grids → `data/anchors/` |
| `democratic_merge_screens.py` | Promote screen types + text corrections |
| `admin_merge.py` | Fold pHash contributions into `knowledge.json` |
| `admin_train.py` | Train EfficientNet-B0 + MobileNetV3-Small |
| `admin_train_metric.py` | Train ArcFace embedder + gallery |
| `admin_audit_staging.py` | Read-only orphan check (monthly cron) |
| `admin_drain_stale_staging.py` | Manual cleanup when audit breaches |
| `admin_scrub_knowledge.py` | Remove bad pHash entries |
| `admin_clean_labels.py` | Remove bad crop labels |

---

## GitHub Actions workflows

| Workflow | Cadence | Runs |
|---|---|---|
| `merge_staging.yml` | every 2 h, `22 */2 * * *` | All four mergers |
| `train_central_model.yml` | hourly, `0 * * * *` | `admin_train.py` |
| `train_metric_model.yml` | daily, `45 0 * * *` | `admin_train_metric.py` |
| `audit_staging_health.yml` | monthly, `0 4 1 * *` | `admin_audit_staging.py` |
| `drain_stale_staging.yml` | manual only | `admin_drain_stale_staging.py` |

---

## Manual commands

```sh
# Dry-run all mergers
python democratic_merge_crops.py
python democratic_merge_anchors.py
python democratic_merge_screens.py
python admin_merge.py

# Apply with default Z3 thresholds (NEW=1, UPDATE>=2)
python democratic_merge_crops.py --apply
python admin_merge.py --apply

# Train with upload
python admin_train.py --train --min 1

# Local FastAPI smoke test
python main.py
```

---

## Documentation

- [Data Lifecycle](docs/DATA_LIFECYCLE.md) — end-to-end data flow
- [Technical Overview](docs/technical_overview.md) — components and contracts
- [User Guide](docs/user_guide.md) — admin commands + recovery procedures
- [Ingestion Validation](docs/INGESTION_VALIDATION.md) — the upload whitelist and what it gates
- [Docs index](docs/README.md) — one line per document
- [Agent Guidelines](CLAUDE.md) — AI-assisted-dev rules
- [Changelog](CHANGELOG.md) — release notes

---

## License

MIT
