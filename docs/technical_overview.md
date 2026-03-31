# Technical Overview - SETS WARP Backend

## System Architecture
The backend serves as a central hub for the WARP (Weapon & Armor Recognition Program) ecosystem. It facilitates community-driven data collection and automated model training.

### Components
1. **FastAPI Service ([main.py](../main.py))**:
   - Handles contributions from mobile clients.
   - Serves the merged knowledge base and model version metadata.
   - Manages rate limiting and basic data validation.
2. **Model Trainer ([admin_train.py](../admin_train.py))**:
   - A standalone script for democratic voting on contributions.
   - Trains two PyTorch models: `icon_classifier` (EfficientNet-B0) and `screen_classifier` (MobileNetV3-Small).
   - Builds `ship_type_corrections.json` from Ship Type / Ship Tier OCR correction pairs.
   - Uploads trained models and correction map to Hugging Face Hub.
3. **Data Storage (Hugging Face Dataset)**:
   - Stores raw contributions, crops, and screenshots.
   - Hosts the final models, metadata, and correction maps.

## Core Workflows

### 1. Data Contribution
- Client sends a base64 encoded PNG crop + metadata to `/contribute`.
- Backend validates the image (size, uniformity) and uploads it to the `contributions/` folder in the HF Dataset.

### 2. Model Training (Automated/Manual)
- Triggered via CLI, GitHub Actions Schedule, or [Webhook Dispatcher](../main.py).
- `admin_train.py` scans `staging/` folders in the dataset.
- Applies democratic voting on icon crops: 1 unique `install_id` = 1 vote.
- Downloads winning samples and trains/fine-tunes models using PyTorch.
- Uploads updated models and `model_version.json` to the model repository.

### 3. Ship Type / Tier OCR Correction Map
- `collect_text_corrections()` filters staging annotations where `slot` is `Ship Type`
  or `Ship Tier`.
- For each `(ml_name, name)` pair where `ml_name != name`: votes are cast per `install_id`.
- Majority wins per `ml_name` key → `ship_type_corrections.json`:
  ```json
  {"F1eet Support Cruiser": "Fleet Support Cruiser", ...}
  ```
- Uploaded to `sets-sto/warp-knowledge/models/ship_type_corrections.json`.
- Clients download and apply corrections in `text_extractor.py` before ShipDB lookup.
- Community anchors threshold: groups with n=1 contributor are accepted as tentative
  truth; n≥2 contributors use median aggregation. (Changed from n≥3 in 2026-03-31.)

### 4. Knowledge Merging
- [admin_merge.py](../admin_merge.py) (or `/admin/merge` endpoint) consolidates raw contributions into a unified `knowledge.json` mapping.

## Infrastructure
- **Hosting**: Render (FastAPI service).
- **CI/CD**: GitHub Actions (for automated training triggers). See [.github/workflows/train_central_model.yml](../.github/workflows/train_central_model.yml).
- **Storage**: Hugging Face Hub (Dataset repo).

## GitHub Actions Training Workflow

The workflow (`.github/workflows/train_central_model.yml`) runs on a schedule (every hour) and on manual dispatch. Key characteristics:

- **Runner**: `ubuntu-latest` — **no GPU**. PyTorch is installed with `--index-url https://download.pytorch.org/whl/cpu`. Training runs on CPU.
- **Timeout**: 60 minutes hard cap.
- **Skip logic**: `--skip-if-unchanged` exits early (~60s) if no new crops have arrived since the last training run.

### Known Pitfalls

| Issue | Root cause | Fix applied |
|---|---|---|
| `AssertionError: Torch not compiled with CUDA enabled` | Nested `torch.device()` in condition was always truthy → always selected `cuda` | Use `torch.cuda.is_available()` directly |
| `AttributeError: 'RepoFolder' object has no attribute 'type'` | `list_repo_tree()` returns `RepoFolder` objects which have no `.type` attribute | Use `isinstance(e, RepoFolder)` |
| Download timeout >1h | Per-contributor `snapshot_download` loop caused N full dataset metadata scans | Single bulk `snapshot_download` call with all patterns |
| CPU training exceeds 60 min limit | EfficientNet-B0 on CPU with ~3000 samples: one epoch >8 min × 30 epochs = hours | `deadline` parameter in both `train()` functions; `main()` sets `now + 50 min` |
| `snapshot_download` hangs indefinitely (~1h) on a specific file | `httpx` (used internally) is async — `socket.setdefaulttimeout` has no effect on it; one file stalls the entire download silently | Replaced with `urllib.request` + `ThreadPoolExecutor(16)` — urllib uses blocking sockets, socket timeout of 120 s kills any stalled read |

### Time Budget

Training budget is set at **50 minutes** from when `main()` enters the training block. This leaves approximately 10 minutes for model upload to HF Hub. Both `train()` and `train_screen_classifier()` accept a `deadline: float | None` parameter (a `time.monotonic()` timestamp); each epoch checks the deadline before starting and exits early if exceeded. The best model state accumulated so far is still saved and uploaded.

### Crop & Screenshot Download Strategy

Files are downloaded in parallel using `urllib.request` in a `ThreadPoolExecutor(max_workers=16)`. Only the exact files needed (known SHA + install_id pairs from the voting step) are fetched — no full-repo metadata scan. `socket.setdefaulttimeout(120)` is set globally before downloads begin; any stalled TCP read raises `socket.timeout` after 2 minutes, which is caught per-file and logged as a skip. `snapshot_download` (httpx/async) was abandoned because httpx async I/O bypasses Python's socket timeout mechanism entirely.

### HF Listing Strategy

`_list_staging_folders()` uses `list_repo_tree(..., recursive=False)` to fetch only the top-level `staging/` directory — O(1) API call instead of a full recursive scan. Falls back to `list_repo_files()` only on exception.

## Render Deployment

The FastAPI service is deployed on Render as a web service (`render.yaml`). Key notes:

- **Python version**: 3.12 (specified in `render.yaml`)
- **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health check**: `/health`
- **Dependencies**: `requirements.txt` with **pinned exact versions** (`==`) — unpinned deps caused `starlette-1.0.0` to be installed, which is incompatible with `fastapi 0.135.x` and caused uvicorn to start and immediately exit (port scan timeout on Render).

### Render Known Pitfalls

| Issue | Root cause | Fix |
|---|---|---|
| `Port scan timeout reached, no open ports detected` | `starlette-1.0.0` (major release, breaking changes) installed via unpinned `starlette>=...` | Pin all deps with `==` in `requirements.txt` |
| Deploy picks up wrong package versions | No upper bounds in requirements → new releases break compatibility | Use `==` pins, update deliberately |

## Related Documentation
- [User Guide](./user_guide.md)
- [Agent Guidelines (CLAUDE.md)](../CLAUDE.md)
- [Main README](../README.md)
