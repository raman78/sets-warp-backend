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
   - Uploads trained models to Hugging Face Hub.
3. **Data Storage (Hugging Face Dataset)**:
   - Stores raw contributions, crops, and screenshots.
   - Hosts the final models and metadata.

## Core Workflows

### 1. Data Contribution
- Client sends a base64 encoded PNG crop + metadata to `/contribute`.
- Backend validates the image (size, uniformity) and uploads it to the `contributions/` folder in the HF Dataset.

### 2. Model Training (Automated/Manual)
- Triggered via CLI, GitHub Actions Schedule, or [Webhook Dispatcher](../main.py).
- `admin_train.py` scans `staging/` folders in the dataset.
- Applies democratic voting: 1 unique `install_id` = 1 vote.
- Downloads winning samples and trains/fine-tunes models using PyTorch.
- Uploads updated models and `model_version.json` to the model repository.

### 3. Knowledge Merging
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

### Time Budget

Training budget is set at **50 minutes** from when `main()` enters the training block. This leaves approximately 10 minutes for model upload to HF Hub. Both `train()` and `train_screen_classifier()` accept a `deadline: float | None` parameter (a `time.monotonic()` timestamp); each epoch checks the deadline before starting and exits early if exceeded. The best model state accumulated so far is still saved and uploaded.

### HF Listing Strategy

`_list_staging_folders()` uses `list_repo_tree(..., recursive=False)` to fetch only the top-level `staging/` directory — O(1) API call instead of a full recursive scan. Falls back to `list_repo_files()` only on exception.

## Related Documentation
- [User Guide](./user_guide.md)
- [Agent Guidelines (CLAUDE.md)](../CLAUDE.md)
- [Main README](../README.md)
