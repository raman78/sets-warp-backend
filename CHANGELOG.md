# Changelog

## [Unreleased]

### Added
- **Documentation refresh.** Rewrote `docs/technical_overview.md` to
  cover the four democratic mergers (`democratic_merge_crops.py`,
  `democratic_merge_anchors.py`, `democratic_merge_screens.py`,
  `admin_merge.py`), the staging vs `data/` contract, Z3 asymmetric
  thresholds (NEW=1, UPDATE>=2), drain-on-promote, the bulk endpoints
  (`/contribute/bulk-crops`, `/upload/screen-types`, `/upload/anchors`),
  and the audit safety net (`admin_audit_staging.py` + monthly
  `audit_staging_health.yml`, plus the manual-dispatch
  `admin_drain_stale_staging.py`). Added `docs/DATA_LIFECYCLE.md` with
  an end-to-end client → backend → staging → mergers → data/ → training
  → models → client diagram, mirroring the client-side
  `docs/DATA_LIFECYCLE.md` in the `sto-warp` repo. Updated
  `docs/user_guide.md` and `README.md` to list every endpoint, script,
  and workflow; previous docs only covered `admin_train.py` and
  `admin_merge.py`. No code changes.
- **Phase 1 — backend-proxy bulk endpoints.** Added `POST /contribute/bulk-crops`, `POST /upload/screen-types`, `POST /upload/anchors` to `main.py`. Each accepts a batch (≤50 crops / ≤20 screens / ≤20 grids) and produces a single HF commit to `sets-sto/sto-icon-dataset` (configurable via `HF_ICONS_REPO_ID`). These let the `sto-warp` client drop its write-scoped HF token in Phase 2 — uploads will flow through the backend's server-side token instead. Mirrors validation + last-wins jsonl dedup from `warp/trainer/sync.py`.
- Created agent guidelines (`CLAUDE.md`, `GEMINI.md`, `GPT.md`) to standardize AI assistant behavior.
- Added `/docs` directory with `technical_overview.md` and `user_guide.md`.
- Added `_load_env()` to `main.py` for seamless local development with `.env` files.
- Automated training trigger via GitHub Actions API in `main.py` (replacing Bitbucket).

### Changed
- Translated `admin_merge.py` and all internal logs/comments to English for consistency.
- Updated `main.py` to use **Atomic Uploads** (via `create_commit`) for data contributions, ensuring data integrity.
- Optimized Hugging Face repository listing in `main.py` and `admin_merge.py` using `list_repo_tree` to prevent timeouts.
- Consolidated per-contributor `snapshot_download` loops into a single bulk call with all patterns for both icon crops and screen screenshots, eliminating redundant full-repo metadata scans on each call.
- Replaced `snapshot_download` (httpx/async) with direct parallel `urllib.request` downloads in `ThreadPoolExecutor(max_workers=16)`: urllib uses blocking sockets so `socket.setdefaulttimeout(120)` applies, killing stalled transfers after 2 min. Only exact needed files are downloaded.
- Pinned all production dependencies to exact versions in `requirements.txt` to ensure reproducible Render deploys.

### Fixed
- Fixed `httpx.RemoteProtocolError: Server disconnected without sending a response` in `admin_train.py` by optimizing repository scanning.
- Fixed potential data inconsistency in contributions by grouping JSON and PNG uploads into a single HF commit.
- Fixed GitHub Actions training workflow always targeting CUDA device: nested `torch.device()` in condition was always truthy, forcing `cuda` even on CPU-only runners → replaced with `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- Fixed `AttributeError: 'RepoFolder' object has no attribute 'type'` in `_list_staging_folders`: replaced `e.type == 'dir'` check with `isinstance(e, RepoFolder)`, preventing silent fallback to slow full-repo listing.
- Fixed GitHub Actions training timeout (>1h): per-contributor `snapshot_download` loop caused N full dataset metadata scans; replaced with a single call listing all patterns at once.
- Fixed CPU training exceeding 60 min CI timeout: added `deadline` parameter (monotonic timestamp) to `train()` and `train_screen_classifier()`; `main()` sets deadline = now + 50 min, leaving ~10 min buffer for model upload.
- Fixed Render deploy failure: `starlette-1.0.0` (major release) is incompatible with `fastapi 0.135.x`, causing uvicorn to start and immediately shut down — port scan timeout on Render. Pinning `starlette` via `fastapi==0.135.1` in `requirements.txt` resolves the issue.
- Fixed `snapshot_download` hanging indefinitely (~1h) on a specific file: `httpx` async I/O ignores `socket.setdefaulttimeout`; replaced with `urllib.request` parallel downloads which respect socket-level timeouts. Stalled downloads now abort after 2 minutes.
