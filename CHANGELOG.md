# Changelog

## [Unreleased]

### Fixed
- **A pHash vote is never forgotten.** `admin_merge.py` tallied only the
  contributions that arrived since its last run, then marked them processed,
  so a dissent that fell short was lost and changing an entry needed two
  matching votes inside one two-hour run. 205 such votes were found
  forgotten on 2026-09-25. `knowledge.json` now keeps a running tally,
  `votes` (phash → {name: votes}), which replaces the per-run `losers`. An
  entry changes when a challenger has more votes than the current name and
  at least `--min`. A hash can be shared by different pictures (one carried
  five items), so the tally keeps every name and `GET /knowledge` returns it
  beside `knowledge`, which older clients keep reading unchanged.
  `admin_scrub_knowledge.py` removes a scrubbed name from the tally too.
- **A maintainer's relabel now reaches a model.** `training_manifest.json`
  recorded crop SHAs alone, and a RELABEL keeps the SHA and changes the
  name — so `--skip-if-unchanged` saw an identical set, printed "no new
  crops" and skipped. For ever: a relabel contributes zero new SHAs, so it
  never counted towards `MIN_NEW_CROPS` either. The manifest now carries
  `label_digest`, a hash of the sorted (SHA, label) pairs; a changed digest
  defeats the skip and waives the threshold. It is taken before the
  min-votes filter narrows the set, so it describes what the next run
  compares against. A manifest without a digest is compared on SHAs alone,
  as before, rather than forcing one pointless hour of training.

- **The movement audit times the waiting work, not the calendar.** It
  breached when crops were in staging and the last promotion was older than
  the limit — two independent facts. On 2026-09-18 nothing had been uploaded
  for six days, so nothing had been promoted for six days; 115 crops arrived
  that morning, the audit ran four hours later and reported "the merge is
  running and not landing" about a pipeline that landed all 115 at the next
  run, 54 minutes afterwards. It now measures the oldest upload the last
  promotion did not cover, and falls back to the promotion's own age for the
  case that really is a stall — staging holding crops that a promotion ran
  over and left behind.
- **The crop review no longer keeps its own copy of "looks colourful".**
  `admin_reject_crops._looks_real` now calls sto-warp's
  `icon_matcher._virtual_crop_looks_real`, the way `_looks_blank` and
  `load_canonical_names` already delegate — one definition, so the tool
  flags exactly what the client refuses to seed. The copy had drifted:
  the client learned that the game's yellow *NEW* ribbon is chrome rather
  than icon content, and the copy had not, so fifteen empty slots wearing
  one were queued for review and fourteen of them were labelled correctly.

  The local bright/rich copy stays as the fallback for an environment
  without sto-warp, and `audit_virtual_poison.yml` now installs the client
  with `--no-deps` so CI has the real rule. Both the scan and the audit
  print a `Heuristic:` line naming which one answered — the fallback
  breaches on correct data, and a count alone cannot say that.
- **A screenshot filed under two screen types now has both copies drained.**
  The vote was already one per `(install_id, sha)` and stays that way, but
  only the *voting* path was recorded, so a second copy from the same install
  sat in staging for ever. That matters because the vote goes to whichever
  copy `glob` reaches first: with a stale copy surviving, every merge run was
  a fresh coin toss over the label.

  The state arises client-side — sto-warp's `set_screen_type` copied a
  screenshot into the new type's folder and left the old copy, so the
  classifier's guess and the user's correction of it both went up. Measured on
  the maintainer's store 2026-09-11: 534 files against 287 labels, 110
  screenshots filed under two or three mutually exclusive types, 20 as both
  `BOFFS` and `SPACE_BOFFS`. The client no longer creates it; this makes sure
  the backend clears what it already holds, and stays right if another client
  does the same.

### Added
- **`GET /quota` — which rate-limit bucket is full, and under which address.**
  A refused client sees only `429` and cannot tell its own install bucket from
  the per-IP one it shares with everyone behind the same address. The endpoint
  reports both, plus `resolved_ip` and the raw `forwarded_for` header. It is a
  read and is not rate limited, because a diagnostic that counted against the
  caps would be part of the problem it exists to diagnose.

  It exists to settle a question the code raises and nobody has checked:
  `_get_client_ip` takes the **rightmost** `X-Forwarded-For` entry, which
  identifies the caller only when exactly one trusted proxy sits in front of
  the app. That was true of the Render deployment the function was written for
  — the comment still says so — but production has been an HF Space for some
  time. With more than one hop, every client resolves to the same
  infrastructure address and `MAX_REQ_PER_IP` stops being a per-user cap and
  becomes a **global** 500 requests a day for the whole community. One call
  from a machine with a known public address decides it. No behaviour is
  changed here; nothing is fixed until that reading is in.

  Prompted by a client-side backlog measured 2026-09-06: 127 corrected screen
  types unshared for days, every POST answered `429`, while the client's own
  guards read as having room — they counted items and accepted contributions,
  where the server counts requests. The client half is fixed in `sto-warp`
  (`warp.backend_budget`).
- **`SKILLS` / `SPACE_SKILLS` / `GROUND_SKILLS` / `DISCARD` accepted as screen
  types.** The client has offered these labels since the skill-tree feature
  landed, but `democratic_merge_screens.SCREEN_TYPES` dropped them on the way
  from staging to `data/`, so neither could ever accumulate a single sample —
  and `admin_train.SCREEN_TYPES` would have ignored them anyway. All three
  lists now agree: ingestion whitelist (`config/labels.json`) and the merger
  take all four, the classifier trains on `SKILLS` and `DISCARD` (the
  `SPACE_`/`GROUND_` variants are stored but not separate classes, exactly as
  `TRAITS` has always worked). `SC_MIN_CLASS_SAMPLES = 5` keeps a new class out
  of the model until it has enough samples, so nothing changes until the data
  is there. DISCARD matters because a screenshot with no build content on it
  (a doff roster, a loading screen) currently has to be forced into one of the
  build types.
- **`/model/version` now reports the ArcFace embedder separately.** The
  payload carries `embedder_trained_at`, `embedder_n_classes` and
  `embedder_recall`, read from `models/icon_embedder_meta.json` on HF
  (`_load_embedder_meta_from_hf`). The softmax classifier and the embedder
  are published by two workflows with different cadences —
  `train_central_model.yml` is hourly but skips until ≥10 new crops have
  merged, `train_metric_model.yml` is daily and unconditional — so a
  single `trained_at` could not tell a client that a fresher embedder was
  waiting. Missing meta ⇒ the fields are simply absent, and older clients
  ignore them.
- **Virtual-crop review tooling.** `admin_reject_crops.py` reviews colourful
  `__empty__`/`__inactive__` crops in `data/` (real icons the client logs as
  `CommunitySeed: POISON skip`): `--scan` (read-only) shallow-clones the
  dataset, flags virtual-label crops that trip the bright/rich heuristic
  (0.15/0.15, in sync with the client's `_virtual_crop_looks_real`), and
  writes a montage PNG + a decisions TSV; pixels are read from sto-warp's
  local crop mirror when present (falls back to `hf_hub_download`). `--apply`
  commits one atomic change — REJECT drops from `data/` + drains staging,
  RELABEL rewrites the `name` (same sha), all decisions appended to a review
  ledger `data/reviewed_virtual.jsonl`. RELABEL targets are validated against
  sto-warp's live cargo (`warp.data.cargo.canonical_names()` — the source of
  truth, no re-parsing), so a typo can never enter the dataset.
- **Maintainer console.** `admin_console.py` — a PySide6 GUI (optional
  `[admin]` extra, NOT installed on the Space Docker runtime) that shells out
  to the review/merge/audit CLIs; RELABEL is a searchable cargo dropdown.
- **Anti-resurrection denylist.** `democratic_merge_crops.py` now reads the
  review ledger and skips `decision==REJECT` shas, so a re-uploaded rejected
  crop can never be re-promoted.
- **Virtual-poison audit.** `admin_audit_virtual_poison.py` + monthly
  `audit_virtual_poison.yml` (1st of month, 05:00 UTC) count colourful virtual
  crops not yet resolved in the ledger and exit 1 on breach — the automated
  reminder to review new mislabels (mirrors `admin_audit_staging.py`).
- **Automatic Space deploy.** Added `deploy_space.py` (uploads the four
  runtime files — `main.py`, `requirements.txt`, `space/Dockerfile`,
  `space/README.md` — to `spaces/sets-sto/warp-backend` in a single
  `HfApi` commit) and `.github/workflows/deploy_space.yml`, which runs it
  on every push to `main` touching a runtime file. A `git push` to GitHub
  now redeploys the live Space automatically; reuses the existing
  `HF_TOKEN` Actions secret (Space write scope). Replaces the manual
  clone/copy/push in `space/README_deploy.md`.

### Fixed
- **The ingestion whitelist was never live.** `config/labels.json` is not in
  `deploy_space.RUNTIME_FILES`, so the Space never received it;
  `_load_labels_bundled()` raised, `_get_labels()` returned
  `{'screen_types': [], 'slots': {}}`, and every gate that reads
  `if allowed_...:` switched itself off. Production served
  `/config/labels → {"screen_types": [], "slots": {}}` and accepted any
  screen_type, any anchor build_type and any slot names. The file now ships
  (and the deploy workflow triggers on it), the failure logs at ERROR instead
  of WARNING, and `GET /health` reports `validation: enforcing | DISABLED
  (empty whitelist)` so the same rot cannot go unnoticed again. The fail-open
  itself is kept: a transient HF outage must not black-hole uploads.
- **Anchor grids for slotless screens are now rejected.** A declared-but-empty
  `slots[<build_type>]` (DISCARD, SKILLS, SPACE_SKILLS, GROUND_SKILLS) means
  the screen has no icon slots, so a grid claiming one is invalid by
  definition; previously an empty list was indistinguishable from a missing
  entry and disabled the check.

### Changed
- **Docs now name HF Space (not Render) as production.** `technical_overview.md`
  (§1/§2/§6), `DATA_LIFECYCLE.md`, `CLAUDE.md`, `space/README_deploy.md`, and
  `test_backend.py`'s `BACKEND_URL` updated to reflect the live host
  (`sets-sto-warp-backend.hf.space`). `render.yaml` marked legacy-fallback.

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
