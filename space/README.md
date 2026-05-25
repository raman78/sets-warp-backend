---
title: WARP Backend
emoji: 🛰
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# sets-sto / warp-backend

Proxy FastAPI for the WARP knowledge sync pipeline.

Receives community contributions (icon crops, screen-type screenshots,
anchor grids) from the `sto-warp` client and commits them to the
`sets-sto/warp-knowledge` HuggingFace dataset.

The HF write token lives **only** as a Space secret here — clients no
longer need (or can request) write access to the dataset, which closes
the shared-token security risk that existed when each user held a copy
of `hub_token.txt`.

## Source

Primary repo: <https://github.com/raman78/sets-warp-backend>
This Space mirrors the runtime files (`main.py`, `requirements.txt`,
`Dockerfile`) from there. Code changes happen on GitHub and are pushed
to this Space as a deployment step.

## Endpoints

See the source repo for the full API reference. Main surface:

- `GET /health` — service status
- `GET /knowledge` — merged icon knowledge base (phash → name)
- `GET /model/version` — latest trained model metadata
- `POST /contribute` — single crop + label (legacy, → `sets-sto/warp-knowledge`)
- `POST /contribute/bulk-crops` — batch of confirmed crops + annotations (→ `sets-sto/sto-icon-dataset`)
- `POST /upload/screen-types` — batch of screen-type screenshots (→ `sets-sto/sto-icon-dataset`)
- `POST /upload/anchors` — batch of anchor grids (→ `sets-sto/sto-icon-dataset`)
- `POST /admin/merge` — admin-only knowledge merge

## Secrets

Configured in Space Settings → Secrets:

- `HF_TOKEN` — fine-grained write token, scoped to both
  `datasets/sets-sto/warp-knowledge` (current `/contribute`) and
  `datasets/sets-sto/sto-icon-dataset` (Phase 1 bulk endpoints)
- `HF_REPO_ID` — e.g. `sets-sto/warp-knowledge` (target for `/contribute`, `/admin/merge`)
- `HF_ICONS_REPO_ID` — e.g. `sets-sto/sto-icon-dataset` (target for Phase 1 bulk endpoints; defaults to that value if unset)
- `ADMIN_KEY` — guards `/admin/*` endpoints
- `MAX_REQ_PER_IP` — per-IP daily rate cap
