# Deploying to the HF Space

This directory holds the Space-specific deployment artefacts. The Space
repo (`sets-sto/warp-backend` on HuggingFace) is a deployment mirror of
the runtime files in this GitHub repo, *not* a development checkout.

## Layout

The Space repo root contains:

```
README.md          ← copied from space/README.md (has the HF YAML
                     frontmatter the Space runtime needs)
Dockerfile         ← copied from space/Dockerfile
main.py            ← copied from project root
requirements.txt   ← copied from project root
```

Admin/training scripts (`admin_*.py`, `democratic_merge_crops.py`,
`hf_clone.py`, `setup.py`, etc.) stay on GitHub and **are not pushed to
the Space** — they aren't needed at runtime and shipping them would
just enlarge the image and the attack surface.

## Automatic deploy (default)

A push to `main` that touches a runtime file (`main.py`, `requirements.txt`,
`space/Dockerfile`, `space/README.md`, `deploy_space.py`) triggers
`.github/workflows/deploy_space.yml`, which runs `deploy_space.py` to upload
all four files to the Space in **one commit** → one rebuild. Nothing to do by
hand — just push. Uses the existing `HF_TOKEN` Actions secret (needs Space
write scope; if the deploy step 403s, widen that token's scope).

## Manual deploy (fallback)

Run the same script locally with a write token:

```sh
HF_TOKEN=hf_xxx python deploy_space.py            # upload + rebuild
HF_TOKEN=hf_xxx python deploy_space.py --dry-run  # show the file plan only
```

After the build completes (watch the "Logs" tab in the Space UI), check:

```sh
curl https://sets-sto-warp-backend.hf.space/health
# → {"status":"ok","repo":"sets-sto/warp-knowledge"}
```

## Required env (set in Space Settings → Variables and secrets BEFORE pushing)

Both `Variables` and `Secrets` end up as `os.environ[...]` at runtime,
but Secrets are write-only after save (Settings UI shows a placeholder
only). Use Secret for anything that grants access; Variable for plain
config the public could read anyway.

### Secrets (sensitive)

| name         | source / value                                                                                                                                                                                                |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HF_TOKEN`   | fine-grained write token to **both** `datasets/sets-sto/warp-knowledge` (current `/contribute`) **and** `datasets/sets-sto/sto-icon-dataset` (Phase 1 bulk endpoints). Without the second scope, bulk uploads will 401. |
| `ADMIN_KEY`  | copy from current Render env (guards `/admin/*`)                                                                                                                                                              |

### Variables (public config)

| name             | value                              |
|------------------|------------------------------------|
| `HF_REPO_ID`     | `sets-sto/warp-knowledge`          |
| `MAX_REQ_PER_IP` | `500` (or match current Render)    |

Without `HF_TOKEN` the Space will boot but every `/contribute` will
return HTTP 503 with `"Storage unavailable, please try later"` — the
same failure mode that surfaced on Render when its token was wrong.
