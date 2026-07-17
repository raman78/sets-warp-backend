"""deploy_space.py — push the runtime files to the HF Space.

The live backend runs as a HuggingFace Space (`sets-sto/warp-backend`,
public URL `sets-sto-warp-backend.hf.space`). The Space is a *deployment
mirror*: only the four runtime files below live there, copied from this
GitHub repo. Uploading them as a single commit triggers exactly one Space
rebuild.

This replaces the manual clone/copy/push dance in `space/README_deploy.md`
and is what `.github/workflows/deploy_space.yml` runs on every push to
`main` that touches a runtime file — so "push fix to GitHub" now redeploys
the Space automatically.

Auth: reads the HF write token from `$HF_TOKEN` (never hard-coded). The
token must have write access to the Space repo; the dataset-scoped token
already used by the merge workflows works only if it also covers the Space.

Usage::

    HF_TOKEN=hf_xxx python deploy_space.py            # deploy
    HF_TOKEN=hf_xxx python deploy_space.py --dry-run  # show plan, upload nothing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# {source in this repo → path in the Space repo}. Mirrors space/README_deploy.md.
RUNTIME_FILES: dict[str, str] = {
    'space/Dockerfile':   'Dockerfile',
    'space/README.md':    'README.md',
    'main.py':            'main.py',
    'requirements.txt':   'requirements.txt',
}

SPACE_REPO = os.environ.get('HF_SPACE_REPO_ID', 'sets-sto/warp-backend')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Deploy runtime files to the HF Space.')
    p.add_argument('--dry-run', action='store_true',
                   help='print the file plan and exit without uploading')
    p.add_argument('--repo', default=SPACE_REPO,
                   help=f'target Space repo id (default: {SPACE_REPO})')
    args = p.parse_args(argv)

    # Resolve + verify every source file before touching the network.
    ops_src: list[tuple[Path, str]] = []
    missing: list[str] = []
    for src, dst in RUNTIME_FILES.items():
        sp = ROOT / src
        if sp.is_file():
            ops_src.append((sp, dst))
        else:
            missing.append(src)
    if missing:
        print(f'ERROR: missing runtime file(s): {", ".join(missing)}', file=sys.stderr)
        return 2

    print(f'Deploy target: spaces/{args.repo}')
    for sp, dst in ops_src:
        print(f'  {sp.relative_to(ROOT)}  →  {dst}')

    if args.dry_run:
        print('Dry-run — nothing uploaded.')
        return 0

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('ERROR: HF_TOKEN is not set', file=sys.stderr)
        return 2

    from huggingface_hub import CommitOperationAdd, HfApi

    sha = os.environ.get('GITHUB_SHA', '')
    msg = f'Deploy runtime files{f" ({sha[:10]})" if sha else ""}'

    operations = [
        CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(sp))
        for sp, dst in ops_src
    ]
    HfApi(token=token).create_commit(
        repo_id=args.repo,
        repo_type='space',
        operations=operations,
        commit_message=msg,
    )
    print(f'Uploaded {len(operations)} file(s) in one commit — Space is rebuilding.')
    print(f'Watch: https://huggingface.co/spaces/{args.repo}  '
          f'| verify: curl https://sets-sto-warp-backend.hf.space/health')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
