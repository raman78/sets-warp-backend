"""
hf_commit.py — validate a commit before sending it, and send it in pieces
=========================================================================

Every tool here builds one unbounded list of operations and makes one
`create_commit`. That worked while the volumes were small. On 2026-07-16 the
crop merge crossed whatever size the endpoint accepts and started answering

    400 Bad Request  (no message body, only a Request ID)

on every scheduled run. 2281 crops sat un-promoted for seven weeks, and the
workflow was green throughout because the step piped its output through
`tee`. The merge itself never changed; only its size did.

Two things follow, and this module is both of them:

`validate_ops`
    An opaque 400 is a debugging dead end. Check the conditions that produce
    one and that are visible from this side — duplicate paths, empty paths,
    malformed paths — and refuse to send, naming what is wrong. A named
    refusal costs one run; an unnamed 400 cost seven weeks.

`commit_chunked`
    Send at most `CHUNK` operations per commit. The number is not from a
    spec: HF does not document the limit and did not name it in the failure.
    It is well under every commit size that has ever succeeded in this repo.

Atomicity: chunking gives it up, and that is the intended trade. The commit
this replaced could not be applied at all at production size, so the choice
was never between atomic and staged — it was between staged and nothing.
Callers order their stages so that every intermediate state is consistent:
additions first, the index or manifest that references them second,
deletions last. An interrupted run then leaves duplicates to re-do, never a
reference to something missing.
"""

from __future__ import annotations

CHUNK = 256


def validate_ops(ops: list) -> list[str]:
    """Return a list of reasons this operation set would be refused.

    Empty list means nothing detectable is wrong — not that the commit will
    succeed. The server has the last word; this only catches the faults we
    can see without asking it.
    """
    problems: list[str] = []
    seen: dict[str, int] = {}
    for op in ops:
        path = getattr(op, 'path_in_repo', '')
        if not path:
            problems.append(f'operation with an empty path: {op!r}')
            continue
        if path.startswith('/') or '//' in path or path.endswith('/'):
            problems.append(f'malformed path: {path!r}')
        seen[path] = seen.get(path, 0) + 1
    for path, n in sorted(seen.items()):
        if n > 1:
            problems.append(f'{n} operations target the same path: {path!r}')
    return problems


def commit_chunked(api, repo_id: str, repo_type: str, ops: list,
                   message: str, chunk: int = CHUNK,
                   label: str = '', validate: bool = True) -> int:
    """Commit `ops` in batches of at most `chunk`. Returns the commit count.

    Raises SystemExit before writing anything when `validate_ops` objects, so
    a malformed set never reaches the endpoint and the log says why.
    """
    if validate:
        problems = validate_ops(ops)
        if problems:
            for p in problems[:20]:
                print(f'  MALFORMED: {p}')
            if len(problems) > 20:
                print(f'  … and {len(problems) - 20} more')
            raise SystemExit(
                f'Refusing to commit: {len(problems)} malformed operation(s). '
                f'Nothing was written.')
    if not ops:
        return 0

    total = (len(ops) + chunk - 1) // chunk
    tag = f'{label} ' if label else ''
    for i in range(0, len(ops), chunk):
        part = ops[i:i + chunk]
        n = i // chunk + 1
        if total > 1:
            print(f'  commit {tag}{n}/{total} ({len(part)} ops)…')
        api.create_commit(
            repo_id        = repo_id,
            repo_type      = repo_type,
            operations     = part,
            commit_message = message if total == 1 else f'{message} [{tag}{n}/{total}]',
        )
    return total


def commit_adds_then_deletes(api, repo_id: str, repo_type: str, ops: list,
                             message: str, chunk: int = CHUNK) -> int:
    """Commit `ops` in chunks, every addition before any deletion.

    The safe order for a set that can no longer go in one commit. An addition
    that lands without its deletion leaves a redundant file — inert, and the
    next run removes it. A deletion that lands without its addition removes
    the only copy of something. So additions go first regardless of the order
    the caller built them in.

    A path that is both added and deleted would be mis-ordered by this rule,
    so it is not allowed: `validate_ops` refuses two operations on one path,
    and that refusal happens before anything is written.
    """
    problems = validate_ops(ops)
    if problems:
        for p in problems[:20]:
            print(f'  MALFORMED: {p}')
        raise SystemExit(
            f'Refusing to commit: {len(problems)} malformed operation(s). '
            f'Nothing was written.')

    adds    = [o for o in ops if type(o).__name__ != 'CommitOperationDelete']
    deletes = [o for o in ops if type(o).__name__ == 'CommitOperationDelete']
    n  = commit_chunked(api, repo_id, repo_type, adds, message,
                        chunk=chunk, label='add', validate=False)
    n += commit_chunked(api, repo_id, repo_type, deletes, message,
                        chunk=chunk, label='delete', validate=False)
    return n
