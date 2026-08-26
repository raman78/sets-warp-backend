# Ingestion validation

Every client upload passes a whitelist before it reaches HuggingFace. This
document covers what that whitelist gates, where it comes from, how it fails,
and the three-list chain a new screen type must clear before the classifier
can learn it.

Scope: the whitelist only. Regex-level field validation (`install_id`,
coordinate ranges, aspect bounds) is listed in
[`technical_overview.md` §2](technical_overview.md#validation). The staging →
`data/` promotion the whitelist feeds is in
[`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md).

---

## Source of truth

`config/labels.json` — three keys:

| Key | Type | Meaning |
|---|---|---|
| `screen_types` | `list[str]` | Screen types accepted for upload |
| `slots` | `dict[str, list[str]]` | Per **screen type**, the slot names it may carry |
| `anchor_build_types` | `dict[str, list[str]]` | Per **build type**, the screen types it is folded from — and whose slot lists it inherits |

Resolution order, in `_load_labels_from_hf` (`main.py:945`):

```
HF dataset config/labels.json  ──found──►  use it
        │
        └── missing / unreadable / no screen_types[] ──►  bundled copy
                                                          (main.py:929)
```

The HF copy is the documented runtime override so the whitelist can be widened
without a redeploy. It has never been seeded — `sets-sto/sto-icon-dataset`
answers `Entry not found` for `config/labels.json` — so in practice the bundled
file is what runs. Results are cached for `LABELS_CACHE_TTL = 300` s
(`main.py:154`).

### Two vocabularies, and why the third key exists

An anchor grid is keyed by the client's *build type*, not by a screen type.
WARP CORE folds one into the other before it writes `anchors.json`
(`trainer_window._STYPE_TO_BUILD`):

| Screen type | Build type |
|---|---|
| `SPACE_EQ`, `SPACE_MIXED` | `SPACE` |
| `GROUND_EQ`, `GROUND_MIXED` | `GROUND` |
| `TRAITS` | `SPACE_TRAITS` |
| `BOFFS` | `BOFFS` |
| `SPECIALIZATIONS` | `SPEC` |

The anchor gate used to validate `build_type` against `screen_types[]`, which
made `SPACE` and `GROUND` unrepresentable: every such grid was refused with
`build_type 'SPACE' not in whitelist` and re-offered on the client's next sync,
for good. `BOFFS` and `SPACE_TRAITS` passed only because the two vocabularies
spell those the same. Measured against one maintainer's store: 112 of 176
grids refused on the build type, another 11 on slots — a `TRAITS` screen mixes
both environments, so a `SPACE_TRAITS` grid legitimately carries ground trait
slots that `slots['SPACE_TRAITS']` does not list.

`anchor_build_types` states the fold, so slot lists stay declared once, per
screen type, and a build type inherits the union of its sources'. Both
problems close: `SPACE_TRAITS → ['TRAITS']` inherits the mixed list.

### Marker-keyed BOFF seats

A slot name like `Boff Seat R[E+O]_484` carries what the detector saw, down to
the marker's Y in pixels, so it cannot appear in any whitelist. The shape is
fixed and that is what `_ANCHOR_SEAT_RE` checks —
`Boff Seat <L|R>[<code>]_<n>`, with the seat code optional for the legacy form
(`warp/recognition/boff_keys.py` in sto-warp). Anything else shaped like a
seat key but not matching, `Boff Seat X[T]_1` for instance, is still refused.

## What each gate rejects

| Endpoint | Field | Gate | Line |
|---|---|---|---|
| `POST /upload/screen-types` | `screen_type` | must be in `screen_types[]` | `main.py:497` |
| `POST /upload/anchors` | `grid.build_type` | must be a key of `anchor_build_types` | `main.py:595` |
| `POST /upload/anchors` | `grid.slots` keys | must be in the inherited slot union, or match `_ANCHOR_SEAT_RE` | `main.py:608` |

Anchor slot checking distinguishes two cases that used to look the same
(`_anchor_whitelist`, `main.py:992`):

| Inherited slot set | Meaning | Effect |
|---|---|---|
| build type absent from the map | not described | build type refused |
| declared, empty | no icon slots on any source screen | **every** grid for it is refused |
| declared, non-empty | known slot set | stray slot names refused |
| map itself empty | bundled + HF both unusable | no enforcement (fail-open) |

A build type is only as good as the screen types it inherits from: mapping one
to `DISCARD` or a skill tree would give it an empty set and refuse every grid,
which is correct — those screens carry no icon slots.

An HF `labels.json` predating `anchor_build_types` falls back to the **bundled**
map rather than switching the gate off, because the bundled copy ships with the
code that reads it. Only a total failure of both fails open.

## Fail-open, and how it rotted

An empty `screen_types[]` disables every gate above — each is written as
`if allowed_… and value not in allowed_…`. That is deliberate: a transient HF
outage plus a bundled-load failure must not black-hole uploads.

The trap is that nothing distinguished "transient" from "the file was never
deployed". Until 2026-08-14 the Space ran with no whitelist at all:

```
GET /config/labels  →  {"schema_version":1,"screen_types":[],"slots":{}}
```

Any `screen_type` matching `^[a-zA-Z0-9_-]{1,40}$`, any anchor `build_type`
and any slot name was accepted. Three links in the chain each have to hold,
and only the last one was verified when the feature was built:

```
repo config/labels.json
   │  deploy_space.RUNTIME_FILES        (deploy_space.py:33)
   ▼
Space repo config/labels.json
   │  COPY --chown=user config/ ./config/   (space/Dockerfile:24)
   ▼
container /home/user/app/config/labels.json
   │  _load_labels_bundled()            (main.py:929)
   ▼
enforcing
```

The file was in the repo but not in `RUNTIME_FILES`, so it never reached the
Space; once it did, the Dockerfile copied only `main.py`, so it never reached
the image. Uploading and deploying are separate failures with the same
symptom.

## Health signal

`GET /health` (`main.py:250`) reports which state the backend is in:

```json
{"status":"ok","validation":"enforcing","screen_types":15}
{"status":"ok","validation":"DISABLED (empty whitelist)","screen_types":0}
```

This exists because a backend running wide open is otherwise indistinguishable
from a healthy one — the endpoints answer `200`, uploads succeed, nothing in
the logs looks wrong. `_load_labels_bundled` also logs the failure at `ERROR`
naming the consequence, not at `WARNING`.

Both were added in the same commit as the deploy fix, and immediately earned
it: the first deploy still reported `DISABLED`, which is how the missing
Dockerfile `COPY` was found.

## A screen type's path to the model

Three lists must agree before a label can become a classifier class. They are
deliberately not the same list — the first two decide what may be *stored*, the
third what is *trained*.

| # | List | File | Rejects by |
|---|---|---|---|
| 1 | `screen_types[]` | `config/labels.json` | HTTP 400 at upload |
| 2 | `SCREEN_TYPES` | `democratic_merge_screens.py:95` | skipped at staging → `data/` (`:221`) |
| 3 | `SCREEN_TYPES` | `admin_train.py:90` | ignored when reading `data/screen_types` (`:479`) |

List 3 is narrower on purpose: `SPACE_TRAITS` / `GROUND_TRAITS` are stored but
train as `TRAITS`, and the skill-tree variants follow the same pattern — one
class per visually distinct screen, not one per label.

A class that clears all three still waits for data: `SC_MIN_CLASS_SAMPLES = 5`
(`admin_train.py:82`) drops any class with fewer than five samples, so the
model is unchanged until the community has contributed enough.

`UNKNOWN` is intentionally absent from list 1. It is the client's
not-yet-classified sentinel, not a label; sto-warp skips it at upload rather
than posting a batch that would 400 (`warp/trainer/sync.py`, screen-type
upload loop).

### Why DISCARD matters

Without a `DISCARD` class the screen classifier has to force every screenshot
into one of the build types. A doff roster in the test corpus lands in
`TRAITS`, which then runs ship-info extraction over a screen that has no ship
on it. Giving the model a class for "no build content here" is what lets it
decline instead of guessing — but only once samples exist, per the threshold
above.

## Failure modes

| Symptom | Cause | First check |
|---|---|---|
| `/health` says `DISABLED (empty whitelist)` | whitelist never reached the container | the three links in [Fail-open](#fail-open-and-how-it-rotted), bottom-up |
| Client logs `screen-types backend rejected (HTTP 400)` | label outside `screen_types[]` | is it a real label or the `UNKNOWN` sentinel? |
| Anchors rejected as `has no icon slots` | grid sent for a slotless screen | client-side layout detection produced a grid it should not have |
| A label uploads fine but never appears in `data/screen_types/` | list 2 | `democratic_merge_screens.py:95` |
| A class exists in `data/` but not in the model | list 3, or the sample threshold | `admin_train.py:90`, then `SC_MIN_CLASS_SAMPLES` |

## Operational knobs

| Knob | Default | Effect |
|---|---|---|
| `HF_ICONS_REPO_ID` | `sets-sto/sto-icon-dataset` (`main.py:74`) | Empty string skips the HF override and uses the bundled file only |
| `LABELS_CACHE_TTL` | 300 s (`main.py:154`) | How long a whitelist change takes to take effect without a restart |
| `SC_MIN_CLASS_SAMPLES` | 5 (`admin_train.py:82`) | Samples a new class needs before it joins the model |

## Open questions

1. **The HF override has never been used.** Seeding
   `sets-sto/sto-icon-dataset:config/labels.json` would allow widening the
   whitelist without a deploy, but it also adds a second source of truth that
   can silently diverge from the repo. Decide whether to seed it or drop the
   override and read the bundled file only.
2. **Rejections are invisible in aggregate.** A client that keeps sending a
   refused label retries forever and nothing on this side counts it. A
   per-reason counter — on `/health` or in the monthly audit — would surface a
   client/whitelist mismatch instead of leaving it in one user's log.
