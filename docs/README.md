# sets-warp-backend documentation

**Start here:** [`technical_overview.md`](technical_overview.md) — components,
endpoints, mergers, trainers, and how they fit together.

| Doc | Scope |
|---|---|
| [`technical_overview.md`](technical_overview.md) | Component map, endpoint table, rate limits, the four mergers, the audit safety net |
| [`DATA_LIFECYCLE.md`](DATA_LIFECYCLE.md) | One crop's journey: upload → staging → vote → `data/` → training → client |
| [`INGESTION_VALIDATION.md`](INGESTION_VALIDATION.md) | The upload whitelist: what it gates, how it fails open, and the three lists a screen type must clear to reach the model |
| [`user_guide.md`](user_guide.md) | Operator guide — running the service, the admin scripts, the workflows |

Client-side documentation (recognition pipeline, trainer UI, sync) lives in the
`sto-warp` repository, indexed in its `docs/README.md`.
