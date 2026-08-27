"""cpg0014 copairs-reproduction pipeline.

Convention: every module in this package is a library -- given inputs,
return outputs (DataFrames/arrays/dicts), no disk writes and no plotting of
its own, beyond an explicit, caller-given cache path (e.g. `cache_dir`/
`ap_cache_path` in `copairs_pipeline`). This applies to the analysis
modules (`copairs_pipeline`, `reversion`, `batch_report`) just as much as
the shared utilities (`load`, `features`, `plot`, `paths`): an analysis
module's `compute_*` functions load/transform/score and return results,
they don't decide what gets persisted.

Deciding what to persist -- which parquet/JSON files to write, which
`imaging.plot` figures to render and save -- is the job of the top-level
driver scripts (`run_pipeline.py`, `run_reversion.py`), not anything in
this package. Keeping that line intact is what lets the analysis functions
be called and tested without touching disk.
"""
