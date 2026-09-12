"""Central path constants for the proteomics normalize -> merge -> correct
pipeline. Mirrors `imaging/paths.py`'s layout."""

from pathlib import Path

PROTEOMICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTEOMICS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROTEOMICS_DATA_DIR = DATA_DIR / "proteomics"
INTERIM_DIR = PROTEOMICS_DATA_DIR / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "proteomics"

# The three nELISA export formats (see README/user description for why
# "normalized" is the default): raw concentration (heavy LOD-imputation
# artifacts), raw signal-to-noise (per-analyte dynamic range varies 30-300x),
# and the LLOD=0/ULOD=1 normalized signal used everywhere downstream here.
PGML_CSV = PROTEOMICS_DATA_DIR / "brd4_all-plates_pgml_sample.csv"
NELISA_SIGNAL_CSV = PROTEOMICS_DATA_DIR / "brd4_all-plates_nelisa-signal_sample.csv"
NORMALIZED_NELISA_CSV = PROTEOMICS_DATA_DIR / "brd4_all-plates_normalized-nelisa-signal_sample.csv"
RAW_PROTE_PATH = NORMALIZED_NELISA_CSV

NORMALIZED_INTERIM_PATH = INTERIM_DIR / "normalized.pkl"
RAW_INTERIM_PATH = INTERIM_DIR / "raw.pkl"

COPAIRS_RESULTS_DIR = RESULTS_DIR / "copairs"
# Null-distribution cache for copairs' permutation test, mirroring
# `imaging.paths.CACHE_DIR` -- separate from it since proteomics'
# (n_pos_pairs, n_total_pairs) configs don't line up with imaging's.
NULL_CACHE_DIR = PROTEOMICS_DIR / ".copairs_null_cache"


def corrected_interim_path(method: str, covariates: tuple) -> Path:
    """One cache file per (method, covariates) combination -- a single
    shared filename would silently return a stale, differently-corrected
    matrix after switching `method`/`covariates` between calls."""
    tag = f"{method}_{'-'.join(covariates)}"
    return INTERIM_DIR / f"corrected_{tag}.pkl"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COPAIRS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
