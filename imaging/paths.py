"""Central path constants for the imaging/ copairs-reproduction pipeline."""

from pathlib import Path

IMAGING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = IMAGING_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGING_DATA_DIR = DATA_DIR / "imaging"
RESULTS_DIR = PROJECT_ROOT / "results" / "imaging" / "copairs"
BATCH_REPORT_DIR = PROJECT_ROOT / "results" / "imaging" / "batch_correction_report"
CACHE_DIR = IMAGING_DIR / ".copairs_null_cache"

METADATA_TSV = IMAGING_DATA_DIR / "metadata_cpg0014.tsv"
BATCH_OUTLIERS_TSV = IMAGING_DATA_DIR / "batch_outliers.tsv"
CELLPROFILER_PARQUET = IMAGING_DATA_DIR / "augmented_combined.parquet"
CPCNN_TSV_GZ = IMAGING_DATA_DIR / "cpcnn_avg_cellCounWeigh.tsv.gz"
UNIDINO_PKL = IMAGING_DATA_DIR / "unidino_data_w_batch5_well_avg.pkl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
