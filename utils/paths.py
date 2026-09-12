"""Path constants shared across domains.

`METADATA_TSV` (the curated cpg0014 metadata) is imaging's primary metadata
table, but proteomics also reads it as a join-key bridge (see
`proteomics.imaging_metadata`), so it lives here rather than in
`imaging.paths`.
"""

from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGING_DATA_DIR = DATA_DIR / "imaging"

METADATA_TSV = IMAGING_DATA_DIR / "metadata_cpg0014.tsv"
