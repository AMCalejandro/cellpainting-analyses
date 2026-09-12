"""Domain-agnostic library shared by `imaging` and `proteomics`.

Neither domain package may import from the other; anything both need lives
here instead. Every module in this package operates on generic data shapes
(feature matrices, `Metadata_*`-convention DataFrames) with no logic
hardcoded to one domain.
"""
