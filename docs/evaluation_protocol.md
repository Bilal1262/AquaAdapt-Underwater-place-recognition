# Place-recognition evaluation protocol

The test split is subsampled into interleaved query and database sequences. For each
query, the evaluator removes itself and database frames inside the temporal exclusion
window. A remaining database image is a positive when its TUM translation is within the
configured radius (1.5 m by default).

Only queries with at least one remaining pose positive are eligible. The report separates
total, eligible, and ineligible queries and gives coverage. Recall@K, MRR, first-positive
rank, and top-1 translation errors use eligible queries only.

Robustness keeps the database clean, corrupts only queries, and preserves identical
selection across methods and severities. Severity zero is the clean control. A fixed seed
controls random haze structure, noise, and marine snow.

The normalized-progress fallback is disabled by default. Results from such a fallback
must be labeled `APPROXIMATE PROGRESS-BASED EVALUATION`; the current pose implementation
does not silently activate it.

