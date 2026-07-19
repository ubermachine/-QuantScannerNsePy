
## 2026-07-19 - Vectorizing DuckDB fetchnumpy and CMF indicator
**Learning:** DuckDB's `fetchnumpy()` allows retrieving bulk dictionary of arrays much faster than using `.fetchall()` to parse millions of rows via Python loops. We can combine `fetchnumpy()` with `np.split` via boundaries for grouped operations. In addition, `cmf` can be heavily optimized by leveraging `numpy.lib.stride_tricks.sliding_window_view` to compute moving sums rather than python loops.
**Action:** Use `fetchnumpy()` + `np.split` to avoid O(N) python iteration loops over DB rows. Rely on `sliding_window_view` for sum based indicator functions instead of python loops.
