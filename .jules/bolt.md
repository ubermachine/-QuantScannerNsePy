## 2025-02-19 - Vectorizing financial indicators
**Learning:** Found significant bottlenecks in `bollinger` and `cmf` indicators in `indicators.py` because they used python `for` loops across large NumPy arrays to compute rolling means/std and rolling sums.
**Action:** Replaced for loops with `np.lib.stride_tricks.sliding_window_view` (for `bollinger`) and `np.convolve` with `np.divide` (for `cmf`). This dropped the execution time from ~7 seconds to ~0.06 seconds for `bollinger` on 3000 data points.
