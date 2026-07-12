## 2026-07-12 - Vectorizing financial indicators
**Learning:** In a codebase heavily relying on NumPy arrays for calculations over large backtest data, using Python `for` loops combined with tiny slices incurs massive overhead due to Python-level dispatching, type-checking (`isinstance`, `hasattr`), and array allocation.
**Action:** Always prefer native vectorized NumPy operations like `np.lib.stride_tricks.sliding_window_view` for rolling windows, `np.convolve` for rolling sums/averages, and `np.cumsum`/`np.where` for sequential states to significantly boost performance.
