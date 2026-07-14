## 2023-10-25 - Python `for` Loop Overhead in NumPy Calculations
**Learning:** Calling `mean()` and `std()` in a Python `for` loop over NumPy array slices creates enormous overhead compared to vectorized equivalent. The `bollinger` function was a major bottleneck taking ~70% of execution time in `run_backtest`.
**Action:** Use vectorized rolling window operations like `numpy.lib.stride_tricks.sliding_window_view` to compute moving statistics over arrays which provided roughly an 80x speedup in this codebase's architecture without losing accuracy or readability.
