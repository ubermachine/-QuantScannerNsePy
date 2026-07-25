## 2024-07-25 - Python Loops in NumPy indicator functions are slow
**Learning:** In heavily used financial indicator functions like `bollinger` or `cmf`, explicit Python `for` loops iterating over time series arrays cause severe performance bottlenecks (1.14s down to 0.07s for 50k rows in bollinger).
**Action:** When working on array computations, aggressively replace Python `for` loops with vectorized methods like `numpy.lib.stride_tricks.sliding_window_view` for rolling windows (means/stds) and `np.convolve` for rolling sums.
