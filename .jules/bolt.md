
## 2024-05-18 - NumPy Sliding Window View for Financial Indicators
**Learning:** In purely NumPy-based numerical applications like QuantScanner, writing explicit Python loops to slice arrays and call `.mean()` or `.std()` repeatedly (e.g. for Bollinger Bands) is a massive performance bottleneck due to Python overhead on each iteration.
**Action:** Always prefer `numpy.lib.stride_tricks.sliding_window_view(arr, period)` paired with vector operations along the last axis (`axis=-1`) to fully push rolling computations down to the C level, yielding ~100x+ speedups for larger arrays.
