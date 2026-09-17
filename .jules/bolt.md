## 2024-11-20 - Vectorized Python loops for Rolling Window arrays
**Learning:** Python loops over numpy arrays are a severe bottleneck for rolling indicator functions. Vectorization using `numpy.lib.stride_tricks.sliding_window_view` can provide an order-of-magnitude performance improvement.
**Action:** When creating new indicators, strongly prefer vectorized window operations (such as `sliding_window_view` combined with `.mean(axis=-1)`) over loops, especially for array inputs.
