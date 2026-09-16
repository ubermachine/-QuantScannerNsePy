## 2026-09-16 - Vectorize cmf and bollinger indicators
**Learning:** cmf and bollinger indicators used for loops for simple sliding window computations resulting in performance bottlenecks.
**Action:** Replaced for loop with np.convolve in cmf, and with sliding_window_view in bollinger to vectorize the computations.
