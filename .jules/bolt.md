## 2025-01-20 - Vectorizing trailing rolling metrics
**Learning:** Python `for` loops computing trailing metrics (e.g. `mean`, `std`, `sum`) block by block cause huge performance bottlenecks in this codebase. A backtest loops over arrays of data multiple times, slowing down processing dramatically.
**Action:** Always prefer `numpy.lib.stride_tricks.sliding_window_view` over `for` loops in trailing computations (e.g., Bollinger bands, CMF) to leverage fully C-optimized array broadcasts and reductions.
