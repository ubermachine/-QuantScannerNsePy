
## 2023-11-20 - DuckDB fetchnumpy() Optimization
**Learning:** `duckdb.fetchnumpy()` combined with Pandas datetime conversion is significantly faster than `fetchall()` with Python-level array creation when loading bulk data into numpy arrays. Using `np.split` based on boundary detection eliminates the need to group rows in a dictionary using a python loop, leading to >2x performance improvement for array creation.
**Action:** When extracting large amounts of grouped row data into vectorized Numpy arrays, prefer `fetchnumpy()` along with numpy array boundary splitting instead of iterating row-by-row in Python.
