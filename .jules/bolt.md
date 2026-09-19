
## 2024-05-18 - DuckDB Bulk Fetch Array Vectorization
**Learning:** In DuckDB for Python, extracting bulk row groupings into NumPy arrays via `fetchnumpy()` combined with vectorized string array boundary detection (`np.where(arr[:-1] != arr[1:])[0] + 1`) and `np.split()` is nearly 2x faster than using standard Python `fetchall()` and loop iteration over rows to build array groups, mitigating high iteration overhead in core scanner tasks.
**Action:** Always prefer `fetchnumpy` + vectorized numpy logic for database loads when standard groupings are required.

## 2024-05-18 - DuckDB fetchnumpy and pandas interactions
**Learning:** DuckDB's `fetchnumpy()` returns dates as `numpy.datetime64`. To maintain compatibility with downstream codebase functions expecting standard Python `datetime` objects (which rely on attributes like `.year` and `.isoformat()`), using `pandas.to_datetime().to_pydatetime()` is a necessary and highly performant way to convert the date arrays, much faster than python list comprehensions.
**Action:** When using `fetchnumpy()` and downstream code requires Python `datetime` objects, `pandas.to_datetime(array).to_pydatetime()` is the fastest safe method, provided pandas is imported.
