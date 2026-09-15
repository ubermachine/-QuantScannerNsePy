## 2025-02-18 - DuckDB Bulk Load Vectorisation
**Learning:** DuckDB's `fetchnumpy()` returns dates as `numpy.datetime64`, which breaks downstream Python code expecting standard `datetime` objects.
**Action:** Always cast the date column from DuckDB using `pandas.to_datetime().to_pydatetime()` when migrating from `fetchall()` to `fetchnumpy()` to maintain `datetime` interface compatibility.
