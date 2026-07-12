import os
import pytest
import duckdb
import pandas as pd

# Import DB path and _ensure_db from core.py
# We will temporarily mock the DB path in tests to avoid overwriting production DB
import core

def test_ensure_db_from_lake(tmp_path):
    """Verify that _ensure_db loads the Parquet tables from the centralized data lake."""
    test_db = os.path.join(tmp_path, "test_quantscanner.duckdb")
    
    # Backup original DB settings
    original_db = core.DB
    core.DB = test_db
    
    # We will also create a mock local lake in tmp_path/market-data-lake/data/
    lake_dir = os.path.join(tmp_path, "market-data-lake", "data")
    os.makedirs(lake_dir, exist_ok=True)
    
    # Create mock parquet files with schemas matching core tables
    # 1. DailyBars
    df_daily = pd.DataFrame({
        "Ticker": ["TCS.NS", "RELIANCE.NS"],
        "Date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "Open": [3000.0, 2400.0],
        "High": [3050.0, 2420.0],
        "Low": [2980.0, 2380.0],
        "Close": [3020.0, 2410.0],
        "Volume": [100000, 200000]
    })
    # Convert date to datetime for parquet
    df_daily.to_parquet(os.path.join(lake_dir, "DailyBars.parquet"), index=False)
    
    # 2. WeeklyBars
    df_weekly = df_daily.copy()
    df_weekly.to_parquet(os.path.join(lake_dir, "WeeklyBars.parquet"), index=False)
    
    # 3. SectorDailyBars
    df_sector = df_daily.copy()
    df_sector.to_parquet(os.path.join(lake_dir, "SectorDailyBars.parquet"), index=False)
    
    # 4. StockMetadatas
    df_meta = pd.DataFrame({
        "Ticker": ["TCS.NS", "RELIANCE.NS"],
        "Name": ["Tata Consultancy Services", "Reliance Industries"],
        "Sector": ["IT", "Energy"]
    })
    df_meta.to_parquet(os.path.join(lake_dir, "StockMetadatas.parquet"), index=False)
    
    # Mock core._get_parquet_path to return these local paths
    def mock_get_parquet_path(table_name):
        return os.path.join(lake_dir, f"{table_name}.parquet").replace("\\", "/")
        
    original_get_path = getattr(core, "_get_parquet_path", None)
    core._get_parquet_path = mock_get_parquet_path
    
    try:
        # Call implementation
        core._ensure_db()
        
        # Verify the database has the tables loaded
        assert os.path.exists(test_db)
        con = duckdb.connect(test_db)
        
        # Verify tables exist and have data
        res = con.execute("SELECT count(*) FROM DailyBars").fetchone()[0]
        assert res == 2
        
        res_meta = con.execute("SELECT Sector FROM StockMetadatas WHERE Ticker = 'TCS.NS'").fetchone()[0]
        assert res_meta == "IT"
        
        con.close()
    finally:
        # Restore original settings
        core.DB = original_db
        if original_get_path is not None:
            core._get_parquet_path = original_get_path
