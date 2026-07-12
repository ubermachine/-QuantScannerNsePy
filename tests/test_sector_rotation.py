import pytest
from core import get_sector_rotation

def test_get_sector_rotation_structure():
    """Verify that get_sector_rotation returns updated nomenclature and benchmark data."""
    data = get_sector_rotation()
    
    # 1. Verify required keys exist in return dictionary
    expected_keys = [
        "sectors",
        "accelerating",
        "recovering",
        "decelerating",
        "underperforming",
        "rotation_signal",
        "benchmark_history"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from get_sector_rotation return dict"
        
    # 2. Verify sectors have correct quadrant names
    valid_quadrants = ["Accelerating", "Recovering", "Decelerating", "Underperforming"]
    for s in data["sectors"]:
        assert s["quadrant"] in valid_quadrants, f"Invalid quadrant name: {s['quadrant']}"
        
    # 3. Verify benchmark_history exists and contains daily price records
    benchmark = data["benchmark_history"]
    assert isinstance(benchmark, list)
    if benchmark:
        assert "date" in benchmark[0]
        assert "close" in benchmark[0]
        assert isinstance(benchmark[0]["close"], float)
