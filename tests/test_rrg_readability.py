import pytest
from core import get_sector_rotation

def test_sector_direction_calculation():
    """Verify that get_sector_rotation includes a valid direction arrow (↗, ↘, ↙, ↖) for each sector."""
    data = get_sector_rotation()
    
    assert "sectors" in data
    assert len(data["sectors"]) > 0
    
    valid_arrows = ["↗", "↘", "↙", "↖"]
    for s in data["sectors"]:
        assert "direction" in s, f"Sector {s['name']} missing 'direction' key"
        assert s["direction"] in valid_arrows, f"Invalid direction arrow '{s['direction']}' for sector {s['name']}"
