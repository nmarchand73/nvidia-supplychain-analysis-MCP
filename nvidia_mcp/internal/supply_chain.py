"""
Internal wrapper for supply chain analysis functions
Imports from scripts/fetch_supply_chain_data.py
"""
import sys
from pathlib import Path

# Add scripts directory to path to import from scripts
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from fetch_supply_chain_data import (
    fetch_stock_data,
    calculate_correlations,
    analyze_supply_chain_impact,
    NVIDIA_ECOSYSTEM
)

__all__ = [
    'fetch_stock_data',
    'calculate_correlations',
    'analyze_supply_chain_impact',
    'NVIDIA_ECOSYSTEM'
]

