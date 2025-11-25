"""
Internal wrapper for technical analysis functions
Imports from scripts/technical_analysis.py
"""
import sys
from pathlib import Path

# Add scripts directory to path to import from scripts
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from technical_analysis import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    identify_support_resistance,
    analyze_volume_profile,
    detect_patterns,
    generate_signals,
    scan_supply_chain
)

__all__ = [
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'identify_support_resistance',
    'analyze_volume_profile',
    'detect_patterns',
    'generate_signals',
    'scan_supply_chain'
]

