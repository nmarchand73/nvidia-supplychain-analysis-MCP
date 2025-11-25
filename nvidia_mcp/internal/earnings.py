"""
Internal wrapper for earnings calendar functions
Imports from scripts/earnings_calendar.py
"""
import sys
from pathlib import Path

# Add scripts directory to path to import from scripts
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from earnings_calendar import (
    get_earnings_calendar,
    analyze_earnings_impact,
    identify_catalyst_clusters,
    get_insider_sentiment,
    compile_event_calendar
)

__all__ = [
    'get_earnings_calendar',
    'analyze_earnings_impact',
    'identify_catalyst_clusters',
    'get_insider_sentiment',
    'compile_event_calendar'
]

