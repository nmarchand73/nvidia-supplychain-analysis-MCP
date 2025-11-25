#!/usr/bin/env python3
"""
MCP Server for Nvidia Supply Chain Analysis
Main server entry point using FastMCP
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import FastMCP from installed mcp package
# Use importlib to avoid conflicts with local folder name
import importlib.util
spec = importlib.util.find_spec("mcp.server.fastmcp")
if spec is None:
    raise ImportError("mcp.server.fastmcp not found. Please install mcp package: pip install mcp")

# Import FastMCP by loading the module directly
fastmcp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fastmcp_module)
FastMCP = fastmcp_module.FastMCP

# Import local tools
# Use absolute import when running as script, relative when as module
try:
    from .tools import (
        analyze_supply_chain as analyze_supply_chain_func,
        get_supply_chain_data as get_supply_chain_data_func,
        get_earnings_calendar as get_earnings_calendar_func,
        get_trading_instructions as get_trading_instructions_func,
        get_portfolio_execution_guide as get_portfolio_execution_guide_func
    )
except ImportError:
    # Fallback for running as script
    from nvidia_mcp.tools import (
        analyze_supply_chain as analyze_supply_chain_func,
        get_supply_chain_data as get_supply_chain_data_func,
        get_earnings_calendar as get_earnings_calendar_func,
        get_trading_instructions as get_trading_instructions_func,
        get_portfolio_execution_guide as get_portfolio_execution_guide_func
    )

# Create FastMCP server instance
app = FastMCP("nvidia-supply-chain-analysis")


@app.tool()
def analyze_supply_chain(
    analysis_type: str,
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    reference_ticker: str = "NVDA"
) -> str:
    """
    Main entry point for supply chain analysis. Supports multiple analysis types: 
    health, technical, earnings, bottlenecks, pairs, portfolio, full
    
    Args:
        analysis_type: Type of analysis (health, technical, earnings, bottlenecks, pairs, portfolio, full)
        tickers: Optional list of tickers to analyze (defaults to full supply chain)
        period: Time period for analysis (default: "3mo")
        reference_ticker: Reference ticker for correlations (default: "NVDA")
    
    Returns:
        JSON string with analysis results
    """
    result = analyze_supply_chain_func(
        analysis_type=analysis_type,
        tickers=tickers,
        period=period,
        reference_ticker=reference_ticker
    )
    return json.dumps(result, indent=2, default=str)


@app.tool()
def get_supply_chain_data(
    tickers: List[str],
    period: str = "3mo"
) -> str:
    """
    Fetch raw stock data for specific ticker(s) with optional period
    
    Args:
        tickers: List of stock tickers to fetch
        period: Time period (default: "3mo")
    
    Returns:
        JSON string with stock data
    """
    result = get_supply_chain_data_func(tickers=tickers, period=period)
    return json.dumps(result, indent=2, default=str)


@app.tool()
def get_earnings_calendar(
    tickers: Optional[List[str]] = None
) -> str:
    """
    Get upcoming earnings dates and events for supply chain tickers
    
    Args:
        tickers: Optional list of tickers (defaults to full supply chain)
    
    Returns:
        JSON string with earnings calendar data
    """
    result = get_earnings_calendar_func(tickers=tickers)
    return json.dumps(result, indent=2, default=str)


@app.tool()
def get_trading_instructions(
    action: str = "both",
    order_type: Optional[str] = None,
    include_fractional: bool = True
) -> str:
    """
    Get step-by-step instructions for buying/selling stocks on Interactive Brokers Mobile App.
    Based on official IBKR Mobile app documentation and best practices.
    
    Args:
        action: "buy", "sell", or "both" (default: "both") - which instructions to include
        order_type: Specific order type to focus on (Market, Limit, Stop, Stop Limit) or None for all types
        include_fractional: Whether to include fractional share trading setup instructions (default: True)
    
    Returns:
        JSON string with detailed trading instructions including:
        - Setup steps (app download, fractional shares enablement)
        - Buy order instructions (multiple methods)
        - Sell order instructions
        - Order configuration details (quantity, order type, price, time-in-force)
        - Order management (view, modify, cancel)
        - Order type explanations and best practices
        - Tips and risk management advice
    """
    result = get_trading_instructions_func(
        action=action,
        order_type=order_type,
        include_fractional=include_fractional
    )
    return json.dumps(result, indent=2, default=str)


@app.tool()
def get_portfolio_execution_guide(
    portfolio_allocation: dict,
    total_amount: float,
    current_prices: dict = None
) -> str:
    """
    Generate step-by-step execution guide for implementing a portfolio allocation on IBKR Mobile App.
    Converts portfolio percentages into actionable trading instructions with order priorities.
    
    Args:
        portfolio_allocation: Dictionary mapping tickers to percentage allocations
            Example: {"NVDA": 30.0, "TSM": 30.0, "LRCX": 10.0, "AMAT": 10.0, "MU": 10.0, "ASX": 5.0, "ASML": 5.0}
        total_amount: Total portfolio value in base currency (e.g., 500 for 500€)
        current_prices: Optional dictionary of current stock prices for accurate share calculations
            Example: {"NVDA": 177.17, "TSM": 284.40, "LRCX": 152.49}
    
    Returns:
        JSON string with execution plan including:
        - Allocation breakdown with dollar amounts and estimated shares
        - Step-by-step execution instructions
        - Order priority recommendations (largest positions first)
        - Fractional share requirements
        - Order type recommendations based on position size
    """
    # Handle current_prices - convert to Dict[str, float] or keep None
    if current_prices is not None:
        if not isinstance(current_prices, dict):
            # If it's a string, try to parse it as JSON
            if isinstance(current_prices, str):
                try:
                    current_prices = json.loads(current_prices)
                except json.JSONDecodeError:
                    current_prices = None
        
        # Ensure all values are floats
        if current_prices is not None:
            current_prices = {str(k): float(v) for k, v in current_prices.items()}
    
    # Ensure portfolio_allocation values are floats
    portfolio_allocation = {str(k): float(v) for k, v in portfolio_allocation.items()}
    
    result = get_portfolio_execution_guide_func(
        portfolio_allocation=portfolio_allocation,
        total_amount=float(total_amount),
        current_prices=current_prices
    )
    return json.dumps(result, indent=2, default=str)


if __name__ == "__main__":
    app.run()
