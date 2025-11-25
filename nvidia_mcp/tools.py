"""
MCP Tool implementations for Nvidia Supply Chain Analysis
Main entry point tools that route to internal functions
"""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import internal modules
try:
    from .internal import supply_chain, technical, earnings, trading_instructions, portfolio_optimizer
except ImportError:
    # Fallback for running as script
    from nvidia_mcp.internal import supply_chain, technical, earnings, trading_instructions, portfolio_optimizer


def analyze_supply_chain(
    analysis_type: str,
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    reference_ticker: str = "NVDA"
) -> Dict[str, Any]:
    """
    Main entry point for supply chain analysis.
    Routes to appropriate internal functions based on analysis_type.
    
    Args:
        analysis_type: Type of analysis to perform
            - "health": Supply chain health assessment
            - "technical": Technical analysis and trading signals
            - "earnings": Earnings calendar and event analysis
            - "bottlenecks": Focused analysis on bottleneck tickers
            - "pairs": Pair trading opportunities
            - "portfolio": Portfolio allocation recommendations
            - "full": Comprehensive analysis combining all aspects
        tickers: List of tickers to analyze (defaults to full supply chain)
        period: Time period for analysis (default: "3mo")
        reference_ticker: Reference ticker for correlations (default: "NVDA")
    
    Returns:
        Dictionary with analysis results
    """
    # Default to full supply chain if no tickers specified
    if tickers is None:
        tickers = list(supply_chain.NVIDIA_ECOSYSTEM.keys())
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": analysis_type,
        "tickers": tickers,
        "period": period
    }
    
    try:
        if analysis_type == "health":
            # Supply chain health assessment
            analysis = supply_chain.analyze_supply_chain_impact(reference_ticker)
            
            # Calculate health score (simplified version)
            health_score = _calculate_health_score(analysis)
            
            result.update({
                "health_score": health_score,
                "supply_chain_data": analysis.get("supply_chain", {}),
                "correlations": analysis.get("correlations", {}),
                "momentum_indicators": analysis.get("momentum_indicators", {}),
                "risk_indicators": analysis.get("risk_indicators", [])
            })
            
        elif analysis_type == "technical":
            # Technical analysis and trading signals
            scan_results = technical.scan_supply_chain(tickers, period)
            
            # Get detailed signals for each ticker
            detailed_signals = {}
            for ticker in tickers[:10]:  # Limit to first 10 for performance
                try:
                    signals = technical.generate_signals(ticker, period)
                    if "error" not in signals:
                        detailed_signals[ticker] = signals
                except Exception as e:
                    detailed_signals[ticker] = {"error": str(e)}
            
            result.update({
                "summary": scan_results.get("summary", {}),
                "detailed_signals": detailed_signals
            })
            
        elif analysis_type == "earnings":
            # Earnings calendar and event analysis
            event_calendar = earnings.compile_event_calendar(tickers)
            
            # Analyze earnings impact for key tickers
            earnings_impacts = {}
            for ticker in tickers[:5]:  # Limit to first 5 for performance
                try:
                    impact = earnings.analyze_earnings_impact(ticker)
                    if "error" not in impact:
                        earnings_impacts[ticker] = impact
                except Exception as e:
                    earnings_impacts[ticker] = {"error": str(e)}
            
            result.update({
                "event_calendar": event_calendar,
                "earnings_impacts": earnings_impacts
            })
            
        elif analysis_type == "bottlenecks":
            # Focused analysis on bottleneck tickers
            bottleneck_tickers = ["SK", "MU", "TSM", "ASML"]
            bottleneck_data = {}
            
            for ticker in bottleneck_tickers:
                if ticker in tickers or ticker in supply_chain.NVIDIA_ECOSYSTEM:
                    try:
                        stock_data = supply_chain.fetch_stock_data([ticker], period)
                        technical_signals = technical.generate_signals(ticker, period)
                        earnings_data = earnings.get_earnings_calendar([ticker])
                        
                        bottleneck_data[ticker] = {
                            "stock_data": stock_data.get(ticker, {}),
                            "technical_signals": technical_signals if "error" not in technical_signals else {},
                            "earnings": earnings_data.get(ticker, {})
                        }
                    except Exception as e:
                        bottleneck_data[ticker] = {"error": str(e)}
            
            result.update({
                "bottleneck_analysis": bottleneck_data
            })
            
        elif analysis_type == "pairs":
            # Pair trading opportunities
            correlations = supply_chain.calculate_correlations(tickers, period)
            
            # Define classic pairs
            classic_pairs = [
                ("NVDA", "AMD"),
                ("TSM", "INTC"),
                ("AMAT", "LRCX"),
                ("MU", "SK")
            ]
            
            pair_analysis = {}
            if correlations is not None:
                for ticker1, ticker2 in classic_pairs:
                    if ticker1 in correlations.columns and ticker2 in correlations.columns:
                        corr_value = correlations.loc[ticker1, ticker2]
                        
                        # Get price data for spread calculation
                        try:
                            data1 = supply_chain.fetch_stock_data([ticker1], period)
                            data2 = supply_chain.fetch_stock_data([ticker2], period)
                            
                            price1 = data1.get(ticker1, {}).get("current_price")
                            price2 = data2.get(ticker2, {}).get("current_price")
                            
                            if price1 and price2:
                                spread = abs(price1 - price2)
                                avg_price = (price1 + price2) / 2
                                spread_pct = (spread / avg_price) * 100 if avg_price > 0 else 0
                                
                                pair_analysis[f"{ticker1}/{ticker2}"] = {
                                    "correlation": float(corr_value),
                                    "spread": spread,
                                    "spread_pct": spread_pct,
                                    "price1": price1,
                                    "price2": price2
                                }
                        except Exception as e:
                            pair_analysis[f"{ticker1}/{ticker2}"] = {"error": str(e)}
            
            result.update({
                "correlation_matrix": correlations.to_dict() if correlations is not None else {},
                "pair_analysis": pair_analysis
            })
            
        elif analysis_type == "portfolio":
            # Portfolio allocation recommendations using intelligent optimization
            analysis = supply_chain.analyze_supply_chain_impact(reference_ticker)
            
            # Generate optimized portfolios using hybrid algorithm
            try:
                optimized_portfolios = portfolio_optimizer.optimize_portfolio(
                    tickers=tickers,
                    period=period,
                    risk_profile="all"
                )
                
                # Format the portfolios for output
                portfolio_recommendations = {}
                if "portfolios" in optimized_portfolios:
                    for profile_name, profile_data in optimized_portfolios["portfolios"].items():
                        # Convert weights to percentages and format
                        weights = profile_data.get("weights", {})
                        portfolio_recommendations[profile_name] = {
                            ticker: round(weight * 100, 2) 
                            for ticker, weight in weights.items()
                        }
                
                result.update({
                    "portfolio_recommendations": portfolio_recommendations,
                    "portfolio_metrics": {
                        profile_name: {
                            "expected_return": profile_data.get("expected_return"),
                            "volatility": profile_data.get("volatility"),
                            "sharpe_ratio": profile_data.get("sharpe_ratio"),
                            "risk_level": profile_data.get("risk_level")
                        }
                        for profile_name, profile_data in optimized_portfolios.get("portfolios", {}).items()
                    },
                    "allocation_rationale": {
                        profile_name: profile_data.get("allocation_rationale", [])
                        for profile_name, profile_data in optimized_portfolios.get("portfolios", {}).items()
                    },
                    "correlation_matrix": optimized_portfolios.get("correlation_matrix", {}),
                    "rebalancing_recommendations": optimized_portfolios.get("rebalancing_recommendations", {}),
                    "supply_chain_data": analysis.get("supply_chain", {}),
                    "risk_indicators": analysis.get("risk_indicators", [])
                })
            except Exception as e:
                # Fallback to basic analysis if optimization fails
                result.update({
                    "portfolio_recommendations": {
                        "error": f"Optimization failed: {str(e)}",
                        "fallback": "Using basic supply chain analysis"
                    },
                    "supply_chain_data": analysis.get("supply_chain", {}),
                    "risk_indicators": analysis.get("risk_indicators", [])
                })
            
        elif analysis_type == "full":
            # Comprehensive analysis combining all aspects
            health_analysis = supply_chain.analyze_supply_chain_impact(reference_ticker)
            technical_scan = technical.scan_supply_chain(tickers[:10], period)
            earnings_cal = earnings.compile_event_calendar(tickers[:10])
            
            result.update({
                "health_assessment": {
                    "supply_chain_data": health_analysis.get("supply_chain", {}),
                    "correlations": health_analysis.get("correlations", {}),
                    "momentum_indicators": health_analysis.get("momentum_indicators", {}),
                    "risk_indicators": health_analysis.get("risk_indicators", [])
                },
                "technical_analysis": technical_scan.get("summary", {}),
                "earnings_calendar": earnings_cal.get("upcoming_events", []),
                "catalyst_clusters": earnings_cal.get("catalyst_clusters", [])
            })
            
        else:
            result["error"] = f"Unknown analysis type: {analysis_type}. Valid types: health, technical, earnings, bottlenecks, pairs, portfolio, full"
            
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
    
    return result


def get_supply_chain_data(
    tickers: List[str],
    period: str = "3mo"
) -> Dict[str, Any]:
    """
    Fetch raw stock data for specific ticker(s).
    
    Args:
        tickers: List of stock tickers to fetch
        period: Time period (default: "3mo")
    
    Returns:
        Dictionary with stock data for each ticker
    """
    try:
        data = supply_chain.fetch_stock_data(tickers, period)
        return {
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "data": data
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_earnings_calendar(
    tickers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get upcoming earnings dates and events.
    
    Args:
        tickers: List of tickers (defaults to full supply chain)
    
    Returns:
        Dictionary with earnings calendar data
    """
    try:
        if tickers is None:
            tickers = list(supply_chain.NVIDIA_ECOSYSTEM.keys())
        
        calendar_data = earnings.get_earnings_calendar(tickers)
        clusters = earnings.identify_catalyst_clusters(calendar_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "earnings_calendar": calendar_data,
            "catalyst_clusters": clusters
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }


def _calculate_health_score(analysis: Dict[str, Any]) -> float:
    """
    Calculate a composite health score (0-100) based on supply chain analysis.
    Simplified version based on available data.
    """
    score = 50.0  # Base score
    
    # Adjust based on momentum indicators
    momentum = analysis.get("momentum_indicators", {})
    top_performers = momentum.get("top_performers", [])
    if top_performers:
        avg_return = sum(ret for _, ret in top_performers) / len(top_performers)
        if avg_return > 10:
            score += 20
        elif avg_return > 5:
            score += 10
    
    # Adjust based on correlations
    correlations = analysis.get("correlations", {})
    if correlations:
        avg_corr = sum(abs(v) for v in correlations.values()) / len(correlations)
        if avg_corr > 0.7:
            score += 15
        elif avg_corr < 0.3:
            score -= 10
    
    # Adjust based on risk indicators
    risk_indicators = analysis.get("risk_indicators", [])
    score -= len(risk_indicators) * 5
    
    # Clamp between 0 and 100
    return max(0, min(100, score))


def get_trading_instructions(
    action: str = "both",
    order_type: str | None = None,
    include_fractional: bool = True
) -> Dict[str, Any]:
    """
    Get step-by-step instructions for buying/selling stocks on Interactive Brokers Mobile App.
    
    Args:
        action: "buy", "sell", or "both" (default: "both")
        order_type: Specific order type to focus on (Market, Limit, Stop, Stop Limit) or None for all
        include_fractional: Whether to include fractional share trading instructions (default: True)
    
    Returns:
        Dictionary with detailed trading instructions
    """
    try:
        return trading_instructions.get_ibkr_mobile_trading_instructions(
            action=action,
            order_type=order_type,
            include_fractional=include_fractional
        )
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_portfolio_execution_guide(
    portfolio_allocation: Dict[str, float],
    total_amount: float,
    current_prices: Dict[str, float] | None = None
) -> Dict[str, Any]:
    """
    Generate step-by-step execution guide for implementing a portfolio allocation on IBKR Mobile.
    
    Args:
        portfolio_allocation: Dictionary mapping tickers to percentage allocations (e.g., {"NVDA": 30, "TSM": 30})
        total_amount: Total portfolio value in base currency (e.g., 500 for 500€)
        current_prices: Optional dictionary of current stock prices for share calculations
    
    Returns:
        Dictionary with execution plan and instructions
    """
    try:
        return trading_instructions.get_portfolio_execution_guide(
            portfolio_allocation=portfolio_allocation,
            total_amount=total_amount,
            current_prices=current_prices
        )
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }

