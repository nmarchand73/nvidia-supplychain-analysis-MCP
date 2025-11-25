"""
Portfolio Optimization Module for Nvidia Supply Chain
Implements hybrid optimization combining MPT, correlation analysis, and supply chain intelligence
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys
from pathlib import Path

# Import internal modules
try:
    from . import supply_chain, technical, earnings
except ImportError:
    # Fallback for running as script
    from nvidia_mcp.internal import supply_chain, technical, earnings


# Supply chain tier mapping for intelligent weighting
SUPPLY_CHAIN_TIERS = {
    "Tier 1": ["NVDA", "TSM"],  # Core companies
    "Tier 2": ["ASML", "MU", "SK"],  # Critical suppliers
    "Tier 3": ["AMAT", "LRCX", "KLAC"],  # Equipment suppliers
    "Tier 4": ["AMKR", "ASX"],  # Packaging
    "Tier 5": ["SMCI", "DELL", "HPE"]  # System integrators
}

# Tier importance weights (higher = more important)
TIER_WEIGHTS = {
    "Tier 1": 1.0,
    "Tier 2": 0.8,
    "Tier 3": 0.6,
    "Tier 4": 0.4,
    "Tier 5": 0.3
}


def calculate_expected_returns(
    tickers: List[str],
    period: str = "3mo",
    use_momentum: bool = True
) -> pd.Series:
    """
    Calculate expected returns for portfolio optimization.
    Combines historical returns with momentum and technical indicators.
    
    Args:
        tickers: List of stock tickers
        period: Time period for analysis
        use_momentum: Whether to incorporate momentum adjustments
    
    Returns:
        Series of expected returns (annualized)
    """
    expected_returns = {}
    
    # Fetch stock data
    stock_data = supply_chain.fetch_stock_data(tickers, period)
    
    for ticker in tickers:
        if ticker not in stock_data or stock_data[ticker].get("error"):
            continue
        
        data = stock_data[ticker]
        
        # Skip if critical data is None
        if data.get("period_return") is None:
            continue
        
        # Base expected return from historical performance
        period_return = data.get("period_return", 0)
        if period_return is None:
            period_return = 0.0
        period_return = float(period_return)
        
        # Annualize the return (assuming 3mo period)
        if period == "3mo":
            annualized_return = period_return * 4
        elif period == "6mo":
            annualized_return = period_return * 2
        elif period == "1y":
            annualized_return = period_return
        else:
            # Default to 3mo annualization
            annualized_return = period_return * 4
        
        # Adjust for momentum if enabled
        if use_momentum:
            try:
                tech_signals = technical.generate_signals(ticker, period)
                if "error" not in tech_signals:
                    momentum_1m = tech_signals.get("momentum", {}).get("1_month", 0)
                    if momentum_1m is not None:
                        momentum_1m = float(momentum_1m)
                        # Momentum adjustment: positive momentum increases expected return
                        momentum_factor = 1 + (momentum_1m / 100) * 0.3  # 30% of momentum
                        annualized_return *= momentum_factor
            except Exception:
                pass  # Use base return if technical analysis fails
        
        # Ensure annualized_return is a valid number
        if annualized_return is None:
            annualized_return = 0.0
        annualized_return = float(annualized_return)
        
        expected_returns[ticker] = annualized_return / 100  # Convert to decimal
    
    return pd.Series(expected_returns)


def calculate_covariance_matrix(
    tickers: List[str],
    period: str = "3mo"
) -> pd.DataFrame:
    """
    Calculate covariance matrix for portfolio optimization.
    
    Args:
        tickers: List of stock tickers
        period: Time period for calculation
    
    Returns:
        Covariance matrix DataFrame
    """
    # Get correlation matrix
    correlations = supply_chain.calculate_correlations(tickers, period)
    
    if correlations is None or correlations.empty:
        # Fallback: identity matrix
        return pd.DataFrame(
            np.eye(len(tickers)),
            index=tickers,
            columns=tickers
        )
    
    # Get individual volatilities
    stock_data = supply_chain.fetch_stock_data(tickers, period)
    volatilities = {}
    
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].get("error"):
            vol = stock_data[ticker].get("volatility")
            if vol is None:
                vol = 30.0  # Default 30% if missing
            else:
                vol = float(vol)
            volatilities[ticker] = vol / 100  # Convert to decimal
        else:
            volatilities[ticker] = 0.30  # Default 30%
    
    # Build covariance matrix: cov(i,j) = corr(i,j) * vol(i) * vol(j)
    covariance = pd.DataFrame(
        index=tickers,
        columns=tickers
    )
    
    for ticker1 in tickers:
        for ticker2 in tickers:
            if ticker1 in correlations.index and ticker2 in correlations.columns:
                corr = correlations.loc[ticker1, ticker2]
                if pd.isna(corr) or corr is None:
                    corr = 0.5  # Default correlation if NaN/None
                else:
                    corr = float(corr)
            else:
                corr = 0.5  # Default correlation
            
            vol1 = float(volatilities.get(ticker1, 0.30))
            vol2 = float(volatilities.get(ticker2, 0.30))
            
            # Annualize if needed (assuming daily data)
            if period == "3mo":
                # Scale for 3 months of data
                covariance.loc[ticker1, ticker2] = corr * vol1 * vol2
            else:
                covariance.loc[ticker1, ticker2] = corr * vol1 * vol2
    
    return covariance


def score_supply_chain_health(ticker: str) -> float:
    """
    Calculate composite supply chain health score for a ticker.
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Health score (0-100)
    """
    score = 50.0  # Base score
    
    try:
        # Get supply chain analysis
        analysis = supply_chain.analyze_supply_chain_impact("NVDA")
        
        # Check if ticker is in supply chain data
        if ticker in analysis.get("supply_chain", {}):
            sc_data = analysis["supply_chain"][ticker]
            
            # Momentum score (0-30 points)
            period_return = sc_data.get("3m_return", 0)
            if period_return:
                if period_return > 20:
                    score += 30
                elif period_return > 10:
                    score += 20
                elif period_return > 5:
                    score += 10
                elif period_return < -10:
                    score -= 20
                elif period_return < -5:
                    score -= 10
            
            # Correlation score (0-20 points)
            if ticker in analysis.get("correlations", {}):
                corr = analysis["correlations"][ticker]
                if abs(corr) > 0.7:
                    score += 20
                elif abs(corr) > 0.5:
                    score += 10
                elif abs(corr) < 0.2:
                    score -= 10
            
            # Volatility penalty (0-10 points)
            volatility = sc_data.get("volatility", 30)
            if volatility:
                if volatility > 50:
                    score -= 10
                elif volatility < 20:
                    score += 5
        
        # Tier importance bonus
        tier_weight = 0
        for tier, tickers_in_tier in SUPPLY_CHAIN_TIERS.items():
            if ticker in tickers_in_tier:
                tier_weight = TIER_WEIGHTS.get(tier, 0.5)
                break
        
        score *= (0.8 + tier_weight * 0.2)  # Adjust by tier importance
        
    except Exception:
        pass  # Return base score if analysis fails
    
    # Clamp between 0 and 100
    return max(0, min(100, score))


def apply_correlation_diversification(
    weights: Dict[str, float],
    correlations: pd.DataFrame,
    max_correlation: float = 0.7
) -> Dict[str, float]:
    """
    Adjust weights to reduce concentration in highly correlated positions.
    
    Args:
        weights: Current portfolio weights
        correlations: Correlation matrix
        max_correlation: Maximum allowed correlation before adjustment
    
    Returns:
        Adjusted weights dictionary
    """
    adjusted_weights = weights.copy()
    
    # Find highly correlated pairs
    for ticker1 in weights:
        if ticker1 not in correlations.index:
            continue
        
        total_corr_weight = weights.get(ticker1, 0)
        
        for ticker2 in weights:
            if ticker1 == ticker2 or ticker2 not in correlations.columns:
                continue
            
            corr = correlations.loc[ticker1, ticker2]
            if pd.isna(corr):
                continue
            
            if abs(corr) > max_correlation:
                # Reduce weight of the smaller position
                weight1 = weights.get(ticker1, 0) or 0.0
                weight2 = weights.get(ticker2, 0) or 0.0
                
                if weight1 > weight2:
                    # Reduce weight2 more
                    reduction = weight2 * 0.2  # Reduce by 20%
                    adjusted_weights[ticker2] = max(0, weight2 - reduction)
                    # Redistribute to less correlated positions
                else:
                    # Reduce weight1 more
                    reduction = weight1 * 0.2
                    adjusted_weights[ticker1] = max(0, weight1 - reduction)
    
    # Renormalize to sum to 1
    total = sum(v if v is not None else 0.0 for v in adjusted_weights.values())
    if total is not None and total > 0:
        adjusted_weights = {k: (v if v is not None else 0.0) / total for k, v in adjusted_weights.items()}
    else:
        # Fallback: equal weights if total is 0 or None
        n = len(adjusted_weights)
        if n > 0:
            adjusted_weights = {k: 1.0 / n for k in adjusted_weights.keys()}
    
    return adjusted_weights


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Tuple[float, float, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        weights: Portfolio weights array
        expected_returns: Expected returns series
        covariance: Covariance matrix
        risk_free_rate: Risk-free rate (default 2%)
    
    Returns:
        Tuple of (expected_return, volatility, sharpe_ratio)
    """
    # Ensure weights are aligned with expected_returns
    # Convert to numpy arrays for efficient computation
    weights_array = np.array(weights)
    returns_array = expected_returns.values
    
    # Portfolio expected return
    portfolio_return = np.dot(weights_array, returns_array)
    
    # Portfolio variance: w^T * Cov * w
    portfolio_variance = np.dot(weights_array, np.dot(covariance.values, weights_array))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    if portfolio_volatility is None or portfolio_volatility <= 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Ensure all return values are floats, not None
    portfolio_return = float(portfolio_return) if portfolio_return is not None else 0.0
    portfolio_volatility = float(portfolio_volatility) if portfolio_volatility is not None else 0.0
    sharpe_ratio = float(sharpe_ratio) if sharpe_ratio is not None else 0.0
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


def optimize_portfolio_mpt(
    tickers: List[str],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.02,
    target_sharpe: Optional[float] = None,
    max_volatility: Optional[float] = None
) -> Dict[str, float]:
    """
    Optimize portfolio using Modern Portfolio Theory.
    
    Args:
        tickers: List of stock tickers
        expected_returns: Expected returns series
        covariance: Covariance matrix
        risk_free_rate: Risk-free rate
        target_sharpe: Target Sharpe ratio (if None, maximize Sharpe)
        max_volatility: Maximum allowed volatility
    
    Returns:
        Dictionary of optimized weights
    """
    n = len(tickers)
    
    # Initial guess: equal weights
    x0 = np.array([1.0 / n] * n)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds: weights between 0 and 1 (long-only portfolio)
    bounds = tuple((0, 1) for _ in range(n))
    
    # Objective function: negative Sharpe ratio (minimize negative = maximize Sharpe)
    def objective(weights):
        ret, vol, sharpe = portfolio_performance(weights, expected_returns, covariance, risk_free_rate)
        return -sharpe  # Negative because we're minimizing
    
    # Additional constraints
    additional_constraints = [constraints]
    
    if max_volatility:
        def volatility_constraint(weights):
            _, vol, _ = portfolio_performance(weights, expected_returns, covariance, risk_free_rate)
            return max_volatility - vol  # Must be >= 0
        
        additional_constraints.append({'type': 'ineq', 'fun': volatility_constraint})
    
    # Optimize
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=additional_constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            # Convert to dictionary, ensuring all values are floats
            weight_dict = {}
            for i in range(n):
                val = weights[i]
                if val is not None:
                    weight_dict[tickers[i]] = float(val)
            # Filter out near-zero weights (< 1%)
            weight_dict = {k: v for k, v in weight_dict.items() if v is not None and v >= 0.01}
            # Renormalize
            total = sum(v if v is not None else 0.0 for v in weight_dict.values())
            if total is not None and total > 0:
                weight_dict = {k: (v if v is not None else 0.0) / total for k, v in weight_dict.items()}
            return weight_dict
        else:
            # Fallback to equal weights if optimization fails
            return {ticker: 1.0 / n for ticker in tickers}
    
    except Exception:
        # Fallback to equal weights
        return {ticker: 1.0 / n for ticker in tickers}


def generate_risk_profiles(
    tickers: List[str],
    period: str = "3mo",
    risk_free_rate: float = 0.02
) -> Dict[str, Dict[str, Any]]:
    """
    Generate multiple risk-adjusted portfolios.
    
    Args:
        tickers: List of stock tickers
        period: Analysis period
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary with aggressive, balanced, and conservative portfolios
    """
    # Calculate inputs
    expected_returns = calculate_expected_returns(tickers, period)
    covariance = calculate_covariance_matrix(tickers, period)
    
    # Filter to only tickers with valid data (non-empty expected returns and valid covariance)
    valid_tickers = [t for t in tickers if t in expected_returns.index and t in covariance.index]
    
    # Additional check: ensure expected returns are not None/NaN
    valid_tickers = [
        t for t in valid_tickers 
        if pd.notna(expected_returns.get(t)) and expected_returns.get(t) is not None
    ]
    
    if not valid_tickers:
        return {
            "error": "No valid tickers for optimization"
        }
    
    expected_returns = expected_returns[valid_tickers]
    covariance = covariance.loc[valid_tickers, valid_tickers]
    
    portfolios = {}
    
    # 1. Aggressive Portfolio (maximize Sharpe, allow higher volatility)
    aggressive_weights = optimize_portfolio_mpt(
        valid_tickers,
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
        max_volatility=None  # No volatility constraint
    )
    
    # Ensure weights array matches expected_returns index order
    weights_array = np.array([aggressive_weights.get(t, 0) for t in expected_returns.index])
    agg_ret, agg_vol, agg_sharpe = portfolio_performance(
        weights_array,
        expected_returns,
        covariance,
        risk_free_rate
    )
    
    portfolios["aggressive"] = {
        "weights": aggressive_weights,
        "expected_return": float(agg_ret * 100),  # Convert to percentage
        "volatility": float(agg_vol * 100),
        "sharpe_ratio": float(agg_sharpe),
        "risk_level": "high"
    }
    
    # 2. Balanced Portfolio (moderate volatility constraint)
    # Calculate target volatility (median of individual volatilities)
    stock_data = supply_chain.fetch_stock_data(valid_tickers, period)
    volatilities = []
    for t in valid_tickers:
        if t in stock_data and not stock_data[t].get("error"):
            vol = stock_data[t].get("volatility")
            if vol is not None:
                volatilities.append(float(vol) / 100)
            else:
                volatilities.append(0.30)  # Default 30%
    target_vol = np.median(volatilities) if volatilities else 0.25
    
    balanced_weights = optimize_portfolio_mpt(
        valid_tickers,
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
        max_volatility=target_vol
    )
    
    # Ensure weights array matches expected_returns index order
    weights_array = np.array([balanced_weights.get(t, 0) for t in expected_returns.index])
    bal_ret, bal_vol, bal_sharpe = portfolio_performance(
        weights_array,
        expected_returns,
        covariance,
        risk_free_rate
    )
    
    portfolios["balanced"] = {
        "weights": balanced_weights,
        "expected_return": float(bal_ret * 100),
        "volatility": float(bal_vol * 100),
        "sharpe_ratio": float(bal_sharpe),
        "risk_level": "medium"
    }
    
    # 3. Conservative Portfolio (lower volatility constraint)
    conservative_vol = target_vol * 0.7  # 30% lower volatility
    
    conservative_weights = optimize_portfolio_mpt(
        valid_tickers,
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
        max_volatility=conservative_vol
    )
    
    # Ensure weights array matches expected_returns index order
    weights_array = np.array([conservative_weights.get(t, 0) for t in expected_returns.index])
    cons_ret, cons_vol, cons_sharpe = portfolio_performance(
        weights_array,
        expected_returns,
        covariance,
        risk_free_rate
    )
    
    portfolios["conservative"] = {
        "weights": conservative_weights,
        "expected_return": float(cons_ret * 100),
        "volatility": float(cons_vol * 100),
        "sharpe_ratio": float(cons_sharpe),
        "risk_level": "low"
    }
    
    # Add correlation analysis for each portfolio
    correlations = supply_chain.calculate_correlations(valid_tickers, period)
    
    for profile_name in portfolios:
        weights = portfolios[profile_name]["weights"]
        
        # Apply correlation diversification
        if correlations is not None and not correlations.empty:
            adjusted_weights = apply_correlation_diversification(weights, correlations)
            portfolios[profile_name]["weights"] = adjusted_weights
            
            # Recalculate metrics with adjusted weights
            weights_array = np.array([adjusted_weights.get(t, 0) for t in expected_returns.index])
            adj_ret, adj_vol, adj_sharpe = portfolio_performance(
                weights_array,
                expected_returns,
                covariance,
                risk_free_rate
            )
            portfolios[profile_name]["expected_return"] = float(adj_ret * 100)
            portfolios[profile_name]["volatility"] = float(adj_vol * 100)
            portfolios[profile_name]["sharpe_ratio"] = float(adj_sharpe)
        
        # Add supply chain health scores
        health_scores = {}
        for ticker in weights:
            health_scores[ticker] = score_supply_chain_health(ticker)
        portfolios[profile_name]["health_scores"] = health_scores
        
        # Add allocation rationale
        rationale = []
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights[:5]:  # Top 5 positions
            health = health_scores.get(ticker, 50)
            rationale.append({
                "ticker": ticker,
                "allocation": float(weight * 100),
                "health_score": health,
                "reason": _generate_allocation_reason(ticker, weight, health)
            })
        portfolios[profile_name]["allocation_rationale"] = rationale
    
    return portfolios


def _generate_allocation_reason(ticker: str, weight: float, health_score: float) -> str:
    """Generate human-readable reason for allocation."""
    reasons = []
    
    if weight > 0.20:
        reasons.append("Large position due to high expected return")
    elif weight > 0.10:
        reasons.append("Significant allocation for diversification")
    
    if health_score > 70:
        reasons.append("Strong supply chain health")
    elif health_score < 40:
        reasons.append("Lower allocation due to supply chain concerns")
    
    # Check tier
    for tier, tickers_in_tier in SUPPLY_CHAIN_TIERS.items():
        if ticker in tickers_in_tier:
            if tier == "Tier 1":
                reasons.append("Core supply chain position")
            break
    
    return "; ".join(reasons) if reasons else "Balanced portfolio allocation"


def optimize_portfolio(
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    risk_profile: str = "all"
) -> Dict[str, Any]:
    """
    Main portfolio optimization function.
    
    Args:
        tickers: List of tickers to optimize (defaults to full supply chain)
        period: Analysis period
        risk_profile: "aggressive", "balanced", "conservative", or "all"
    
    Returns:
        Dictionary with optimized portfolios and analysis
    """
    if tickers is None:
        tickers = list(supply_chain.NVIDIA_ECOSYSTEM.keys())
    
    # Pre-filter tickers with valid data to avoid None division errors
    stock_data = supply_chain.fetch_stock_data(tickers, period)
    valid_tickers = [
        t for t in tickers 
        if t in stock_data 
        and not stock_data[t].get("error")
        and stock_data[t].get("period_return") is not None
        and stock_data[t].get("volatility") is not None
    ]
    
    if not valid_tickers:
        return {
            "error": "No tickers with valid data for optimization",
            "error_type": "ValueError"
        }
    
    # Use only valid tickers
    tickers = valid_tickers
    
    # Generate risk profiles
    portfolios = generate_risk_profiles(tickers, period)
    
    if "error" in portfolios:
        return portfolios
    
    # Select requested profile(s)
    result = {
        "timestamp": datetime.now().isoformat(),
        "period": period,
        "tickers_analyzed": tickers
    }
    
    if risk_profile == "all":
        result["portfolios"] = portfolios
    elif risk_profile in portfolios:
        result["portfolios"] = {risk_profile: portfolios[risk_profile]}
    else:
        result["portfolios"] = portfolios  # Return all if invalid profile
    
    # Add correlation matrix
    correlations = supply_chain.calculate_correlations(tickers, period)
    if correlations is not None:
        result["correlation_matrix"] = correlations.to_dict()
    
    # Add rebalancing recommendations
    result["rebalancing_recommendations"] = {
        "frequency": "quarterly",
        "triggers": [
            "Significant correlation changes (>0.2)",
            "Major supply chain events",
            "Earnings calendar shifts",
            "Volatility spikes (>50% increase)"
        ]
    }
    
    return result

