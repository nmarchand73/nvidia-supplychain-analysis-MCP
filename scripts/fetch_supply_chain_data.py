#!/usr/bin/env python3
"""
Fetch and analyze Nvidia supply chain stock data
This script retrieves current and historical data for Nvidia and its key suppliers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

# Key Nvidia supply chain companies and their relationships
NVIDIA_ECOSYSTEM = {
    "NVDA": {
        "name": "NVIDIA Corporation",
        "role": "Primary company",
        "category": "GPU Designer"
    },
    "TSM": {
        "name": "Taiwan Semiconductor Manufacturing",
        "role": "Primary chip manufacturer",
        "category": "Foundry",
        "relationship": "Manufactures all advanced GPUs (4nm, 5nm nodes)"
    },
    "ASML": {
        "name": "ASML Holding",
        "role": "EUV lithography equipment",
        "category": "Equipment Supplier",
        "relationship": "Supplies critical lithography to TSMC"
    },
    "AMAT": {
        "name": "Applied Materials",
        "role": "Semiconductor equipment",
        "category": "Equipment Supplier",
        "relationship": "Provides manufacturing equipment"
    },
    "LRCX": {
        "name": "Lam Research",
        "role": "Wafer fabrication equipment",
        "category": "Equipment Supplier",
        "relationship": "Etching and deposition equipment"
    },
    "KLAC": {
        "name": "KLA Corporation",
        "role": "Process control equipment",
        "category": "Equipment Supplier",
        "relationship": "Inspection and metrology"
    },
    "SK": {
        "name": "SK Hynix",
        "role": "HBM memory supplier",
        "category": "Memory Supplier",
        "relationship": "Supplies HBM3 memory for AI GPUs"
    },
    "MU": {
        "name": "Micron Technology",
        "role": "Memory supplier",
        "category": "Memory Supplier",
        "relationship": "Alternative HBM supplier"
    },
    "AMKR": {
        "name": "Amkor Technology",
        "role": "Packaging and test",
        "category": "OSAT",
        "relationship": "Advanced packaging for GPUs"
    },
    "ASX": {
        "name": "ASE Group",
        "role": "Assembly and test",
        "category": "OSAT",
        "relationship": "IC packaging services"
    },
    "SMCI": {
        "name": "Super Micro Computer",
        "role": "Server systems",
        "category": "System Integrator",
        "relationship": "AI server manufacturer using Nvidia GPUs"
    },
    "DELL": {
        "name": "Dell Technologies",
        "role": "Enterprise systems",
        "category": "System Integrator",
        "relationship": "Enterprise AI infrastructure"
    },
    "HPE": {
        "name": "Hewlett Packard Enterprise",
        "role": "Enterprise systems",
        "category": "System Integrator",
        "relationship": "HPC and AI systems"
    }
}

def fetch_stock_data(tickers, period="3mo"):
    """
    Fetch stock data for multiple tickers
    
    Args:
        tickers: List of stock tickers
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        Dictionary with ticker data
    """
    data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            # Get current info
            info = stock.info
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
            start_price = hist['Close'].iloc[0] if len(hist) > 0 else None
            
            # Convert history to dict with string keys for JSON serialization
            history_dict = None
            if len(hist) > 0:
                # Convert DataFrame to dict, then convert Timestamp keys to strings
                history_dict = {}
                for date_idx, row in hist.iterrows():
                    date_str = date_idx.strftime('%Y-%m-%d') if hasattr(date_idx, 'strftime') else str(date_idx)
                    history_dict[date_str] = {
                        col: float(val) if pd.notna(val) and isinstance(val, (int, float, np.number)) else None
                        for col, val in row.items()
                    }
            
            data[ticker] = {
                "current_price": float(current_price) if current_price is not None and pd.notna(current_price) else None,
                "period_return": float(((current_price / start_price - 1) * 100)) if current_price and start_price and pd.notna(current_price) and pd.notna(start_price) else None,
                "volatility": float(hist['Close'].pct_change().std() * np.sqrt(252) * 100) if len(hist) > 1 else None,
                "volume_avg": float(hist['Volume'].mean()) if len(hist) > 0 and pd.notna(hist['Volume'].mean()) else None,
                "market_cap": float(info.get('marketCap')) if info.get('marketCap') is not None else None,
                "pe_ratio": float(info.get('trailingPE')) if info.get('trailingPE') is not None else None,
                "beta": float(info.get('beta')) if info.get('beta') is not None else None,
                "52w_high": float(info.get('fiftyTwoWeekHigh')) if info.get('fiftyTwoWeekHigh') is not None else None,
                "52w_low": float(info.get('fiftyTwoWeekLow')) if info.get('fiftyTwoWeekLow') is not None else None,
                "history": history_dict
            }
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}", file=sys.stderr)
            data[ticker] = {"error": str(e)}
    
    return data

def calculate_correlations(tickers, period="3mo"):
    """
    Calculate price correlations between stocks
    
    Args:
        tickers: List of stock tickers
        period: Time period for correlation calculation
    
    Returns:
        Correlation matrix as DataFrame
    """
    price_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if len(hist) > 0:
                price_data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error fetching {ticker} for correlation: {e}", file=sys.stderr)
    
    if price_data:
        df = pd.DataFrame(price_data)
        # Calculate returns
        returns = df.pct_change().dropna()
        # Calculate correlation
        correlation = returns.corr()
        return correlation
    
    return None

def analyze_supply_chain_impact(reference_ticker="NVDA"):
    """
    Analyze the impact and relationships within the supply chain
    
    Args:
        reference_ticker: The main ticker to compare others against
    
    Returns:
        Analysis results
    """
    tickers = list(NVIDIA_ECOSYSTEM.keys())
    
    # Fetch data
    print("Fetching stock data...", file=sys.stderr)
    stock_data = fetch_stock_data(tickers)
    
    # Calculate correlations
    print("Calculating correlations...", file=sys.stderr)
    correlations = calculate_correlations(tickers)
    
    # Analyze results
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "reference": reference_ticker,
        "supply_chain": {},
        "correlations": {},
        "momentum_indicators": {},
        "risk_indicators": []
    }
    
    # Process each company
    for ticker, info in NVIDIA_ECOSYSTEM.items():
        if ticker in stock_data and not stock_data[ticker].get("error"):
            data = stock_data[ticker]
            
            analysis["supply_chain"][ticker] = {
                "name": info["name"],
                "role": info["role"],
                "category": info["category"],
                "relationship": info.get("relationship", ""),
                "current_price": data.get("current_price"),
                "3m_return": data.get("period_return"),
                "volatility": data.get("volatility"),
                "market_cap": data.get("market_cap"),
                "pe_ratio": data.get("pe_ratio"),
                "beta": data.get("beta")
            }
            
            # Add correlation to NVDA
            if correlations is not None and ticker in correlations.columns and reference_ticker in correlations.columns:
                analysis["correlations"][ticker] = float(correlations.loc[ticker, reference_ticker])
    
    # Identify momentum leaders/laggards
    returns = [(t, analysis["supply_chain"][t]["3m_return"]) 
               for t in analysis["supply_chain"] 
               if analysis["supply_chain"][t].get("3m_return") is not None]
    
    returns_sorted = sorted(returns, key=lambda x: x[1], reverse=True)
    
    if returns_sorted:
        analysis["momentum_indicators"]["top_performers"] = returns_sorted[:3]
        analysis["momentum_indicators"]["laggards"] = returns_sorted[-3:]
    
    # Identify risk indicators
    for ticker in analysis["supply_chain"]:
        company = analysis["supply_chain"][ticker]
        
        # High volatility warning
        if company.get("volatility") and company["volatility"] > 50:
            analysis["risk_indicators"].append({
                "ticker": ticker,
                "type": "high_volatility",
                "value": company["volatility"],
                "message": f"{ticker} shows high volatility ({company['volatility']:.1f}%)"
            })
        
        # Correlation divergence
        if ticker in analysis["correlations"]:
            corr = analysis["correlations"][ticker]
            if abs(corr) < 0.3 and ticker != reference_ticker:
                analysis["risk_indicators"].append({
                    "ticker": ticker,
                    "type": "low_correlation",
                    "value": corr,
                    "message": f"{ticker} shows low correlation with NVDA ({corr:.2f})"
                })
    
    return analysis

def calculate_expected_returns(tickers, period="3mo"):
    """
    Calculate expected returns for portfolio optimization.
    Estimates forward-looking returns based on momentum and historical performance.
    
    Args:
        tickers: List of stock tickers
        period: Time period for analysis
    
    Returns:
        Dictionary mapping tickers to expected annualized returns (as decimals)
    """
    stock_data = fetch_stock_data(tickers, period)
    expected_returns = {}
    
    for ticker in tickers:
        if ticker not in stock_data or stock_data[ticker].get("error"):
            continue
        
        data = stock_data[ticker]
        period_return = data.get("period_return", 0)
        
        if period_return is None:
            period_return = 0
        
        # Annualize based on period
        if period == "3mo":
            annualized = period_return * 4 / 100  # Convert to decimal
        elif period == "6mo":
            annualized = period_return * 2 / 100
        elif period == "1y":
            annualized = period_return / 100
        else:
            annualized = period_return * 4 / 100  # Default to 3mo
        
        expected_returns[ticker] = annualized
    
    return expected_returns

def calculate_covariance_matrix(tickers, period="3mo"):
    """
    Calculate full covariance matrix for MPT optimization.
    
    Args:
        tickers: List of stock tickers
        period: Time period for calculation
    
    Returns:
        Covariance matrix as DataFrame
    """
    # Get correlation matrix
    correlations = calculate_correlations(tickers, period)
    
    if correlations is None or correlations.empty:
        # Fallback: create identity matrix
        return pd.DataFrame(
            np.eye(len(tickers)),
            index=tickers,
            columns=tickers
        )
    
    # Get individual volatilities
    stock_data = fetch_stock_data(tickers, period)
    volatilities = {}
    
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].get("error"):
            vol = stock_data[ticker].get("volatility", 30)
            if vol is None:
                vol = 30
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
            else:
                corr = 0.5  # Default correlation
            
            vol1 = volatilities.get(ticker1, 0.30)
            vol2 = volatilities.get(ticker2, 0.30)
            
            covariance.loc[ticker1, ticker2] = corr * vol1 * vol2
    
    return covariance

def get_supply_chain_scores(tickers):
    """
    Get health scores for each supply chain tier.
    
    Args:
        tickers: List of stock tickers
    
    Returns:
        Dictionary mapping tickers to health scores (0-100)
    """
    analysis = analyze_supply_chain_impact("NVDA")
    scores = {}
    
    # Tier definitions
    tier_importance = {
        "Tier 1": 1.0,  # NVDA, TSM
        "Tier 2": 0.8,  # ASML, MU, SK
        "Tier 3": 0.6,  # AMAT, LRCX, KLAC
        "Tier 4": 0.4,  # AMKR, ASX
        "Tier 5": 0.3   # SMCI, DELL, HPE
    }
    
    tier_mapping = {
        "NVDA": "Tier 1",
        "TSM": "Tier 1",
        "ASML": "Tier 2",
        "MU": "Tier 2",
        "SK": "Tier 2",
        "AMAT": "Tier 3",
        "LRCX": "Tier 3",
        "KLAC": "Tier 3",
        "AMKR": "Tier 4",
        "ASX": "Tier 4",
        "SMCI": "Tier 5",
        "DELL": "Tier 5",
        "HPE": "Tier 5"
    }
    
    for ticker in tickers:
        score = 50.0  # Base score
        
        if ticker in analysis.get("supply_chain", {}):
            sc_data = analysis["supply_chain"][ticker]
            
            # Momentum component (0-30 points)
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
            
            # Correlation component (0-20 points)
            if ticker in analysis.get("correlations", {}):
                corr = analysis["correlations"][ticker]
                if abs(corr) > 0.7:
                    score += 20
                elif abs(corr) > 0.5:
                    score += 10
                elif abs(corr) < 0.2:
                    score -= 10
            
            # Volatility adjustment (0-10 points)
            volatility = sc_data.get("volatility", 30)
            if volatility:
                if volatility > 50:
                    score -= 10
                elif volatility < 20:
                    score += 5
        
        # Apply tier importance
        tier = tier_mapping.get(ticker, "Tier 3")
        importance = tier_importance.get(tier, 0.5)
        score *= (0.8 + importance * 0.2)
        
        scores[ticker] = max(0, min(100, score))
    
    return scores

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Nvidia supply chain stocks')
    parser.add_argument('--period', default='3mo', help='Analysis period (1mo, 3mo, 6mo, 1y)')
    parser.add_argument('--format', default='json', choices=['json', 'summary'], help='Output format')
    parser.add_argument('--ticker', help='Specific ticker to analyze')
    
    args = parser.parse_args()
    
    if args.ticker:
        # Analyze specific ticker
        data = fetch_stock_data([args.ticker], period=args.period)
        print(json.dumps(data, indent=2, default=str))
    else:
        # Full supply chain analysis
        analysis = analyze_supply_chain_impact()
        
        if args.format == 'json':
            print(json.dumps(analysis, indent=2, default=str))
        else:
            # Summary format
            print("\n=== NVIDIA SUPPLY CHAIN ANALYSIS ===")
            print(f"Timestamp: {analysis['timestamp']}")
            print("\n--- Top Performers (3M) ---")
            for ticker, return_pct in analysis["momentum_indicators"].get("top_performers", []):
                name = analysis["supply_chain"][ticker]["name"]
                print(f"{ticker}: {name} - {return_pct:.1f}%")
            
            print("\n--- Laggards (3M) ---")
            for ticker, return_pct in analysis["momentum_indicators"].get("laggards", []):
                name = analysis["supply_chain"][ticker]["name"]
                print(f"{ticker}: {name} - {return_pct:.1f}%")
            
            print("\n--- Risk Indicators ---")
            for risk in analysis["risk_indicators"]:
                print(f"⚠️  {risk['message']}")
            
            print("\n--- Key Correlations with NVDA ---")
            corr_sorted = sorted(analysis["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)
            for ticker, corr in corr_sorted[:5]:
                if ticker != "NVDA":
                    name = analysis["supply_chain"][ticker]["name"]
                    print(f"{ticker} ({name}): {corr:.2f}")

if __name__ == "__main__":
    main()
