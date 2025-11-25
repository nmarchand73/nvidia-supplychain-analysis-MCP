# Nvidia Supply Chain Analysis - MCP Server

MCP (Model Context Protocol) server for analyzing Nvidia's semiconductor supply chain ecosystem. This server provides AI assistants with tools to analyze supply chain health, technical indicators, earnings events, bottlenecks, pair trading opportunities, and portfolio recommendations.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package:
```bash
pip install -e .
```

## MCP Server Usage

### Running the Server

The MCP server uses FastMCP and can be run with:

```bash
python -m nvidia_mcp.server
```

### Available Tools

#### 1. `analyze_supply_chain`

Main entry point for supply chain analysis. Supports multiple analysis types:

**Parameters:**
- `analysis_type` (required): Type of analysis to perform
  - `"health"`: Supply chain health assessment with correlations, momentum, and risk indicators
  - `"technical"`: Technical analysis and trading signals (RSI, MACD, Bollinger Bands)
  - `"earnings"`: Earnings calendar and event analysis
  - `"bottlenecks"`: Focused analysis on bottleneck tickers (SK, MU, TSM, ASML)
  - `"pairs"`: Pair trading opportunities and correlation analysis
  - `"portfolio"`: Portfolio allocation recommendations
  - `"full"`: Comprehensive analysis combining all aspects
- `tickers` (optional): List of tickers to analyze (defaults to full supply chain)
- `period` (optional): Time period for analysis (default: "3mo")
- `reference_ticker` (optional): Reference ticker for correlations (default: "NVDA")

**Example:**
```json
{
  "analysis_type": "health",
  "period": "6mo",
  "reference_ticker": "NVDA"
}
```

#### 2. `get_supply_chain_data`

Fetch raw stock data for specific ticker(s).

**Parameters:**
- `tickers` (required): List of stock tickers to fetch
- `period` (optional): Time period (default: "3mo")

**Example:**
```json
{
  "tickers": ["NVDA", "TSM", "ASML"],
  "period": "1y"
}
```

#### 3. `get_earnings_calendar`

Get upcoming earnings dates and events for supply chain tickers.

**Parameters:**
- `tickers` (optional): List of tickers (defaults to full supply chain)

**Example:**
```json
{
  "tickers": ["NVDA", "TSM"]
}
```

#### 4. `get_trading_instructions`

Get step-by-step instructions for buying/selling stocks on Interactive Brokers Mobile App. Based on official IBKR Mobile app documentation.

**Parameters:**
- `action` (optional): "buy", "sell", or "both" (default: "both") - which instructions to include
- `order_type` (optional): Specific order type to focus on (Market, Limit, Stop, Stop Limit) or None for all types
- `include_fractional` (optional): Whether to include fractional share trading setup instructions (default: true)

**Example:**
```json
{
  "action": "both",
  "order_type": "Limit",
  "include_fractional": true
}
```

**Returns:**
- Setup steps (app download, fractional shares enablement)
- Buy order instructions (multiple methods: watchlist, trade button, search)
- Sell order instructions
- Order configuration details (quantity, order type, price, time-in-force)
- Order management (view, modify, cancel)
- Order type explanations and best practices
- Tips and risk management advice

#### 5. `get_portfolio_execution_guide`

Generate step-by-step execution guide for implementing a portfolio allocation on IBKR Mobile App. Converts portfolio percentages into actionable trading instructions.

**Parameters:**
- `portfolio_allocation` (required): Dictionary mapping tickers to percentage allocations
- `total_amount` (required): Total portfolio value in base currency (e.g., 500 for 500€)
- `current_prices` (optional): Dictionary of current stock prices for accurate share calculations

**Example:**
```json
{
  "portfolio_allocation": {
    "NVDA": 30.0,
    "TSM": 30.0,
    "LRCX": 10.0,
    "AMAT": 10.0,
    "MU": 10.0,
    "ASX": 5.0,
    "ASML": 5.0
  },
  "total_amount": 500,
  "current_prices": {
    "NVDA": 177.17,
    "TSM": 284.40,
    "LRCX": 152.49,
    "AMAT": 243.66,
    "MU": 225.59,
    "ASX": 14.13,
    "ASML": 1003.24
  }
}
```

**Returns:**
- Allocation breakdown with dollar amounts and estimated shares
- Step-by-step execution instructions
- Order priority recommendations (largest positions first)
- Fractional share requirements
- Order type recommendations based on position size

## MCP Client Configuration

### Testing the Server

Test that the server works:

```bash
python -m nvidia_mcp.server
```

### For Cursor

**Configuration file location (Windows):**
- `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- Or workspace: `.cursor/mcp.json`

**Setup Steps:**
1. Create or edit the MCP settings file at the location above
2. Add the server configuration with the correct `cwd` path
3. Restart Cursor

**Example configuration:**
```json
{
  "mcpServers": {
    "nvidia-supply-chain": {
      "command": "python",
      "args": ["-m", "nvidia_mcp.server"],
      "cwd": "C:\\DATA\\DEV\\AGENTS\\FINANCE\\nvidia-supply-chain-analysis"
    }
  }
}
```

### For Other MCP Clients

The server uses stdio transport, so it can be configured with any MCP client that supports stdio servers.

## Analysis Types Explained

### Health Analysis
Provides a comprehensive health assessment of the supply chain:
- Health score (0-100)
- Supply chain data for each company
- Correlations with reference ticker
- Momentum indicators (top performers, laggards)
- Risk indicators

### Technical Analysis
Generates trading signals using technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Support and resistance levels
- Pattern detection (breakouts, breakdowns, trends)
- Volume analysis

### Earnings Analysis
Tracks earnings events and their historical impacts:
- Upcoming earnings dates
- Revenue and earnings estimates
- Historical earnings impact on stock price
- Catalyst clusters (multiple earnings in short period)
- Insider sentiment indicators

### Bottlenecks Analysis
Focused analysis on critical supply chain bottlenecks:
- HBM memory suppliers (SK, MU)
- Advanced packaging (TSM)
- EUV equipment (ASML)

### Pairs Analysis
Identifies pair trading opportunities:
- Correlation analysis between pairs
- Spread calculations
- Classic pairs: NVDA/AMD, TSM/INTC, AMAT/LRCX, MU/SK

### Portfolio Analysis
Generates portfolio allocation recommendations:
- Aggressive Growth portfolio
- Balanced portfolio
- Conservative/Hedged portfolio

### Full Analysis
Comprehensive analysis combining all aspects above.

### Trading Instructions
Provides step-by-step instructions for executing trades on Interactive Brokers Mobile App:
- Buy order instructions (multiple methods: watchlist, trade button, search)
- Sell order instructions
- Order type explanations (Market, Limit, Stop, Stop Limit)
- Fractional share trading setup
- Order management (view, modify, cancel)
- Portfolio execution guide (converts allocations to actionable steps)

## Supply Chain Tickers

Default supply chain includes:
- **NVDA**: NVIDIA Corporation
- **TSM**: Taiwan Semiconductor Manufacturing
- **ASML**: ASML Holding
- **AMAT**: Applied Materials
- **LRCX**: Lam Research
- **KLAC**: KLA Corporation
- **MU**: Micron Technology
- **SMCI**: Super Micro Computer
- **DELL**: Dell Technologies
- **HPE**: Hewlett Packard Enterprise
- And more...

## Dependencies

- `mcp>=1.0.0`: Model Context Protocol Python SDK
- `yfinance>=0.2.0`: Yahoo Finance data fetching
- `pandas>=2.0.0`: Data manipulation and analysis
- `numpy>=1.24.0`: Numerical computations
- `scipy>=1.10.0`: Scientific computing (for portfolio optimization)

## Project Structure

The MCP server is organized as follows:
- `nvidia_mcp/server.py`: FastMCP server entry point
- `nvidia_mcp/tools.py`: Tool implementations that route to internal modules
- `nvidia_mcp/internal/`: Internal wrapper modules that import from scripts
- `scripts/`: Core implementation scripts containing the actual analysis logic
  - `fetch_supply_chain_data.py`: Supply chain data fetching and analysis
  - `technical_analysis.py`: Technical indicator calculations
  - `earnings_calendar.py`: Earnings tracking and analysis

The scripts can be used independently as standalone Python scripts or through the MCP server.

## License

See the main project license.

## Support

For issues or questions, please refer to the documentation in the `references/` directory:
- `references/analysis_strategies.md`: Analysis strategies and methodologies
- `references/supply_chain_map.md`: Supply chain relationships and mappings

