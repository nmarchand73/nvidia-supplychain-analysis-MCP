"""
Trading Instructions Module
Provides step-by-step instructions for buying/selling stocks on Interactive Brokers Mobile App
"""
from typing import Dict, Any, Optional
from datetime import datetime


def get_ibkr_mobile_trading_instructions(
    action: str = "both",
    order_type: Optional[str] = None,
    include_fractional: bool = True
) -> Dict[str, Any]:
    """
    Get detailed instructions for trading stocks on Interactive Brokers Mobile App.
    
    Args:
        action: "buy", "sell", or "both" (default: "both")
        order_type: Specific order type to focus on (Market, Limit, Stop, etc.) or None for all
        include_fractional: Whether to include fractional share trading instructions
    
    Returns:
        Dictionary with step-by-step instructions
    """
    instructions = {
        "timestamp": datetime.now().isoformat(),
        "platform": "Interactive Brokers Mobile App (IBKR Mobile)",
        "action": action,
        "order_type_focus": order_type,
        "fractional_shares_enabled": include_fractional
    }
    
    # General Setup
    setup_steps = {
        "download_and_install": {
            "step": 1,
            "title": "Download and Install IBKR Mobile App",
            "instructions": [
                "Download from App Store (iOS) or Google Play Store (Android)",
                "Open the app and log in using your IBKR credentials",
                "Ensure your account is funded and trading permissions are enabled"
            ]
        },
        "enable_fractional_shares": {
            "step": 2,
            "title": "Enable Fractional Share Trading (Optional)",
            "instructions": [
                "Log into Client Portal on web browser (www.interactivebrokers.com/clientportal)",
                "Navigate to Settings > Account Settings",
                "Click gear icon next to 'Trading Experience & Permissions'",
                "Expand the 'Stocks' section",
                "Check 'United States (Trade in Fractions)'",
                "Save changes",
                "Note: Once enabled, you can specify cash amounts instead of share quantities"
            ],
            "required": False
        }
    }
    
    # Buy Order Instructions
    buy_instructions = {
        "title": "How to Buy Stocks on IBKR Mobile App",
        "methods": [
            {
                "method": "From Watchlist or Portfolio",
                "steps": [
                    "Open the IBKR Mobile app",
                    "Navigate to your Watchlist or Portfolio",
                    "Tap twice on the desired stock symbol to open Quote Details page",
                    "Tap the 'Buy' button at the bottom of the screen",
                    "Order ticket will appear"
                ]
            },
            {
                "method": "Using Trade Button",
                "steps": [
                    "Open the IBKR Mobile app",
                    "Tap the center button with two arrows (Trade icon)",
                    "Search for the desired stock symbol using the search function",
                    "Select the stock from results",
                    "Tap 'Buy' button to open the order ticket"
                ]
            },
            {
                "method": "Using Search Function",
                "steps": [
                    "Tap the magnifying glass icon (Search)",
                    "Type the stock symbol (e.g., NVDA, TSM, ASML)",
                    "Select the stock from search results",
                    "Tap 'Buy' button"
                ]
            }
        ],
        "order_configuration": {
            "quantity": {
                "field": "Quantity",
                "instructions": [
                    "Tap the 'Quantity' field",
                    "Enter the number of shares you wish to purchase",
                    "For fractional shares: Enter cash amount (e.g., $500) if fractional trading is enabled",
                    "The app will calculate the number of fractional shares automatically"
                ]
            },
            "order_type": {
                "field": "Order Type",
                "options": [
                    "Market: Executes immediately at current market price",
                    "Limit: Executes only at your specified price or better",
                    "Stop: Becomes market order when price reaches stop level",
                    "Stop Limit: Combines stop and limit orders",
                    "Trailing Stop: Adjusts stop price as stock moves favorably"
                ],
                "instructions": [
                    "Tap the 'Order Type' field",
                    "Select your desired order type from the list",
                    "Market orders execute immediately but may have price slippage",
                    "Limit orders give price control but may not execute if price doesn't reach your limit"
                ]
            },
            "price": {
                "field": "Price",
                "instructions": [
                    "Only required for Limit, Stop, or Stop Limit orders",
                    "Tap the 'Price' field",
                    "Enter your desired price per share",
                    "For Limit orders: This is the maximum price you're willing to pay",
                    "For Stop orders: This is the price that triggers the order"
                ],
                "required_for": ["Limit", "Stop", "Stop Limit"]
            },
            "time_in_force": {
                "field": "Time-in-Force",
                "options": [
                    "Day: Order expires at end of trading day if not filled",
                    "GTC (Good Till Canceled): Order remains active until filled or canceled",
                    "IOC (Immediate or Cancel): Fills immediately or cancels remaining quantity",
                    "FOK (Fill or Kill): Must fill completely immediately or cancels"
                ],
                "instructions": [
                    "Tap the 'Time-in-Force' field",
                    "Select how long you want the order to remain active",
                    "Day orders are recommended for beginners",
                    "GTC orders remain active until you cancel them"
                ]
            }
        },
        "submission": {
            "title": "Review and Submit Order",
            "steps": [
                "Review all order details carefully:",
                "  - Verify stock symbol is correct",
                "  - Check quantity or dollar amount",
                "  - Confirm order type and price (if applicable)",
                "  - Verify time-in-force setting",
                "Check estimated commission and fees (displayed on order ticket)",
                "Slide the 'Submit Order' slider at the bottom of the screen",
                "Confirm the order submission",
                "You'll receive a confirmation message"
            ]
        }
    }
    
    # Sell Order Instructions
    sell_instructions = {
        "title": "How to Sell Stocks on IBKR Mobile App",
        "methods": [
            {
                "method": "From Portfolio",
                "steps": [
                    "Open the IBKR Mobile app",
                    "Navigate to your Portfolio",
                    "Find and tap on the stock you want to sell",
                    "Tap the 'Sell' button",
                    "Order ticket will appear"
                ]
            },
            {
                "method": "Using Trade Button",
                "steps": [
                    "Tap the center Trade button (two arrows icon)",
                    "Search for the stock symbol you own",
                    "Select the stock",
                    "Tap 'Sell' button"
                ]
            }
        ],
        "order_configuration": {
            "quantity": {
                "field": "Quantity",
                "instructions": [
                    "Tap the 'Quantity' field",
                    "Enter the number of shares to sell",
                    "You can sell all shares or a partial position",
                    "The app shows your current position size",
                    "For fractional shares: Enter cash amount or fractional share quantity"
                ]
            },
            "order_type": {
                "field": "Order Type",
                "instructions": [
                    "Same options as buy orders: Market, Limit, Stop, etc.",
                    "Market: Sells immediately at current market price",
                    "Limit: Sells only at your specified price or better",
                    "Stop Loss: Protects against losses by selling when price drops to stop level"
                ]
            },
            "price": {
                "field": "Price",
                "instructions": [
                    "For Limit orders: Minimum price you're willing to accept",
                    "For Stop orders: Price that triggers the sell order",
                    "Enter your desired price per share"
                ]
            },
            "time_in_force": {
                "field": "Time-in-Force",
                "instructions": [
                    "Same options as buy orders",
                    "Day or GTC are most common for sell orders"
                ]
            }
        },
        "submission": {
            "title": "Review and Submit Sell Order",
            "steps": [
                "Review order details:",
                "  - Confirm you're selling the correct stock",
                "  - Verify quantity matches your intention",
                "  - Check order type and price",
                "Review estimated proceeds (after fees)",
                "Slide the 'Submit Order' slider",
                "Confirm submission"
            ]
        }
    }
    
    # Order Management
    order_management = {
        "title": "Monitoring and Managing Orders",
        "view_orders": {
            "steps": [
                "Tap the Menu icon (three horizontal lines) in top-left corner",
                "Select 'Orders & Trades' from the menu",
                "View all active, filled, or canceled orders",
                "Orders are organized by status:",
                "  - Pending: Waiting to be executed",
                "  - Filled: Successfully executed",
                "  - Canceled: Orders you've canceled"
            ]
        },
        "modify_order": {
            "steps": [
                "Navigate to Orders & Trades",
                "Find the active order you want to modify",
                "Tap on the order",
                "Select 'Modify' option",
                "Change quantity, price, order type, or time-in-force",
                "Review changes and confirm modification"
            ]
        },
        "cancel_order": {
            "steps": [
                "Navigate to Orders & Trades",
                "Find the active order you want to cancel",
                "Tap on the order",
                "Select 'Cancel' option",
                "Confirm cancellation",
                "Order will be removed from active orders"
            ]
        }
    }
    
    # Order Type Details
    order_type_details = {
        "Market": {
            "description": "Executes immediately at the best available market price",
            "pros": ["Fast execution", "Guaranteed fill (for liquid stocks)"],
            "cons": ["Price slippage possible", "No price control"],
            "best_for": ["Liquid stocks", "When speed is priority", "Small orders"]
        },
        "Limit": {
            "description": "Executes only at your specified price or better",
            "pros": ["Price control", "No slippage above limit"],
            "cons": ["May not execute if price doesn't reach limit", "Partial fills possible"],
            "best_for": ["Price-sensitive trades", "Large orders", "When you have a target price"]
        },
        "Stop": {
            "description": "Becomes a market order when price reaches stop level",
            "pros": ["Automatic execution at trigger", "Risk management"],
            "cons": ["No price guarantee after trigger", "Gap risk"],
            "best_for": ["Stop loss protection", "Breakout trading"]
        },
        "Stop Limit": {
            "description": "Becomes a limit order when price reaches stop level",
            "pros": ["Price protection after trigger", "Risk management"],
            "cons": ["May not fill if price gaps", "More complex"],
            "best_for": ["Stop loss with price control", "Volatile stocks"]
        }
    }
    
    # Build response based on action parameter
    if action == "buy":
        instructions["steps"] = {
            "setup": setup_steps if include_fractional else {},
            "buy_order": buy_instructions,
            "order_management": order_management,
            "order_types": order_type_details
        }
    elif action == "sell":
        instructions["steps"] = {
            "setup": setup_steps if include_fractional else {},
            "sell_order": sell_instructions,
            "order_management": order_management,
            "order_types": order_type_details
        }
    else:  # both
        instructions["steps"] = {
            "setup": setup_steps if include_fractional else {},
            "buy_order": buy_instructions,
            "sell_order": sell_instructions,
            "order_management": order_management,
            "order_types": order_type_details
        }
    
    # Add specific order type focus if requested
    if order_type and order_type in order_type_details:
        instructions["focused_order_type"] = {
            "type": order_type,
            "details": order_type_details[order_type],
            "specific_instructions": _get_order_type_specific_instructions(order_type)
        }
    
    # Tips and Best Practices
    instructions["tips"] = {
        "general": [
            "Always review order details before submitting",
            "Check market hours - orders placed outside market hours will queue",
            "Monitor your orders in the Orders & Trades section",
            "Use Limit orders for better price control on larger positions",
            "Set stop losses to protect against unexpected moves"
        ],
        "fractional_shares": [
            "Fractional shares allow you to invest exact dollar amounts",
            "Useful for expensive stocks like ASML ($1000+) or BRK.A",
            "Enables precise portfolio allocation (e.g., exactly 30% in TSM)",
            "Fractional shares can be sold just like whole shares"
        ],
        "risk_management": [
            "Never invest more than you can afford to lose",
            "Use stop losses to limit downside risk",
            "Consider position sizing relative to portfolio",
            "Diversify across multiple stocks and sectors",
            "Review and adjust orders regularly"
        ]
    }
    
    return instructions


def _get_order_type_specific_instructions(order_type: str) -> Dict[str, Any]:
    """Get specific instructions for a particular order type."""
    specific = {
        "Market": {
            "when_to_use": "Use Market orders when you need immediate execution and the stock is liquid",
            "price_expectation": "You'll pay the current ask price (when buying) or receive the current bid price (when selling)",
            "execution_time": "Usually executes within seconds during market hours",
            "tips": [
                "Best for stocks with tight bid-ask spreads",
                "Avoid during high volatility periods if price-sensitive",
                "Check the current bid/ask spread before submitting"
            ]
        },
        "Limit": {
            "when_to_use": "Use Limit orders when you have a specific price target and can wait for execution",
            "price_setting": {
                "buying": "Set limit price at or below current ask price for better chance of execution",
                "selling": "Set limit price at or above current bid price for better chance of execution"
            },
            "execution_time": "May take minutes to hours depending on market conditions",
            "tips": [
                "Set limit price slightly better than current market for faster execution",
                "Monitor the order - you may need to adjust if price moves away",
                "Partial fills are possible - remaining quantity stays active"
            ]
        },
        "Stop": {
            "when_to_use": "Use Stop orders for risk management (stop loss) or breakout trading",
            "price_setting": {
                "stop_loss": "Set stop price below current price (when long) to limit losses",
                "breakout": "Set stop price above current price to enter on momentum"
            },
            "execution_time": "Executes as market order when stop price is reached",
            "tips": [
                "Stop orders don't guarantee execution at stop price - slippage possible",
                "Use Stop Limit for more price control",
                "Consider gap risk - stock may gap through your stop price"
            ]
        },
        "Stop Limit": {
            "when_to_use": "Use Stop Limit when you want stop protection but also price control",
            "price_setting": {
                "stop_price": "Price that triggers the order",
                "limit_price": "Maximum (buy) or minimum (sell) price after trigger"
            },
            "execution_time": "Becomes active limit order when stop price is reached",
            "tips": [
                "Set limit price with some buffer from stop price",
                "May not fill if price gaps through your limit",
                "More complex but offers better price protection than Stop orders"
            ]
        }
    }
    
    return specific.get(order_type, {})


def get_portfolio_execution_guide(
    portfolio_allocation: Dict[str, float],
    total_amount: float,
    current_prices: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate step-by-step execution guide for a portfolio allocation.
    
    Args:
        portfolio_allocation: Dictionary mapping tickers to percentage allocations
        total_amount: Total portfolio value in base currency
        current_prices: Optional dictionary of current stock prices
    
    Returns:
        Dictionary with execution instructions
    """
    execution_plan = {
        "timestamp": datetime.now().isoformat(),
        "total_portfolio_value": total_amount,
        "allocation_breakdown": {},
        "execution_steps": [],
        "order_priority": []
    }
    
    # Calculate dollar amounts and share quantities
    for ticker, percentage in portfolio_allocation.items():
        dollar_amount = total_amount * (percentage / 100)
        
        allocation_info = {
            "ticker": ticker,
            "percentage": percentage,
            "dollar_amount": dollar_amount,
            "estimated_shares": None,
            "order_type_recommendation": "Limit" if dollar_amount > 1000 else "Market"
        }
        
        if current_prices and ticker in current_prices:
            price = current_prices[ticker]
            shares = dollar_amount / price
            allocation_info["estimated_shares"] = round(shares, 4)
            allocation_info["current_price"] = price
            allocation_info["fractional_shares_needed"] = shares < 1.0
        
        execution_plan["allocation_breakdown"][ticker] = allocation_info
    
    # Generate execution steps
    execution_plan["execution_steps"] = [
        {
            "step": 1,
            "action": "Enable Fractional Shares",
            "instructions": [
                "Log into IBKR Client Portal on web",
                "Go to Settings > Account Settings > Trading Experience & Permissions",
                "Enable 'United States (Trade in Fractions)'",
                "This allows precise dollar-amount orders"
            ]
        },
        {
            "step": 2,
            "action": "Prepare Order List",
            "instructions": [
                "Open IBKR Mobile app",
                "Have your portfolio allocation ready",
                "Note: Execute larger positions first for better price control"
            ]
        },
        {
            "step": 3,
            "action": "Execute Orders",
            "instructions": [
                "For each stock in your portfolio:",
                "1. Search for the ticker symbol",
                "2. Tap 'Buy' button",
                "3. Enter dollar amount (not share quantity) if fractional enabled",
                "4. Select 'Limit' order type for positions >$1000",
                "5. Set limit price slightly above current ask (buying) for faster fill",
                "6. Set Time-in-Force to 'Day'",
                "7. Review and submit order",
                "8. Monitor execution in Orders & Trades"
            ]
        },
        {
            "step": 4,
            "action": "Verify Positions",
            "instructions": [
                "After all orders are filled:",
                "Navigate to Portfolio in app",
                "Verify each position matches your target allocation",
                "Check that fractional shares are correctly displayed",
                "Review total portfolio value"
            ]
        }
    ]
    
    # Order priority (largest positions first)
    sorted_allocations = sorted(
        execution_plan["allocation_breakdown"].items(),
        key=lambda x: x[1]["dollar_amount"],
        reverse=True
    )
    
    execution_plan["order_priority"] = [
        {
            "priority": i + 1,
            "ticker": ticker,
            "amount": info["dollar_amount"],
            "reason": "Larger positions benefit from limit orders and careful execution"
        }
        for i, (ticker, info) in enumerate(sorted_allocations)
    ]
    
    return execution_plan

