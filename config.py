"""
常量和配置
"""


DEFAULT_CONFIG = {
    "yfinance_interval": "15m", # “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”
    "funding_rate": 1 # 资金费率
}

# AlphaVantage API Ticker 映射表
ALPHAVANTAGE_TICKER_MAP = {
    # ===== 股票 (US Stocks) =====
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "Nvidia": "NVDA",

    # ===== 加密货币 (Crypto) =====
    "BTC": "CRYPTO:BTC",
    "Bitcoin": "CRYPTO:BTC",
    "Ethereum": "CRYPTO:ETH",
    "Solana": "CRYPTO:SOL",
    "Ripple": "CRYPTO:XRP",
    "Cardano": "CRYPTO:ADA",

    # ===== 外汇 (Forex) =====
    "USD": "FOREX:USD",
    "EUR": "FOREX:EUR",
    "JPY": "FOREX:JPY",
    "GBP": "FOREX:GBP",
    "CNY": "FOREX:CNY",
}

# Yfinance Ticker 映射表
YFINANCE_TICKER_MAP = {
    # ===== 股票 (US Stocks) =====
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "Nvidia": "NVDA",

    # ===== 加密货币 (Crypto) =====
    "bitcoin": "BTC-USD",
    "BTC": "BTC-USD",
    "Bitcoin": "BTC-USD",
    "ETH": "ETH-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Ripple": "RIPPLE-USD",
    "Cardano": "ADA-USD",

    # ===== 外汇 (Forex) =====
    "USD": "FOREX:USD",
    "EUR": "FOREX:EUR",
    "JPY": "FOREX:JPY",
    "GBP": "FOREX:GBP",
    "CNY": "FOREX:CNY",
}

OKX_SYMBOL_MAP = {
    # 现货（spot）
    "spot": {
        "BTC": "BTC/USDT",
        "ETH": "ETH/USDT",
    },
    # 永续合约（Perpetual Swap）
    "perp": {
        "BTC": "BTC/USDT:USDT",
        "ETH": "ETH/USDT:USDT",
    },
    # 币本位永续
    "coin_margined": {
        "BTC": "BTC/USD:BTC",
    },
}

BACKTEST_CONFIG = {
    'start_date': '2025-07-01',
    'end_date': '2025-08-01',
    'initial_capital': 10000.0,
    'symbols': ['BTC'],
    'trade_time': '11:00',
    'interval': '15m',
    'leverage': {'BTC': 3},
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.01,
    'capital_allocation': {'BTC': 1.0},
    'consecutive_loss_limit': 4,
    'pause_days': 7,
}
