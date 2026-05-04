from .common import *
from .deriv_api import DerivWebSocketAPI

def generate_mock_data(n_candles=3000):
    """Generate realistic mock price data"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_candles, freq='1h')
    
    # Generate realistic price movement with trends
    trend = np.linspace(0, 0.02, n_candles)
    noise = np.random.randn(n_candles) * 0.002
    prices = 1.1000 + trend + np.cumsum(noise) * 0.001
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_candles) * 0.0005),
        'low': prices - np.abs(np.random.randn(n_candles) * 0.0005),
        'close': prices + np.random.randn(n_candles) * 0.0003,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    return df


def load_deriv_market_data(api_token, symbol=DEFAULT_SCALPING_SYMBOL, candles=3000, granularity=DEFAULT_SCALPING_GRANULARITY):
    """Load historical candle data from Deriv for training and testing."""
    if granularity not in SUPPORTED_GRANULARITIES:
        supported = ", ".join(str(value) for value in SUPPORTED_GRANULARITIES)
        raise ValueError(
            f"Unsupported candle size: {granularity}. Use one of: {supported}"
        )
    api = DerivWebSocketAPI(app_id="1089", demo=True)
    log_event(f"Loading {candles} candles from Deriv for {symbol}...")
    data = api.fetch_historical_candles(
        symbol=symbol,
        granularity=granularity,
        count=candles,
        api_token=api_token
    )
    if len(data) < 100:
        raise RuntimeError(f"Deriv returned only {len(data)} candles, which is too little to train safely.")
    log_event(f"Loaded {len(data)} real candles from Deriv for {symbol}.")
    return data


def load_deriv_account_status(api_token):
    """Load the current Deriv account info tied to the provided token."""
    api = DerivWebSocketAPI(app_id="1089", demo=True)
    account = api.fetch_account_status(api_token)
    return {
        'balance': float(account.get('balance', 0.0)),
        'account_id': account.get('loginid'),
        'currency': account.get('currency', 'USD'),
        'is_virtual': bool(account.get('is_virtual', False))
    }

