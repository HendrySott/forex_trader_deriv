from . import common
from .common import *
from .agent import DQNAgent
from .checkpoints import CheckpointMixin
from .environment import TradingEnvironment
from .indicators import TechnicalIndicators
from .live_trading import RealtimeTradingMixin
from .market_analysis import MarketAnalysisMixin
from .training import TrainingMixin


class AITradingBot(
    CheckpointMixin,
    MarketAnalysisMixin,
    RealtimeTradingMixin,
    TrainingMixin,
):
    def __init__(self, symbol=DEFAULT_DISPLAY_SYMBOL, initial_balance=10000, lot_size=0.1):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.environment = TradingEnvironment(initial_balance, lot_size)
        self.indicators = TechnicalIndicators()
        
        self.state_size = 10
        self.agent = None
        self.pending_checkpoint_path = None
        
        self.training_complete = False
        self.training_history = []
        self.best_win_rate = 0
        self.best_model_path = None
        self.current_episode = 0
        self.realtime_trading = False
        self.deriv_api = None
        self.current_data = pd.DataFrame()
        self.deriv_api_token = None
        self.market_symbol = DEFAULT_SCALPING_SYMBOL
        self.data_source = "Deriv historical data"
        self.account_balance = float(initial_balance)
        self.account_id = None
        self.account_type = "Unknown"
        self.account_mode = "demo"
        self.order_in_flight = False
        self.order_lock = threading.Lock()
        self.candle_granularity = DEFAULT_SCALPING_GRANULARITY
        self.live_candle = None
        self.last_completed_candle_epoch = None
        self.active_contract_id = None
        self.active_live_trade = None
        self.live_trade_history = []
        self.live_error_blocked = False
        self.live_exit_in_flight = False
        self.live_resale_unavailable = False
        self.position_candles_held = 0
        self.min_profit_target = 0.25
        self.profit_lock_buffer = 0.05
        self.max_loss_tolerance = -0.60
        self.entry_confidence_threshold = 0.18
        self.hold_confidence_threshold = 0.12
        self.strong_trend_threshold = 2.0
        self.max_hold_candles = 5
        self.min_hold_candles = 1
        self.live_min_duration_candles = 1
        self.live_max_duration_candles = 4
        self.min_expected_edge = 0.02
        self.live_duration_fallbacks = (60, 120, 180, 300, 600, 900, 1800)
        self.live_forex_duration_fallbacks = (900, 1800, 3600)
        self.accepted_live_duration_by_symbol = {}
        self.training_small_profit_bonus = 0.0015
        self.training_loss_penalty_scale = 0.0012
        self.training_wait_reward = 0.00015
        self.training_bad_entry_penalty = 0.00035
        self.training_trend_hold_reward = 0.00025
        self.training_stagnation_penalty = 0.0002
        self.training_overhold_penalty = 0.0003
        
        # Create checkpoint directory
        self.checkpoint_dir = "training_checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.latest_checkpoint_file = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")

    def require_model_dependencies(self):
        """Raise a clear error when the ML backend is unavailable."""
        try:
            _load_tensorflow_modules()
            return True
        except ImportError as exc:
            dependency_hint = (
                "TensorFlow is not installed, so training/loading the model cannot run yet. "
                "Install it with `pip install tensorflow` in your project environment."
            )
            if common.TENSORFLOW_IMPORT_ERROR is not None:
                raise RuntimeError(f"{dependency_hint}\nOriginal import error: {common.TENSORFLOW_IMPORT_ERROR}") from exc
            raise RuntimeError(dependency_hint) from exc

    def _ensure_agent(self):
        """Create the agent lazily so the menu still opens without TensorFlow."""
        if self.agent is None:
            self.require_model_dependencies()
            self.agent = DQNAgent(self.state_size)
            if self.pending_checkpoint_path:
                self.agent.load(self.pending_checkpoint_path)
                log_event(f"Loaded checkpoint: {self.pending_checkpoint_path}")
                self.pending_checkpoint_path = None
        return self.agent
