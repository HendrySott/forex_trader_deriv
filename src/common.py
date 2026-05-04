import numpy as np
import pandas as pd
from collections import deque
import random
import time
import json
from datetime import datetime, timedelta
import logging
import os
import sys
import pickle
import threading
import requests
import websocket
import ssl
import importlib

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

tf = None
layers = None
models = None
TENSORFLOW_AVAILABLE = None
TENSORFLOW_IMPORT_ERROR = None


def _load_tensorflow_modules():
    """Import TensorFlow lazily so the CLI can start without ML startup cost."""
    global tf, layers, models, TENSORFLOW_AVAILABLE, TENSORFLOW_IMPORT_ERROR
    if tf is not None and layers is not None and models is not None:
        return tf, layers, models

    try:
        tf = importlib.import_module("tensorflow")
        keras_module = importlib.import_module("tensorflow.keras")
        layers = keras_module.layers
        models = keras_module.models
        TENSORFLOW_AVAILABLE = True
        TENSORFLOW_IMPORT_ERROR = None
        return tf, layers, models
    except ImportError as exc:
        tf = None
        layers = None
        models = None
        TENSORFLOW_AVAILABLE = False
        TENSORFLOW_IMPORT_ERROR = exc
        raise

# Configure logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'trading_bot.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger(__name__)

SUPPORTED_GRANULARITIES = (
    60, 120, 180, 300, 600, 900, 1800,
    3600, 7200, 14400, 28800, 86400
)
DEFAULT_SCALPING_GRANULARITY = 60
DEFAULT_SCALPING_SYMBOL = "frxEURUSD"
DEFAULT_DISPLAY_SYMBOL = "EUR/USD"
ASSET_CHOICES = [
    {"label": "EUR/USD", "symbol": "frxEURUSD", "category": "Forex"},
    {"label": "GBP/USD", "symbol": "frxGBPUSD", "category": "Forex"},
    {"label": "USD/JPY", "symbol": "frxUSDJPY", "category": "Forex"},
    {"label": "AUD/USD", "symbol": "frxAUDUSD", "category": "Forex"},
    {"label": "USD/CAD", "symbol": "frxUSDCAD", "category": "Forex"},
    {"label": "USD/CHF", "symbol": "frxUSDCHF", "category": "Forex"},
    {"label": "NZD/USD", "symbol": "frxNZDUSD", "category": "Forex"},
    {"label": "Gold/USD", "symbol": "frxXAUUSD", "category": "Metal"},
    {"label": "Silver/USD", "symbol": "frxXAGUSD", "category": "Metal"},
]


def log_event(message, level=logging.INFO):
    """Print to the console and persist the same message to the log file."""
    print(message)
    logger.log(level, message)


def is_forex_symbol(symbol):
    """Detect classic forex symbols in Deriv naming."""
    return str(symbol).startswith("frx")


def normalize_deriv_symbol(symbol):
    """Keep Deriv symbols in canonical form, e.g. frxEURUSD."""
    raw_symbol = str(symbol).strip()
    if not raw_symbol:
        return raw_symbol
    if raw_symbol.lower().startswith("frx"):
        return "frx" + raw_symbol[3:].upper()
    return raw_symbol


def get_asset_display_name(symbol):
    """Return a friendly asset label for a Deriv symbol when available."""
    normalized = normalize_deriv_symbol(symbol).upper()
    for asset in ASSET_CHOICES:
        if asset["symbol"].upper() == normalized:
            return asset["label"]
    return symbol


def prompt_asset_symbol(default_symbol=DEFAULT_SCALPING_SYMBOL):
    """Prompt the user to pick a known asset or type a custom Deriv symbol."""
    print("\n📈 AVAILABLE ASSETS")
    print("-"*70)
    for index, asset in enumerate(ASSET_CHOICES, 1):
        default_tag = " (default)" if asset["symbol"] == default_symbol else ""
        print(f"  {index:>2}. {asset['label']:<10} {asset['symbol']:<12} {asset['category']}{default_tag}")
    print("-"*70)
    print("Press Enter to use the default, type a number to select,")
    print("or enter a custom Deriv symbol such as frxEURUSD.")

    alias_map = {}
    for asset in ASSET_CHOICES:
        alias_map[asset["label"].upper()] = asset["symbol"]
        alias_map[asset["label"].replace("/", "").upper()] = asset["symbol"]
        alias_map[asset["symbol"].upper()] = asset["symbol"]
        if "/" in asset["label"]:
            base, quote = asset["label"].split("/", 1)
            alias_map[f"{quote}/{base}".upper()] = asset["symbol"]
            alias_map[f"{quote}{base}".upper()] = asset["symbol"]

    while True:
        raw_value = input(
            f"Select asset (default: {get_asset_display_name(default_symbol)} / {default_symbol}): "
        ).strip()
        if not raw_value:
            return default_symbol
        if raw_value.isdigit():
            selection = int(raw_value)
            if 1 <= selection <= len(ASSET_CHOICES):
                return ASSET_CHOICES[selection - 1]["symbol"]
            print(f"❌ Invalid selection: {raw_value}")
            continue

        normalized = raw_value.replace(" ", "").upper()
        if normalized in alias_map:
            return alias_map[normalized]
        if normalized.startswith("FRX"):
            return normalize_deriv_symbol(normalized)

        print(f"❌ Unknown asset: {raw_value}")
        print("   Choose a number from the list or enter a Deriv symbol like frxXAUUSD.")

logger.info("Logging initialized. Log file: %s", LOG_FILE)



def calculate_return_pct(final_balance, starting_balance):
    """Safely calculate percentage return when starting balance may be zero."""
    if abs(starting_balance) < 1e-9:
        return 0.0
    return ((final_balance - starting_balance) / starting_balance) * 100

