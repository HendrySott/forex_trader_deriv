# AI Trading Bot for Deriv

A modular Python trading bot that uses historical Deriv market data, technical indicators, and a Deep Q-Network agent to train, test, benchmark, and optionally run real-time trading sessions.

> Trading involves risk. This project is for learning, experimentation, and controlled testing. Use demo accounts first and review the code before using any real account.

## Features

- Train a reinforcement learning model on Deriv candle data
- Resume training from saved checkpoints
- Test trained models against historical data
- Run demo/paper trading simulations
- Compare checkpoint performance
- Start and stop real-time Deriv trading from the terminal menu
- Manage checkpoints and training summaries

## Project Structure

```text
.
├── main.py                  # App launcher
├── install_trader_env.sh    # Creates trader env and installs dependencies
├── requirements.txt         # Python dependencies
└── src/
    ├── cli.py               # Main app flow and menu loop
    ├── actions.py           # Menu actions
    ├── bot.py               # AITradingBot core class
    ├── agent.py             # DQN agent
    ├── environment.py       # Trading environment
    ├── training.py          # Training, evaluation, demo trading
    ├── live_trading.py      # Real-time trading logic
    ├── market_analysis.py   # Features, signals, action selection
    ├── checkpoints.py       # Save/load model checkpoints
    ├── deriv_api.py         # Deriv WebSocket API client
    ├── data.py              # Account and market data loading
    ├── indicators.py        # Technical indicators
    ├── ui.py                # Terminal menus and prompts
    └── common.py            # Shared config, imports, logging, helpers
```

## Requirements

- Python 3.10 or newer recommended
- A Deriv API token
- Linux/macOS shell for `install_trader_env.sh`

Python packages are listed in [requirements.txt](requirements.txt):

- `numpy`
- `pandas`
- `requests`
- `websocket-client`
- `tensorflow`

## Installation

Run the installer from the project root:

```bash
./install_trader_env.sh
```

The installer will:

1. Create a virtual environment named `trader`
2. Activate the environment
3. Upgrade pip tooling
4. Install all libraries from `requirements.txt`

If the script is not executable, run:

```bash
chmod +x install_trader_env.sh
./install_trader_env.sh
```

## Running the Bot

Activate the environment and start the app:

```bash
source trader/bin/activate
python main.py
```

The app will ask for:

- Deriv account type: demo or real
- Deriv API token
- Market symbol
- Number of candles to load
- Candle granularity

After setup, the terminal menu lets you train, test, compare, load checkpoints, and manage real-time trading.

## Checkpoints and Logs

Runtime files are created locally:

- `training_checkpoints/` stores model checkpoints and metadata
- `trading_bot.log` stores logs

These files are ignored by Git because they are generated during local use.

## Useful Commands

Compile-check the project:

```bash
python3 -m py_compile main.py src/*.py
```

Run the app:

```bash
source trader/bin/activate
python main.py
```

Deactivate the environment:

```bash
deactivate
```

## Notes

- TensorFlow is loaded lazily, so the app can show dependency errors cleanly.
- Real-time trading requires a valid Deriv token and network access.
- Start with a demo account before using a real account.
