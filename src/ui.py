from .common import *

def prompt_granularity(default=DEFAULT_SCALPING_GRANULARITY):
    """Prompt for a supported candle size in seconds."""
    supported = ", ".join(str(value) for value in SUPPORTED_GRANULARITIES)
    while True:
        raw_value = input(f"Candle size in seconds (default: {default}): ").strip()
        granularity = int(raw_value) if raw_value else default
        if granularity in SUPPORTED_GRANULARITIES:
            return granularity
        print(f"❌ Unsupported candle size: {granularity}")
        print(f"   Supported values: {supported}")


def prompt_account_mode():
    """Ask whether to use a demo or real Deriv account."""
    while True:
        mode = input("\nChoose account type ([d]emo / [r]eal, default: demo): ").strip().lower()
        if not mode or mode in {"d", "demo"}:
            return "demo"
        if mode in {"r", "real"}:
            return "real"
        print("❌ Please enter 'demo' or 'real'.")


def account_matches_mode(account_status, expected_mode):
    """Check whether the authenticated account matches the requested mode."""
    is_demo = bool(account_status.get('is_virtual', False))
    if expected_mode == "demo":
        return is_demo
    return not is_demo


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_header():
    """Print main header"""
    print("\n" + "="*70)
    print(f"                    🤖 AI TRADING BOT - {DEFAULT_DISPLAY_SYMBOL}")
    print("="*70)
    print("   Deep Reinforcement Learning | Auto-Learning | Adaptive Trading")
    print("="*70)


def print_menu():
    """Print main menu"""
    print("\n📋 MAIN MENU")
    print("-"*70)
    print("  TRAINING OPTIONS:")
    print("    1.  Quick Test        (10 episodes, ~2-3 min)")
    print("    2.  Basic Training    (30 episodes, ~10 min)")
    print("    3.  Standard Training (100 episodes, ~30 min)")
    print("    4.  Thorough Training (200 episodes, ~1 hour)")
    print("    5.  Custom Training   (Set your own parameters)")
    print("    6.  Resume Training   (Continue from last checkpoint)")
    print("")
    print("  MODEL MANAGEMENT:")
    print("    7.  Load Specific Model")
    print("    8.  List All Models & Checkpoints")
    print("    9.  Delete Model")
    print("   10.  Model Information")
    print("")
    print("  TESTING & ANALYSIS:")
    print("   11.  Test Model on Historical Data")
    print("   12.  Demo Trading Simulation (Paper Trading)")
    print("   13.  Performance Comparison")
    print("   16.  Learning Benchmark")
    print("")
    print("  LIVE TRADING:")
    print("   14.  Real-Time Trading (Deriv Demo Account)")
    print("   15.  Stop Real-Time Trading")
    print("")
    print("  SYSTEM:")
    print("    0.  Exit")
    print("-"*70)


def print_training_menu():
    """Print custom training menu"""
    print("\n⚙️  CUSTOM TRAINING CONFIGURATION")
    print("-"*70)

