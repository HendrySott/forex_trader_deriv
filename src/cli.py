from .common import *
from .actions import (
    custom_training,
    delete_model,
    demo_trading,
    learning_benchmark,
    list_checkpoints_detailed,
    load_specific_model,
    performance_comparison,
    realtime_trading,
    resume_training,
    test_model,
)
from .bot import AITradingBot
from .data import load_deriv_account_status, load_deriv_market_data
from .ui import (
    account_matches_mode,
    clear_screen,
    print_header,
    print_menu,
    prompt_account_mode,
    prompt_granularity,
)

def main():
    """Main execution with complete menu system"""
    
    # Initialize bot
    bot = AITradingBot(initial_balance=10000, lot_size=0.1)
    latest_checkpoint = bot.load_latest_checkpoint(load_model=False)
    if latest_checkpoint:
        log_event(f"Auto-restored training metadata from: {latest_checkpoint}")

    print("\n🔐 DERIV ACCOUNT DATA SETUP")
    print("="*70)
    print("This version uses real historical Deriv data for training/testing.")
    print("It will also load the current balance from the Deriv account tied to your token.")
    print("="*70)

    selected_mode = prompt_account_mode()
    api_token = input(f"\nEnter your Deriv {selected_mode} API token: ").strip()
    if not api_token:
        print("❌ A Deriv API token is required for main.py")
        return

    symbol = prompt_asset_symbol(bot.market_symbol)
    candle_count = input("Number of candles to load (default: 3000): ").strip()
    candle_count = int(candle_count) if candle_count else 3000
    granularity = prompt_granularity(DEFAULT_SCALPING_GRANULARITY)

    try:
        print("\n💰 Loading account status from Deriv...")
        account_status = load_deriv_account_status(api_token)
        print("\n📊 Loading historical market data from Deriv...")
        data = load_deriv_market_data(
            api_token=api_token,
            symbol=symbol,
            candles=candle_count,
            granularity=granularity
        )
    except Exception as e:
        print(f"❌ Failed to load Deriv account or historical data: {e}")
        logger.error("Failed to load Deriv account or historical data: %s", e, exc_info=True)
        return

    if not account_matches_mode(account_status, selected_mode):
        actual_mode = "demo" if account_status['is_virtual'] else "real"
        print(f"❌ Token/account mismatch: you selected '{selected_mode}' but this token belongs to a '{actual_mode}' account.")
        return

    split_index = int(len(data) * 0.8)
    train_data = data[:split_index].reset_index(drop=True)
    test_data = data[split_index:].reset_index(drop=True)
    bot.deriv_api_token = api_token
    bot.market_symbol = symbol
    bot.data_source = f"Deriv {symbol} ({granularity}s candles)"
    bot.candle_granularity = granularity
    bot.account_balance = account_status['balance']
    bot.account_id = account_status['account_id']
    bot.account_type = "Demo" if account_status['is_virtual'] else "Real"
    bot.account_mode = selected_mode
    bot.initial_balance = bot.account_balance
    bot.environment.initial_balance = bot.account_balance
    bot.environment.balance_scale = max(abs(bot.account_balance), 1.0)
    bot.environment.balance = bot.account_balance
    print(f"✅ Data ready: {len(train_data)} training, {len(test_data)} test samples from {bot.data_source}")
    print(f"✅ Account: {bot.account_type} | ID: {bot.account_id} | Balance: {account_status['currency']} {bot.account_balance:,.2f}")
    
    running = True
    while running:
        clear_screen()
        print_header()
        
        # Show current status
        print(f"\n📊 CURRENT STATUS:")
        print(f"   Model: {'✅ Loaded' if bot.training_complete else '❌ Not trained'}")
        print(f"   Account: {bot.account_type} ({bot.account_id or 'N/A'})")
        print(f"   Deriv Balance: ${bot.account_balance:,.2f}")
        print(f"   Bot Balance: ${bot.environment.balance:,.2f}")
        print(f"   Best Win Rate: {bot.best_win_rate:.1f}%")
        epsilon_display = f"{bot.agent.epsilon:.3f}" if bot.agent is not None else "N/A"
        print(f"   Epsilon: {epsilon_display}")
        print(f"   Total Trades: {len(bot.environment.trades)}")
        print(f"   Data Source: {bot.data_source}")
        print(f"   Real-Time Trading: {'🟢 Active' if bot.realtime_trading else '⚫ Inactive'}")
        
        print_menu()
        
        choice = input("\n👉 Enter your choice (0-16): ")
        
        if choice == "0":
            if bot.realtime_trading:
                bot.stop_realtime_trading()
            print("\n👋 Thank you for using AI Trading Bot!")
            print("Happy trading! 🚀")
            running = False
            
        elif choice == "1":
            print("\n🚀 Quick Test (10 episodes)...")
            bot.train(train_data, episodes=10, save_every=5, test_data=test_data[:500])
            input("\nPress Enter to continue...")
            
        elif choice == "2":
            print("\n🚀 Basic Training (30 episodes)...")
            bot.train(train_data, episodes=30, save_every=10, test_data=test_data[:500])
            input("\nPress Enter to continue...")
            
        elif choice == "3":
            print("\n🚀 Standard Training (100 episodes)...")
            bot.train(train_data, episodes=100, save_every=20, test_data=test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            print("\n🚀 Thorough Training (200 episodes)...")
            bot.train(train_data, episodes=200, save_every=25, test_data=test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            custom_training(bot, train_data, test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "6":
            resume_training(bot, train_data, test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "7":
            load_specific_model(bot)
            input("\nPress Enter to continue...")
            
        elif choice == "8":
            list_checkpoints_detailed(bot)
            input("\nPress Enter to continue...")
            
        elif choice == "9":
            delete_model(bot)
            input("\nPress Enter to continue...")
            
        elif choice == "10":
            bot.print_model_info()
            input("\nPress Enter to continue...")
            
        elif choice == "11":
            test_model(bot, test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "12":
            demo_trading(bot, test_data)
            input("\nPress Enter to continue...")
            
        elif choice == "13":
            performance_comparison(bot, test_data)
            input("\nPress Enter to continue...")

        elif choice == "16":
            learning_benchmark(bot, data)
            input("\nPress Enter to continue...")
            
        elif choice == "14":
            if bot.realtime_trading:
                print("\n⚠️ Real-time trading is already active!")
            else:
                realtime_trading(bot)
            input("\nPress Enter to continue...")
            
        elif choice == "15":
            if bot.realtime_trading:
                bot.stop_realtime_trading()
            else:
                print("\n⚠️ No active real-time trading session")
            input("\nPress Enter to continue...")
            
        else:
            print("\n❌ Invalid option. Please try again.")
            time.sleep(1)

