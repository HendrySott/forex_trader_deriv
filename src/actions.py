from .common import *
from .bot import AITradingBot
from .data import load_deriv_account_status
from .ui import print_training_menu, prompt_granularity

def list_checkpoints_detailed(bot):
    """List all checkpoints with details"""
    checkpoints = bot.get_checkpoint_list()
    
    if not checkpoints:
        print("\n❌ No checkpoints found.")
        return []
    
    print("\n📁 AVAILABLE MODELS & CHECKPOINTS")
    print("="*70)
    print(f"{'#':<4} {'Model Name':<40} {'Size':<10} {'Episodes':<10} {'Date':<20}")
    print("-"*70)
    
    for i, cp in enumerate(checkpoints[:20], 1):
        episode_str = f"Ep {cp['episode']}" if cp['episode'] != 'unknown' else "Unknown"
        print(f"{i:<4} {cp['file'][:40]:<40} {cp['size_kb']:.1f}KB   {episode_str:<10} "
              f"{cp['modified'].strftime('%Y-%m-%d %H:%M')}")
    
    print("="*70)
    return checkpoints


def delete_model(bot):
    """Delete a model"""
    checkpoints = list_checkpoints_detailed(bot)
    if not checkpoints:
        return
    
    try:
        choice = input("\nSelect model to delete (number) or 0 to cancel: ")
        if choice == "0":
            return
        
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            confirm = input(f"Delete {checkpoints[idx]['file']}? (yes/no): ")
            if confirm.lower() == 'yes':
                os.remove(checkpoints[idx]['path'])
                print(f"✅ Deleted: {checkpoints[idx]['file']}")
                
                base_name = checkpoints[idx]['file'].replace('.h5', '')
                for f in os.listdir(bot.checkpoint_dir):
                    if f.startswith(base_name):
                        try:
                            os.remove(os.path.join(bot.checkpoint_dir, f))
                        except:
                            pass
        else:
            print("❌ Invalid selection")
    except Exception as e:
        print(f"❌ Error: {e}")


def custom_training(bot, train_data, test_data):
    """Custom training with user parameters"""
    print_training_menu()
    
    try:
        print("\nEnter training parameters (press Enter for defaults):")
        episodes = input("Number of episodes (default: 50): ").strip()
        episodes = int(episodes) if episodes else 50
        
        save_every = input("Save checkpoint every N episodes (default: 10): ").strip()
        save_every = int(save_every) if save_every else 10
        
        eval_choice = input("Enable periodic evaluation? (yes/no, default: yes): ").strip().lower()
        use_evaluation = eval_choice != 'no'
        
        print(f"\n🚀 Starting custom training:")
        print(f"   Episodes: {episodes}")
        print(f"   Save every: {save_every}")
        print(f"   Evaluation: {'Yes' if use_evaluation else 'No'}")
        
        if use_evaluation:
            bot.train(train_data, episodes=episodes, save_every=save_every, test_data=test_data)
        else:
            bot.train(train_data, episodes=episodes, save_every=save_every, test_data=None)
        
        print("\n✅ Custom training completed!")
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        print("Using default values...")
        bot.train(train_data, episodes=50, save_every=10, test_data=test_data)


def resume_training(bot, train_data, test_data):
    """Resume training from last checkpoint"""
    checkpoints = bot.get_checkpoint_list()
    
    if not checkpoints:
        print("\n❌ No checkpoints found to resume from.")
        return False
    
    print("\n📂 Available checkpoints (most recent first):")
    for i, cp in enumerate(checkpoints[:10], 1):
        print(f"  {i}. {cp['file'][:50]} - {cp['modified'].strftime('%Y-%m-%d %H:%M')} - {cp['size_kb']:.1f}KB")
    
    try:
        choice = input("\nSelect checkpoint to resume from (number) or 0 to cancel: ")
        if choice == "0":
            return False
        
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            bot.load_checkpoint(checkpoints[idx]['path'])
            
            additional = input("\nAdditional episodes to train? (default: 50): ").strip()
            additional = int(additional) if additional else 50
            
            save_every = input("Save every N episodes (default: 10): ").strip()
            save_every = int(save_every) if save_every else 10
            
            print(f"\n🚀 Resuming training for {additional} episodes...")
            bot.train(train_data, episodes=additional, save_every=save_every, test_data=test_data)
            return True
        else:
            print("❌ Invalid selection")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def load_specific_model(bot):
    """Load a specific model"""
    checkpoints = bot.get_checkpoint_list()
    
    if not checkpoints:
        print("\n❌ No models found.")
        return False
    
    print("\n📂 Available models:")
    for i, cp in enumerate(checkpoints[:15], 1):
        print(f"  {i}. {cp['file'][:50]} - {cp['size_kb']:.1f}KB - {cp['modified'].strftime('%Y-%m-%d %H:%M')}")
    
    try:
        choice = input("\nSelect model to load (number) or 0 to cancel: ")
        if choice == "0":
            return False
        
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            bot.load_checkpoint(checkpoints[idx]['path'])
            print(f"\n✅ Model loaded successfully!")
            print(f"   File: {checkpoints[idx]['file']}")
            print(f"   Size: {checkpoints[idx]['size_kb']:.1f}KB")
            print(f"   Date: {checkpoints[idx]['modified'].strftime('%Y-%m-%d %H:%M')}")
            return True
        else:
            print("❌ Invalid selection")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model(bot, test_data):
    """Test current model on historical data"""
    if not bot.training_complete:
        print("\n⚠️  No model loaded. Please load or train a model first.")
        return
    
    print("\n🧪 TESTING MODEL ON HISTORICAL DATA")
    print("="*60)
    print(f"Testing on {len(test_data)} candles...")
    
    original_state = bot.environment.snapshot()
    stats = bot.trade(test_data)
    
    keep = input("\nKeep these test results? (yes/no, default: no): ").strip().lower()
    if keep != 'yes':
        bot.environment.restore(original_state)
        print("✅ Reverted to previous state")
    else:
        print("✅ Test results saved")


def demo_trading(bot, test_data):
    """Demo trading simulation"""
    if not bot.training_complete:
        print("\n⚠️  No model loaded. Please load or train a model first.")
        return
    
    bot.start_demo_trading(test_data)


def performance_comparison(bot, test_data):
    """Compare performance of different models"""
    checkpoints = bot.get_checkpoint_list()
    
    if len(checkpoints) < 2:
        print(f"\n⚠️  Need at least 2 models to compare (found {len(checkpoints)})")
        return
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("="*70)
    print("Select up to 3 models to compare:")
    
    for i, cp in enumerate(checkpoints[:10], 1):
        print(f"  {i}. {cp['file'][:50]} - {cp['modified'].strftime('%Y-%m-%d %H:%M')}")
    
    try:
        selections = input("\nEnter model numbers separated by commas (e.g., 1,3,5): ")
        indices = [int(x.strip()) - 1 for x in selections.split(',')]
        
        results = []
        original_state = bot.environment.snapshot()
        original_balance = original_state['balance']
        
        for idx in indices[:3]:
            if 0 <= idx < len(checkpoints):
                print(f"\n{'='*60}")
                print(f"Testing: {checkpoints[idx]['file'][:50]}")
                print(f"{'='*60}")
                
                bot.load_checkpoint(checkpoints[idx]['path'])
                bot.environment.balance = original_balance
                bot.environment.trades = []
                bot.environment.position = 0
                
                stats = bot.trade(test_data, verbose=False)
                
                results.append({
                    'name': checkpoints[idx]['file'][:40],
                    'final_balance': bot.environment.balance,
                    'return_pct': ((bot.environment.balance - original_balance) / original_balance) * 100,
                    'win_rate': stats['win_rate'],
                    'total_trades': stats['total_trades'],
                    'total_pnl': stats['total_pnl']
                })
                
                print(f"Return: {results[-1]['return_pct']:+.2f}% | Win Rate: {results[-1]['win_rate']:.1f}% | PnL: ${results[-1]['total_pnl']:,.2f}")
                bot.environment.restore(original_state)
        
        if results:
            print("\n" + "="*90)
            print("📊 COMPARISON RESULTS")
            print("="*90)
            print(f"{'Model':<45} {'Return':<12} {'Win Rate':<12} {'Trades':<8} {'PnL':<15}")
            print("-"*90)
            
            for r in results:
                print(f"{r['name'][:44]:<45} {r['return_pct']:>+8.2f}%   {r['win_rate']:>6.1f}%    "
                      f"{r['total_trades']:>6}   ${r['total_pnl']:>12,.2f}")
            
            print("="*90)
            
            best_return = max(results, key=lambda x: x['return_pct'])
            best_winrate = max(results, key=lambda x: x['win_rate'])
            
            print(f"\n🏆 BEST BY RETURN: {best_return['name']}")
            print(f"   Return: {best_return['return_pct']:+.2f}%")
            print(f"   Win Rate: {best_return['win_rate']:.1f}%")
            
            print(f"\n🏆 BEST BY WIN RATE: {best_winrate['name']}")
            print(f"   Win Rate: {best_winrate['win_rate']:.1f}%")
            print(f"   Return: {best_winrate['return_pct']:+.2f}%")
        
        if checkpoints:
            bot.load_checkpoint(checkpoints[0]['path'])
            bot.environment.restore(original_state)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def learning_benchmark(bot, market_data):
    """Run a quick before/after training benchmark."""
    print("\n🧪 LEARNING BENCHMARK")
    print("="*70)
    print("This compares model performance before and after a short training run.")
    print("It uses the current Deriv historical dataset and a temporary checkpoint folder.")
    print("="*70)

    try:
        bot._ensure_agent()
    except Exception as e:
        print(f"❌ Benchmark unavailable: {e}")
        return

    benchmark_bot = AITradingBot(initial_balance=10000, lot_size=bot.lot_size)
    benchmark_bot.checkpoint_dir = os.path.join("training_checkpoints", "benchmark_runtime")
    os.makedirs(benchmark_bot.checkpoint_dir, exist_ok=True)
    benchmark_bot.latest_checkpoint_file = os.path.join(
        benchmark_bot.checkpoint_dir, "latest_checkpoint.json"
    )

    dataset = market_data.copy().reset_index(drop=True)
    if len(dataset) < 200:
        print("❌ Need at least 200 Deriv candles loaded for the benchmark.")
        return
    split_index = max(120, int(len(dataset) * 0.7))
    train_data = dataset[:split_index].reset_index(drop=True)
    test_data = dataset[split_index:].reset_index(drop=True)

    baseline_stats = benchmark_bot.trade(test_data, verbose=False)
    baseline = {
        'balance': benchmark_bot.environment.balance,
        'win_rate': baseline_stats['win_rate'],
        'trades': baseline_stats['total_trades'],
        'pnl': baseline_stats['total_pnl']
    }

    benchmark_bot.train(train_data, episodes=1, save_every=1, test_data=test_data, verbose=False)
    trained_stats = benchmark_bot.trade(test_data, verbose=False)
    trained = {
        'balance': benchmark_bot.environment.balance,
        'win_rate': trained_stats['win_rate'],
        'trades': trained_stats['total_trades'],
        'pnl': trained_stats['total_pnl']
    }

    print("\nRESULTS")
    print("-"*70)
    print(f"{'Stage':<12} {'Balance':<15} {'Win Rate':<12} {'Trades':<8} {'PnL':<12}")
    print("-"*70)
    print(f"{'Before':<12} ${baseline['balance']:<14.2f} {baseline['win_rate']:<11.1f}% {baseline['trades']:<8} ${baseline['pnl']:<11.2f}")
    print(f"{'After':<12} ${trained['balance']:<14.2f} {trained['win_rate']:<11.1f}% {trained['trades']:<8} ${trained['pnl']:<11.2f}")
    print("-"*70)

    if trained['balance'] > baseline['balance']:
        print("✅ Short training improved this benchmark run.")
    elif trained['balance'] < baseline['balance']:
        print("⚠️ Short training did not improve this benchmark run.")
    else:
        print("ℹ️ Short training produced no balance change in this benchmark run.")

    print("Use this as a quick sanity check, not proof of long-term profitability.")


def realtime_trading(bot):
    """Start real-time trading on Deriv"""
    if not bot.training_complete:
        print("\n⚠️  No model loaded. Please load or train a model first.")
        return
    
    print(f"\n🔐 DERIV {bot.account_mode.upper()} ACCOUNT SETUP")
    print("="*60)
    print("To get your API token:")
    print("1. Go to https://app.deriv.com/account/api-token")
    print("2. Log in to your Deriv account")
    print(f"3. Generate a new token for your {bot.account_mode} account")
    print("4. Copy the token")
    print("="*60)
    
    default_note = "Press Enter to reuse the startup token" if bot.deriv_api_token else "API token required"
    api_token = input(f"\nEnter your Deriv API token ({default_note}): ").strip()
    if not api_token:
        api_token = bot.deriv_api_token
    
    if not api_token:
        print("❌ No API token provided")
        return
    
    print("\n📊 Trading Parameters:")
    print(
        f"Current default market: {get_asset_display_name(bot.market_symbol)} "
        f"({bot.market_symbol})"
    )
    symbol = prompt_asset_symbol(bot.market_symbol)
    amount = input("Amount per trade in USD (default: 10): ").strip()
    amount = float(amount) if amount else 10
    bot.candle_granularity = prompt_granularity(bot.candle_granularity or DEFAULT_SCALPING_GRANULARITY)

    print(
        f"Scalping setup: {bot.candle_granularity}s candles and "
        f"{bot.candle_granularity}s contract duration."
    )
    if symbol.startswith("frx"):
        print("⚠️ Forex short-duration contracts can be blocked by Deriv during some hours.")
        print("   If you want uninterrupted scalping, use a synthetic market symbol instead of forex.")
    
    bot.start_realtime_trading(api_token, symbol, amount)
