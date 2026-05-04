from .common import *


class TrainingMixin:
    def train_episode(self, data, episode_num):
        """Train a single episode"""
        self._ensure_agent()
        self.environment.reset()
        self.position_candles_held = 0
        total_reward = 0
        trades_count = 0
        start_balance = self.environment.balance
        features_matrix, valid_mask = self.build_feature_matrix(data)
        
        for i in range(len(data) - 50):
            current_index = i + 50
            next_index = i + 51

            if not valid_mask[current_index]:
                continue
            if self.environment.position != 0:
                self.position_candles_held += 1
            action, decision = self.select_managed_action(data, current_index, training=True)
            if action is None:
                continue
            features = decision['features']

            if next_index >= len(data) or not valid_mask[next_index]:
                continue
            next_features = features_matrix[next_index]
            
            _, reward, done = self.environment.step(
                action, data['close'].iloc[current_index], features
            )
            reward = self._shape_training_reward(
                action,
                float(data['close'].iloc[current_index]),
                decision,
                reward
            )

            if self.environment.position == 0:
                self.position_candles_held = 0
            
            total_reward += reward
            
            if reward != 0:
                trades_count += 1
            
            self.agent.remember(features, action, reward, next_features, done)
            self.agent.replay()
            
            if done:
                break
        
        if episode_num % 5 == 0:
            self.agent.update_target_model()
        
        stats = self.environment.get_statistics()
        episode_data = {
            'episode': episode_num,
            'profit': self.environment.balance - start_balance,
            'win_rate': stats['win_rate'],
            'trades': stats['total_trades'],
            'epsilon': self.agent.epsilon,
            'balance': self.environment.balance,
            'total_pnl': stats['total_pnl']
        }
        self.training_history.append(episode_data)
        
        return episode_data
    
    def train(self, data, episodes=50, save_every=10, test_data=None, verbose=True):
        """Main training function"""
        self._ensure_agent()
        self.current_episode = max(self.current_episode, len(self.training_history))
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 STARTING TRAINING")
            print(f"{'='*70}")
            print(f"Episodes to train: {episodes}")
            print(f"Save every: {save_every} episodes")
            print(f"Current epsilon: {self.agent.epsilon:.3f}")
            print(f"{'='*70}\n")
        
        try:
            for episode in range(1, episodes + 1):
                current_ep = self.current_episode + episode
                episode_data = self.train_episode(data, current_ep)
                
                if verbose and episode % 5 == 0:
                    print(f"Episode {current_ep:3d}: Profit=${episode_data['profit']:8.2f} | "
                          f"Win Rate={episode_data['win_rate']:5.1f}% | "
                          f"Trades={episode_data['trades']:3d} | "
                          f"Epsilon={self.agent.epsilon:.3f}")
                
                if episode % save_every == 0:
                    self.save_checkpoint(current_ep)
                
                if test_data is not None and episode % 20 == 0:
                    win_rate = self.evaluate(test_data[:500], verbose=False)
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        best_path = f"{self.checkpoint_dir}/best_model_{current_ep}ep_win{win_rate:.0f}.weights.h5"
                        self.agent.save(best_path)
                        self.best_model_path = best_path
                        if verbose:
                            print(f"  🎯 NEW BEST! Win rate: {win_rate:.1f}%")
        except KeyboardInterrupt:
            last_episode = self.training_history[-1]['episode'] if self.training_history else self.current_episode
            self.save_checkpoint(last_episode)
            self.training_complete = bool(self.training_history)
            log_event(f"Training interrupted. Progress saved at episode {last_episode}.", logging.WARNING)
            raise
        
        final_episode = self.training_history[-1]['episode'] if self.training_history else self.current_episode
        self.save_checkpoint(final_episode)
        self.training_complete = True
        
        if verbose:
            print(f"\n✅ TRAINING COMPLETED!")
            print(f"Best win rate: {self.best_win_rate:.1f}%")
            print(f"Best model: {self.best_model_path}")
        
        self.save_training_summary()
        return self.best_win_rate
    
    def evaluate(self, test_data, verbose=False):
        """Evaluate model on test data"""
        original_state = self.environment.snapshot()
        stats = self.trade(test_data, verbose=verbose)
        win_rate = stats['win_rate']
        self.environment.restore(original_state)
        
        return win_rate
    
    def trade(self, data, verbose=True):
        """Execute trades on data"""
        self._ensure_agent()
        self.environment.reset()
        self.position_candles_held = 0
        initial_balance = self.environment.balance
        _, valid_mask = self.build_feature_matrix(data)
        
        for i in range(len(data) - 50):
            current_index = i + 50
            if not valid_mask[current_index]:
                continue
            if self.environment.position != 0:
                self.position_candles_held += 1
            action, decision = self.select_managed_action(data, current_index, training=False)
            if action is None:
                continue
            features = decision['features']
            _, _, _ = self.environment.step(action, data['close'].iloc[current_index], features)
            if self.environment.position == 0:
                self.position_candles_held = 0

        if len(data) > 0:
            self.environment.force_close(data['close'].iloc[-1])
            self.position_candles_held = 0
        
        stats = self.environment.get_statistics()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 TRADING RESULTS")
            print(f"{'='*60}")
            print(f"Initial Balance: ${initial_balance:,.2f}")
            print(f"Final Balance:   ${self.environment.balance:,.2f}")
            print(f"Total Return:    {calculate_return_pct(self.environment.balance, initial_balance):+.2f}%")
            print(f"Total Trades:    {stats['total_trades']}")
            print(f"Winning Trades:  {stats['winning_trades']}")
            print(f"Losing Trades:   {stats['losing_trades']}")
            print(f"Win Rate:        {stats['win_rate']:.1f}%")
            print(f"Total PnL:       ${stats['total_pnl']:,.2f}")
            print(f"{'='*60}")
        
        return stats

    def start_demo_trading(self, test_data):
        """Start demo trading simulation"""
        self._ensure_agent()
        if not self.training_complete:
            print("\n⚠️ No model loaded. Please train or load a model first.")
            return
        
        print("\n🎮 STARTING DEMO TRADING SIMULATION")
        print("="*60)
        print("This is a simulation using historical data")
        print("No real money is used")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        self.environment.reset()
        self.position_candles_held = 0
        initial_balance = self.environment.balance
        trades_executed = 0
        
        try:
            for i in range(len(test_data) - 50):
                current_index = i + 50
                if self.environment.position != 0:
                    self.position_candles_held += 1
                action, decision = self.select_managed_action(test_data, current_index, training=False)
                if action is not None:
                    features = decision['features']
                    price = test_data['close'].iloc[current_index]
                    timestamp = test_data['timestamp'].iloc[current_index]
                    reason = decision['reason']

                    if action == 1 and self.environment.position == 0:
                        trades_executed += 1
                        print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M')}] 📈 BUY SIGNAL at {price:.5f} | {reason}")
                        self.environment.step(action, price, features)

                    elif action == 2 and self.environment.position == 0:
                        trades_executed += 1
                        print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M')}] 📉 SELL SIGNAL at {price:.5f} | {reason}")
                        self.environment.step(action, price, features)

                    elif action == 0 and self.environment.position != 0:
                        estimated_pnl = self._estimate_environment_pnl(price)
                        print(
                            f"\n[{timestamp.strftime('%Y-%m-%d %H:%M')}] 💰 CLOSE at {price:.5f} | "
                            f"est. PnL ${estimated_pnl:.2f} | {reason}"
                        )
                        self.environment.step(action, price, features)

                    if self.environment.position == 0:
                        self.position_candles_held = 0
                
                # Progress indicator
                if i % 500 == 0 and i > 0:
                    print(f"\n--- Progress: {i}/{len(test_data)-50} candles processed ---")
                    print(f"Current Balance: ${self.environment.balance:,.2f}")
                    print(f"Trades Executed: {trades_executed}")
                    print("-" * 40)
                
                time.sleep(0.05)  # Simulate real-time delay
        
        except KeyboardInterrupt:
            print("\n\n⏸️ Demo trading stopped")

        if len(test_data) > 0:
            self.environment.force_close(test_data['close'].iloc[-1])
            self.position_candles_held = 0
        
        # Show results
        stats = self.environment.get_statistics()
        print(f"\n{'='*60}")
        print(f"DEMO TRADING RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance:   ${self.environment.balance:,.2f}")
        print(f"Total Return:    {calculate_return_pct(self.environment.balance, initial_balance):+.2f}%")
        print(f"Total Trades:    {stats['total_trades']}")
        print(f"Winning Trades:  {stats['winning_trades']}")
        print(f"Losing Trades:   {stats['losing_trades']}")
        print(f"Win Rate:        {stats['win_rate']:.1f}%")
        print(f"Total PnL:       ${stats['total_pnl']:,.2f}")
        print(f"{'='*60}")
        
        # Ask to keep results
        keep = input("\nKeep these demo results? (yes/no, default: no): ").strip().lower()
        if keep != 'yes':
            self.environment.balance = initial_balance
            self.environment.trades = []
            self.environment.position = 0
            self.environment.entry_price = 0
            print("✅ Reverted to previous state")
