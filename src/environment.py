from .common import *

class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, initial_balance=10000, lot_size=0.1, spread=0.0001):
        self.initial_balance = initial_balance
        self.balance_scale = max(abs(initial_balance), 1.0)
        self.balance = initial_balance
        self.lot_size = lot_size
        self.spread = spread
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.current_step = 0
        
    def reset(self):
        """Reset the environment"""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state"""
        return {
            'balance': self.balance,
            'position': self.position,
            'current_step': self.current_step
        }

    def snapshot(self):
        """Capture mutable environment state for temporary evaluations."""
        return {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_step': self.current_step,
            'trades': [trade.copy() for trade in self.trades]
        }

    def restore(self, state):
        """Restore mutable environment state from a snapshot."""
        self.balance = state['balance']
        self.position = state['position']
        self.entry_price = state['entry_price']
        self.current_step = state['current_step']
        self.trades = [trade.copy() for trade in state['trades']]

    def force_close(self, price):
        """Close any open position using the latest available price."""
        if self.position == 0:
            return 0
        _, reward, _ = self.step(0, price, None)
        return reward
    
    def step(self, action, price, indicators):
        """Execute action: 0: close, 1: buy, 2: sell"""
        reward = 0
        done = False
        
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price + self.spread
            log_event(f"  BUY at {self.entry_price:.5f}")
            
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price - self.spread
            log_event(f"  SELL at {self.entry_price:.5f}")
            
        elif action == 0 and self.position != 0:
            if self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 100000
                self.balance += pnl
                self.trades.append({
                    'type': 'long',
                    'entry': self.entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'timestamp': datetime.now().isoformat()
                })
                reward = pnl / self.balance_scale
                log_event(f"  CLOSED LONG: PnL = ${pnl:.2f}, Balance = ${self.balance:.2f}")
                
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 100000
                self.balance += pnl
                self.trades.append({
                    'type': 'short',
                    'entry': self.entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'timestamp': datetime.now().isoformat()
                })
                reward = pnl / self.balance_scale
                log_event(f"  CLOSED SHORT: PnL = ${pnl:.2f}, Balance = ${self.balance:.2f}")
                
            self.position = 0
            self.entry_price = 0
        
        profit_cap = max(self.initial_balance * 2, self.balance_scale * 2)
        if self.balance <= 0 or self.balance > profit_cap:
            done = True
            
        self.current_step += 1
        
        return indicators, reward, done

    def estimate_open_pnl(self, price):
        """Estimate unrealized PnL for the currently open training position."""
        if self.position == 0:
            return 0.0
        if self.position == 1:
            return (price - self.entry_price) * self.lot_size * 100000
        return (self.entry_price - price) * self.lot_size * 100000
    
    def get_statistics(self):
        """Get trading statistics"""
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        df = pd.DataFrame(self.trades)
        winning = len(df[df['pnl'] > 0])
        total = len(self.trades)
        
        return {
            'total_trades': total,
            'winning_trades': winning,
            'losing_trades': total - winning,
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'max_win': df['pnl'].max() if winning > 0 else 0,
            'max_loss': df['pnl'].min() if total - winning > 0 else 0,
            'win_rate': (winning / total * 100) if total > 0 else 0
        }

