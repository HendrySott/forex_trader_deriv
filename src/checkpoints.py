from .common import *


class CheckpointMixin:
    def _save_metadata(self, path, metadata):
        """Write checkpoint metadata as JSON."""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata_for_model(self, checkpoint_path):
        """Load metadata associated with a saved model, if available."""
        model_name = os.path.basename(checkpoint_path)
        timestamp = model_name.replace('.h5', '').split('_')[-1]
        metadata_path = os.path.join(self.checkpoint_dir, f"metadata_{timestamp}.json")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _load_history_for_model(self, checkpoint_path):
        """Load training history associated with a saved model, if available."""
        model_name = os.path.basename(checkpoint_path)
        timestamp = model_name.replace('.h5', '').split('_')[-1]
        history_path = os.path.join(self.checkpoint_dir, f"history_{timestamp}.json")
        if not os.path.exists(history_path):
            return None
        with open(history_path, 'r') as f:
            return json.load(f)

    def _write_latest_checkpoint_reference(self, checkpoint_path, metadata_path, history_path):
        """Persist a pointer to the most recent checkpoint for automatic resume."""
        latest_data = {
            'checkpoint_path': checkpoint_path,
            'metadata_path': metadata_path,
            'history_path': history_path,
            'saved_at': datetime.now().isoformat()
        }
        self._save_metadata(self.latest_checkpoint_file, latest_data)

    def load_latest_checkpoint(self, load_model=True):
        """Load the most recent saved checkpoint, if one exists."""
        if os.path.exists(self.latest_checkpoint_file):
            try:
                with open(self.latest_checkpoint_file, 'r') as f:
                    latest_data = json.load(f)
                checkpoint_path = latest_data.get('checkpoint_path')
                if checkpoint_path and os.path.exists(checkpoint_path):
                    self.load_checkpoint(checkpoint_path, load_model=load_model)
                    return checkpoint_path
            except Exception as e:
                logger.warning("Could not read latest checkpoint reference: %s", e)

        checkpoints = self.get_checkpoint_list()
        if checkpoints:
            checkpoint_path = checkpoints[0]['path']
            self.load_checkpoint(checkpoint_path, load_model=load_model)
            return checkpoint_path
        return None

    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        self._ensure_agent()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.checkpoint_dir, f"model_ep{episode}_{timestamp}.weights.h5")
        self.agent.save(model_path)
        
        history_path = os.path.join(self.checkpoint_dir, f"history_{timestamp}.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        metadata = {
            'episode': episode,
            'best_win_rate': self.best_win_rate,
            'epsilon': self.agent.epsilon,
            'balance': self.environment.balance,
            'total_trades': len(self.environment.trades),
            'timestamp': timestamp,
            'best_model_path': self.best_model_path,
            'training_complete': self.training_complete,
            'training_history_length': len(self.training_history)
        }
        
        meta_path = os.path.join(self.checkpoint_dir, f"metadata_{timestamp}.json")
        self._save_metadata(meta_path, metadata)
        self._write_latest_checkpoint_reference(model_path, meta_path, history_path)
        log_event(f"Checkpoint saved: {model_path}")
        
        return model_path
    
    def save_training_summary(self):
        """Save training summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.checkpoint_dir, f"summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("BEST PERFORMANCE:\n")
            f.write(f"  Win Rate: {self.best_win_rate:.1f}%\n")
            f.write(f"  Model: {self.best_model_path}\n\n")
            
            stats = self.environment.get_statistics()
            f.write("FINAL STATISTICS:\n")
            f.write(f"  Final Balance: ${self.environment.balance:,.2f}\n")
            f.write(f"  Total Return: {calculate_return_pct(self.environment.balance, self.initial_balance):.2f}%\n")
            f.write(f"  Total Trades: {stats['total_trades']}\n")
            f.write(f"  Win Rate: {stats['win_rate']:.1f}%\n")
            f.write(f"  Total PnL: ${stats['total_pnl']:,.2f}\n\n")
            
            f.write("EPISODE PROGRESSION (Last 50):\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Episode':<10} {'Profit':<12} {'Win Rate':<12} {'Trades':<8} {'Epsilon':<10}\n")
            f.write("-"*70 + "\n")
            
            for ep in self.training_history[-50:]:
                f.write(f"{ep['episode']:<10} ${ep['profit']:<11.2f} "
                       f"{ep['win_rate']:<11.1f}% {ep['trades']:<8} {ep['epsilon']:<10.3f}\n")
        
        log_event(f"Summary saved: {summary_path}")
    
    def load_checkpoint(self, checkpoint_path, load_model=True):
        """Load a specific checkpoint"""
        metadata = self._load_metadata_for_model(checkpoint_path)
        history = self._load_history_for_model(checkpoint_path)
        if history is not None:
            self.training_history = history
        if metadata:
            self.current_episode = metadata.get('episode', self.current_episode)
            self.best_win_rate = metadata.get('best_win_rate', self.best_win_rate)
            self.best_model_path = metadata.get('best_model_path', self.best_model_path)
            self.training_complete = metadata.get('training_complete', True)
            self.environment.balance = metadata.get('balance', self.environment.balance)
        self.training_complete = True
        self.pending_checkpoint_path = checkpoint_path

        if load_model:
            self._ensure_agent()
            if metadata and self.agent is not None:
                self.agent.epsilon = metadata.get('epsilon', self.agent.epsilon)
        else:
            log_event(f"Checkpoint queued for lazy loading: {checkpoint_path}")
    
    def get_checkpoint_list(self):
        """Get list of all checkpoints with details"""
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.h5'):
                path = os.path.join(self.checkpoint_dir, f)
                size = os.path.getsize(path) / 1024
                modified = datetime.fromtimestamp(os.path.getmtime(path))
                
                episode = "unknown"
                if "ep" in f:
                    try:
                        episode = f.split("ep")[1].split("_")[0]
                    except:
                        pass
                
                checkpoints.append({
                    'file': f,
                    'path': path,
                    'size_kb': size,
                    'modified': modified,
                    'episode': episode
                })
        
        return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)
    
    def print_model_info(self):
        """Print current model information"""
        print(f"\n{'='*60}")
        print(f"🤖 MODEL INFORMATION")
        print(f"{'='*60}")
        if self.agent is None:
            print("TensorFlow/model backend: Not available")
            print("Install TensorFlow to enable training and model loading")
        else:
            print(f"State Size: {self.agent.state_size}")
            print(f"Action Size: {self.agent.action_size}")
            print(f"Current Epsilon: {self.agent.epsilon:.4f}")
            print(f"Memory Size: {len(self.agent.memory)}")
        print(f"Training Complete: {self.training_complete}")
        print(f"Total Trades: {len(self.environment.trades)}")
        print(f"Current Balance: ${self.environment.balance:,.2f}")
        print(f"Best Win Rate: {self.best_win_rate:.1f}%")
        print(f"{'='*60}")
