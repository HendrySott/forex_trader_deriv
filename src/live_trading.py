from .common import *


class RealtimeTradingMixin:
    def _reset_realtime_position(self):
        """Release the local live-trading lock when no active contract remains."""
        self.environment.position = 0
        self.environment.entry_price = 0
        self.position_candles_held = 0
        self.active_contract_id = None
        self.active_live_trade = None
        self.live_exit_in_flight = False
        self.live_resale_unavailable = False

    def _reset_live_session_stats(self, starting_balance):
        """Start each live session with a clean trade ledger and synced balance."""
        self.environment.trades = []
        self.environment.position = 0
        self.environment.entry_price = 0
        self.environment.balance = float(starting_balance)
        self.account_balance = float(starting_balance)
        self.position_candles_held = 0
        self.active_contract_id = None
        self.active_live_trade = None
        self.live_trade_history = []
        self.live_error_blocked = False
        self.live_exit_in_flight = False
        self.live_resale_unavailable = False

    def _choose_live_contract_duration(self, confidence, trend_strength):
        """Use stronger setups for slightly longer contract duration."""
        strength_bonus = 1 if trend_strength >= self.strong_trend_threshold else 0
        confidence_bonus = 1 if confidence >= 0.35 else 0
        duration = self.live_min_duration_candles + strength_bonus + confidence_bonus
        return max(self.live_min_duration_candles, min(self.live_max_duration_candles, duration))

    def _candidate_live_durations(self, symbol, requested_seconds):
        """Build an ordered list of live contract durations to try."""
        requested_seconds = max(60, int(requested_seconds))
        cached_seconds = self.accepted_live_duration_by_symbol.get(normalize_deriv_symbol(symbol))
        fallback_pool = (
            self.live_forex_duration_fallbacks
            if is_forex_symbol(symbol)
            else self.live_duration_fallbacks
        )
        candidates = []

        if cached_seconds:
            candidates.append(int(cached_seconds))
        candidates.append(requested_seconds)

        for fallback in fallback_pool:
            if fallback not in candidates:
                candidates.append(fallback)

        longer_candidates = [value for value in candidates if value >= requested_seconds]
        shorter_candidates = [value for value in candidates if value < requested_seconds]
        return longer_candidates + shorter_candidates

    def _buy_contract_with_fallbacks(self, symbol, amount, contract_type, requested_seconds):
        """Try several durations before treating the market as untradable."""
        symbol = normalize_deriv_symbol(symbol)
        attempted_durations = []
        for duration_seconds in self._candidate_live_durations(symbol, requested_seconds):
            attempted_durations.append(duration_seconds)
            buy_data = self.deriv_api.buy_contract(
                symbol,
                amount,
                contract_type,
                duration_seconds,
                "s"
            )
            if buy_data:
                self.accepted_live_duration_by_symbol[symbol] = duration_seconds
                if duration_seconds != requested_seconds:
                    log_event(
                        f"Requested duration {requested_seconds}s was adjusted to {duration_seconds}s "
                        f"for {symbol} because Deriv rejected the original duration."
                    )
                return buy_data, duration_seconds

            error_message = (self.deriv_api.last_error_message or "").lower()
            if "trading is not offered for this duration" not in error_message:
                break

        return None, attempted_durations

    def _extract_live_contract_metrics(self, contract_snapshot):
        """Read the latest live values from a Deriv contract snapshot."""
        if not contract_snapshot:
            return {
                'current_spot': None,
                'profit': 0.0,
                'bid_price': 0.0,
                'buy_price': 0.0,
                'is_sell_available': False,
                'status': 'unknown'
            }

        current_spot = contract_snapshot.get('current_spot')
        profit = float(contract_snapshot.get('profit', 0.0) or 0.0)
        bid_price = float(contract_snapshot.get('bid_price', 0.0) or 0.0)
        buy_price = float(contract_snapshot.get('buy_price', 0.0) or 0.0)
        return {
            'current_spot': float(current_spot) if current_spot is not None else None,
            'profit': profit,
            'bid_price': bid_price,
            'buy_price': buy_price,
            'is_sell_available': bool(contract_snapshot.get('is_valid_to_sell', False) or bid_price > 0),
            'status': contract_snapshot.get('status', 'unknown')
        }

    def _should_exit_live_contract(self, decision, live_metrics):
        """Decide whether to close an open live contract early."""
        current_profit = live_metrics['profit']
        context = decision['context']
        confidence = decision['confidence']
        position_direction = 1 if self.environment.position > 0 else -1
        trend_supports_trade = context['direction'] in {0, position_direction}

        if current_profit >= (self.min_profit_target + self.profit_lock_buffer):
            if context['reversal_risk'] >= 0.35 or confidence < self.hold_confidence_threshold:
                return True, "lock_live_profit"

        if current_profit >= self.min_profit_target and self.position_candles_held >= self.min_hold_candles:
            if not trend_supports_trade or self.position_candles_held >= self.max_hold_candles:
                return True, "take_small_live_profit"

        if current_profit <= self.max_loss_tolerance and self.position_candles_held >= self.min_hold_candles:
            if not trend_supports_trade:
                return True, "cut_live_loss"

        if self.position_candles_held >= self.max_hold_candles and current_profit > -0.05:
            return True, "time_based_live_exit"

        return False, "hold_live_contract"

    def _get_live_session_statistics(self):
        """Return executed/open/settled stats for the current live session."""
        executed_trades = len(self.live_trade_history)
        open_trade_rows = [trade for trade in self.live_trade_history if trade.get('status') == 'open']
        open_trades = len(open_trade_rows)
        settled_trades = [trade for trade in self.live_trade_history if trade.get('status') != 'open']
        settled_count = len(settled_trades)
        open_pnl = float(sum(float(trade.get('latest_profit', 0.0) or 0.0) for trade in open_trade_rows))

        if not settled_trades:
            return {
                'executed_trades': executed_trades,
                'settled_trades': 0,
                'open_trades': open_trades,
                'winning_trades': 0,
                'losing_trades': 0,
                'open_pnl': open_pnl,
                'total_pnl': 0.0,
                'win_rate': 0.0
            }

        settled_df = pd.DataFrame(settled_trades)
        winning = int((settled_df['pnl'] > 0).sum())
        losing = int((settled_df['pnl'] < 0).sum())

        return {
            'executed_trades': executed_trades,
            'settled_trades': settled_count,
            'open_trades': open_trades,
            'winning_trades': winning,
            'losing_trades': losing,
            'open_pnl': open_pnl,
            'total_pnl': float(settled_df['pnl'].sum()),
            'win_rate': (winning / settled_count * 100) if settled_count > 0 else 0.0
        }

    def _start_live_candle(self, timestamp, price):
        """Create a new aggregated candle bucket aligned to the trading timeframe."""
        timestamp_epoch = int(timestamp.timestamp())
        bucket_epoch = (timestamp_epoch // self.candle_granularity) * self.candle_granularity
        bucket_start = datetime.fromtimestamp(bucket_epoch)
        return {
            'bucket_epoch': bucket_epoch,
            'timestamp': bucket_start,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 1
        }

    def _append_completed_live_candle(self, candle):
        """Append one finished live candle and keep only recent history."""
        candle_row = pd.DataFrame([{
            'timestamp': candle['timestamp'],
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        self.current_data = pd.concat([self.current_data, candle_row], ignore_index=True)
        if len(self.current_data) > 200:
            self.current_data = self.current_data.iloc[-200:].reset_index(drop=True)

    def _evaluate_live_signal(self, symbol, decision_timestamp, decision_price, submit_trade):
        """Run one live model evaluation using the latest completed candle history."""
        if len(self.current_data) <= 50 or self.live_error_blocked:
            return

        if self.environment.position != 0 or self.active_contract_id is not None:
            log_event(
                f"Live candle {decision_timestamp.strftime('%Y-%m-%d %H:%M:%S')} closed while a contract is already open."
            )
            return

        action, decision = self.select_managed_action(
            self.current_data,
            len(self.current_data) - 1,
            training=False
        )
        if action is None:
            return

        confidence = decision['confidence']
        trend_strength = decision['context']['trend_strength']
        duration_candles = self._choose_live_contract_duration(confidence, trend_strength)
        action_label = {0: "HOLD/CLOSE", 1: "BUY", 2: "SELL"}.get(action, str(action))
        log_event(
            f"Model evaluated candle {decision_timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"at {decision_price:.5f} -> {action_label} | "
            f"confidence={confidence:.2f} | trend={trend_strength:.2f} | {decision['reason']}"
        )

        if action == 1 and not self.order_in_flight:
            print(
                f"\n[{decision_timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"AI BUY SIGNAL at {decision_price:.5f} | "
                f"duration {duration_candles} candle(s)"
            )
            threading.Thread(
                target=submit_trade,
                args=(symbol, decision_price, decision_timestamp, "CALL", 1, duration_candles, confidence, trend_strength),
                daemon=True
            ).start()

        elif action == 2 and not self.order_in_flight:
            print(
                f"\n[{decision_timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"AI SELL SIGNAL at {decision_price:.5f} | "
                f"duration {duration_candles} candle(s)"
            )
            threading.Thread(
                target=submit_trade,
                args=(symbol, decision_price, decision_timestamp, "PUT", -1, duration_candles, confidence, trend_strength),
                daemon=True
            ).start()

    def _block_live_trading(self, reason):
        """Stop new live entries after a fatal configuration/market error."""
        if self.live_error_blocked:
            return
        self.live_error_blocked = True
        log_event(f"Live trading blocked: {reason}", logging.WARNING)
        print("⚠️ Live trading has been paused because the current symbol/duration is not tradable.")
        if is_forex_symbol(self.market_symbol):
            print("   Forex is closed on weekends and short durations can be restricted on Deriv.")
            print("   Try a longer duration such as 300s or 600s, or switch to a synthetic market.")
            print(f"   For uninterrupted scalping, use a synthetic symbol such as {DEFAULT_SCALPING_SYMBOL}.")

    def start_realtime_trading(self, api_token, symbol=DEFAULT_SCALPING_SYMBOL, amount=10):
        """Start real-time trading on Deriv"""
        self._ensure_agent()
        symbol = normalize_deriv_symbol(symbol)
        if not self.training_complete:
            print("\n⚠️ No model loaded. Please train or load a model first.")
            return

        if is_forex_symbol(symbol) and datetime.utcnow().weekday() >= 5:
            utc_date = datetime.utcnow().strftime('%A, %B %d, %Y')
            print(f"\n⚠️ {symbol} is a forex market, and forex is closed on {utc_date} (UTC).")
            print(f"Use a synthetic symbol like {DEFAULT_SCALPING_SYMBOL} if you want weekend scalping.")
            return
        
        print("\n🚀 Starting Real-Time Trading on Deriv Demo Account")
        print("="*60)
        
        # Initialize Deriv API
        self.deriv_api = DerivWebSocketAPI(app_id="1089", demo=True)
        
        if not self.deriv_api.connect():
            print("❌ Failed to connect to Deriv")
            return
        
        time.sleep(2)
        
        if not self.deriv_api.authenticate(api_token):
            print("❌ Failed to authenticate")
            return

        auth_deadline = time.time() + 10
        while not self.deriv_api.authenticated and time.time() < auth_deadline:
            time.sleep(0.2)
        if not self.deriv_api.authenticated:
            print("❌ Authentication was not confirmed by Deriv")
            return

        try:
            active_symbols = self.deriv_api.fetch_active_symbols()
            active_symbol_codes = {
                item.get('symbol', '').upper()
                for item in active_symbols
                if item.get('symbol')
            }
            if active_symbol_codes and symbol not in active_symbol_codes:
                print(f"⚠️ {symbol} was not returned by Deriv's active-symbol list for this account.")
                print("   Continuing anyway and letting Deriv confirm availability on the actual request.")
        except Exception as e:
            print(f"⚠️ Could not verify live symbol availability before trading: {e}")

        print("\n✅ Connected to Deriv Demo Account")
        print(f"💰 Starting Balance: ${self.deriv_api.balance:.2f}")
        print("="*60)
        
        self.realtime_trading = True
        self.current_data = pd.DataFrame()
        self.order_in_flight = False
        self.live_candle = None
        self.last_completed_candle_epoch = None
        self._reset_live_session_stats(self.deriv_api.balance)

        try:
            preload_candles = self.deriv_api.fetch_historical_candles(
                symbol=symbol,
                granularity=self.candle_granularity,
                count=200,
                api_token=api_token
            )
            self.current_data = preload_candles.tail(200).reset_index(drop=True)
            if not self.current_data.empty:
                last_ts = self.current_data['timestamp'].iloc[-1]
                log_event(
                    f"Live warm-up loaded {len(self.current_data)} candles. "
                    f"Last completed candle: {last_ts.strftime('%Y-%m-%d %H:%M:%S')}"
                )
        except Exception as e:
            log_event(f"Live warm-up failed, starting from fresh ticks only: {e}", logging.WARNING)

        def on_balance_update(balance_data):
            """Keep the local account snapshot aligned with Deriv."""
            updated_balance = float(balance_data.get('balance', self.deriv_api.balance))
            self.deriv_api.balance = updated_balance
            self.account_balance = updated_balance

        def on_contract_update(contract):
            """Finalize local position state when Deriv reports a contract outcome."""
            contract_id = contract.get('contract_id')
            if contract_id != self.active_contract_id:
                return

            live_metrics = self._extract_live_contract_metrics(contract)
            if self.active_live_trade is not None:
                self.active_live_trade.update({
                    'latest_spot': live_metrics['current_spot'],
                    'latest_profit': live_metrics['profit'],
                    'bid_price': live_metrics['bid_price'],
                    'can_sell': live_metrics['is_sell_available'] and not self.live_resale_unavailable
                })

            if not contract.get('is_sold'):
                current_spot = live_metrics['current_spot']
                if current_spot is not None:
                    print(
                        f"\r⏳ Contract {contract_id} active | Spot: {float(current_spot):.5f} | "
                        f"PnL ${live_metrics['profit']:.2f}",
                        end="",
                        flush=True
                    )
                return

            payout = float(contract.get('payout', 0.0) or 0.0)
            buy_price = float(contract.get('buy_price', amount) or amount)
            profit = float(contract.get('profit', payout - buy_price) or 0.0)
            status = contract.get('status', 'unknown')
            exit_spot = contract.get('exit_tick') or contract.get('sell_spot') or contract.get('current_spot')
            print()

            self.account_balance = float(self.deriv_api.balance)
            self.environment.balance = self.account_balance
            if self.active_live_trade is not None:
                self.active_live_trade.update({
                    'exit': float(exit_spot) if exit_spot is not None else self.environment.entry_price,
                    'pnl': profit,
                    'status': status,
                    'closed_at': datetime.now().isoformat()
                })
            self.environment.trades.append({
                'type': 'long' if self.environment.position == 1 else 'short',
                'entry': self.environment.entry_price,
                'exit': float(exit_spot) if exit_spot is not None else self.environment.entry_price,
                'pnl': profit,
                'timestamp': datetime.now().isoformat(),
                'contract_id': contract_id,
                'status': status
            })
            log_event(
                f"Contract {contract_id} settled with {status.upper()}: "
                f"PnL = ${profit:.2f}, Balance = ${self.account_balance:.2f}"
            )
            if profit > 0:
                print(
                    f"✅ CONTRACT WON | ID {contract_id} | PnL ${profit:.2f} | "
                    f"Balance ${self.account_balance:.2f}"
                )
            elif profit < 0:
                print(
                    f"❌ CONTRACT LOST | ID {contract_id} | PnL ${profit:.2f} | "
                    f"Balance ${self.account_balance:.2f}"
                )
            else:
                print(
                    f"ℹ️ CONTRACT FLAT | ID {contract_id} | PnL ${profit:.2f} | "
                    f"Balance ${self.account_balance:.2f}"
                )
            self._reset_realtime_position()

        if 'balance' not in self.deriv_api.callbacks:
            self.deriv_api.callbacks['balance'] = []
        self.deriv_api.callbacks['balance'].append(on_balance_update)
        if 'contract_update' not in self.deriv_api.callbacks:
            self.deriv_api.callbacks['contract_update'] = []
        self.deriv_api.callbacks['contract_update'].append(on_contract_update)

        def submit_trade(signal_symbol, signal_price, signal_timestamp, contract_type, direction, duration_candles, confidence, trend_strength):
            """Place a Deriv order outside the WebSocket callback thread."""
            with self.order_lock:
                if self.order_in_flight or self.environment.position != 0 or self.active_contract_id is not None:
                    return
                self.order_in_flight = True

            try:
                signal_label = "BUY" if direction == 1 else "SELL"
                print(f"\n[{signal_timestamp.strftime('%H:%M:%S')}] {signal_label} order request at {signal_price:.5f}")
                requested_duration_seconds = max(60, int(duration_candles * self.candle_granularity))
                buy_data, used_duration = self._buy_contract_with_fallbacks(
                    signal_symbol,
                    amount,
                    contract_type,
                    requested_duration_seconds
                )
                if buy_data:
                    self.environment.position = direction
                    self.environment.entry_price = signal_price
                    self.position_candles_held = 0
                    self.active_contract_id = buy_data.get('contract_id')
                    self.live_resale_unavailable = False
                    self.active_live_trade = {
                        'contract_id': self.active_contract_id,
                        'type': 'long' if direction == 1 else 'short',
                        'entry': signal_price,
                        'exit': None,
                        'pnl': None,
                        'stake': float(amount),
                        'planned_candles': int(duration_candles),
                        'requested_duration_seconds': int(requested_duration_seconds),
                        'used_duration_seconds': int(used_duration),
                        'confidence': float(confidence),
                        'trend_strength': float(trend_strength),
                        'latest_spot': signal_price,
                        'latest_profit': 0.0,
                        'bid_price': 0.0,
                        'can_sell': False,
                        'resale_unavailable': False,
                        'status': 'open',
                        'opened_at': datetime.now().isoformat()
                    }
                    self.live_trade_history.append(self.active_live_trade)
                    self.deriv_api.subscribe_contract_updates(self.active_contract_id, on_contract_update)
                    print(
                        f"⏳ WAITING FOR CONTRACT TO SETTLE | ID {self.active_contract_id} | "
                        f"{contract_type} | Stake ${amount:.2f} | Duration {used_duration}s"
                    )
                    log_event(
                        f"Live contract opened: ID {self.active_contract_id} | "
                        f"Type {contract_type} | Stake ${amount:.2f} | "
                        f"Duration {used_duration}s"
                    )
                else:
                    log_event(f"Order request was rejected for {signal_label} at {signal_price:.5f}", logging.WARNING)
                    error_message = (self.deriv_api.last_error_message or "").lower()
                    if (
                        "market is presently closed" in error_message
                        or "trading is not offered for this duration" in error_message
                    ):
                        attempted = ", ".join(f"{value}s" for value in used_duration)
                        self._block_live_trading(
                            f"{self.deriv_api.last_error_message}. Tried durations: {attempted}"
                        )
            finally:
                with self.order_lock:
                    self.order_in_flight = False

        def submit_exit_request(contract_id, exit_reason):
            """Attempt to close the current contract early using Deriv's sell endpoint."""
            with self.order_lock:
                if self.live_exit_in_flight or contract_id != self.active_contract_id:
                    return
                self.live_exit_in_flight = True

            try:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] EARLY EXIT request for contract {contract_id} | {exit_reason}")
                sell_data = self.deriv_api.sell_contract(contract_id, minimum_price=0)
                if sell_data:
                    sold_for = float(sell_data.get('sold_for', sell_data.get('price', 0.0)) or 0.0)
                    log_event(
                        f"Early exit requested for contract {contract_id} | "
                        f"sold_for=${sold_for:.2f} | reason={exit_reason}"
                    )
                    if self.active_live_trade is not None:
                        self.active_live_trade['exit_reason'] = exit_reason
                else:
                    error_message = (self.deriv_api.last_error_message or "").lower()
                    if "resale of this contract is not offered" in error_message:
                        self.live_resale_unavailable = True
                        log_event(
                            f"Contract {contract_id} does not support resale. "
                            f"Early exits are disabled until settlement.",
                            logging.WARNING
                        )
                        if self.active_live_trade is not None:
                            self.active_live_trade.update({
                                'can_sell': False,
                                'resale_unavailable': True,
                                'exit_reason': 'resale_not_available'
                            })
                    log_event(f"Early exit request failed for contract {contract_id}", logging.WARNING)
            finally:
                with self.order_lock:
                    self.live_exit_in_flight = False

        if len(self.current_data) > 50:
            last_completed = self.current_data.iloc[-1]
            self._evaluate_live_signal(
                symbol,
                pd.to_datetime(last_completed['timestamp']).to_pydatetime(),
                float(last_completed['close']),
                submit_trade
            )
        else:
            log_event(
                f"Live warm-up has only {len(self.current_data)} candles; waiting for more completed candles."
            )
        
        def on_tick(symbol, tick):
            """Process each incoming tick"""
            if not self.realtime_trading:
                return
            
            price = float(tick['quote'])
            timestamp = datetime.fromtimestamp(tick['epoch'])

            if self.live_candle is None:
                self.live_candle = self._start_live_candle(timestamp, price)
                return

            current_bucket_epoch = self.live_candle['bucket_epoch']
            tick_bucket_epoch = int(timestamp.timestamp()) // self.candle_granularity * self.candle_granularity

            if tick_bucket_epoch == current_bucket_epoch:
                self.live_candle['high'] = max(self.live_candle['high'], price)
                self.live_candle['low'] = min(self.live_candle['low'], price)
                self.live_candle['close'] = price
                self.live_candle['volume'] += 1
                return

            completed_candle = self.live_candle
            self._append_completed_live_candle(completed_candle)
            self.last_completed_candle_epoch = completed_candle['bucket_epoch']
            self.live_candle = self._start_live_candle(timestamp, price)
            if self.environment.position != 0:
                self.position_candles_held += 1

            if len(self.current_data) <= 50:
                if len(self.current_data) in {1, 10, 25, 50}:
                    log_event(
                        f"Live warm-up progress: {len(self.current_data)}/51 completed candles collected."
                    )
                return

            decision_price = float(completed_candle['close'])
            decision_timestamp = completed_candle['timestamp']
            if self.active_contract_id is not None and self.environment.position != 0:
                decision, managed = self.select_managed_action(
                    self.current_data,
                    len(self.current_data) - 1,
                    training=False
                )
                contract_snapshot = self.deriv_api.contract_subscriptions.get(self.active_contract_id, {})
                live_metrics = self._extract_live_contract_metrics(contract_snapshot)
                should_exit, exit_reason = self._should_exit_live_contract(managed, live_metrics)
                if self.live_resale_unavailable:
                    exit_reason = "hold_to_settlement_no_resale"
                    should_exit = False
                log_event(
                    f"Live contract review | ID {self.active_contract_id} | candles_held={self.position_candles_held} | "
                    f"PnL=${live_metrics['profit']:.2f} | decision={managed['reason']} | exit_check={exit_reason}"
                )
                if self.live_resale_unavailable:
                    return
                if should_exit and live_metrics['is_sell_available'] and not self.live_exit_in_flight:
                    threading.Thread(
                        target=submit_exit_request,
                        args=(self.active_contract_id, exit_reason),
                        daemon=True
                    ).start()
                return

            self._evaluate_live_signal(symbol, decision_timestamp, decision_price, submit_trade)
        
        # Subscribe to ticks
        self.deriv_api.subscribe_ticks(symbol, on_tick)
        
        print(f"\n📊 Trading started on {symbol}")
        print("Press Ctrl+C to stop trading")
        print("="*60)
        
        try:
            while self.realtime_trading:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_realtime_trading()
    
    def stop_realtime_trading(self):
        """Stop real-time trading"""
        self.realtime_trading = False
        if self.deriv_api and self.deriv_api.ws:
            self.deriv_api.ws.close()
        log_event("\nReal-time trading stopped")
        
        # Show final statistics
        stats = self._get_live_session_statistics()
        print(f"\n{'='*60}")
        print(f"FINAL TRADING STATISTICS")
        print(f"{'='*60}")
        print(f"Final Balance: ${self.environment.balance:,.2f}")
        print(f"Total Trades: {stats['executed_trades']}")
        print(f"Settled Trades: {stats['settled_trades']}")
        print(f"Open Trades: {stats['open_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        if stats['open_trades'] > 0:
            print(f"Open Trade PnL: ${stats['open_pnl']:,.2f} (unrealized)")
        print(f"{'='*60}")
