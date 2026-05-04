from .common import *


class MarketAnalysisMixin:
    def _estimate_environment_pnl(self, price):
        """Estimate open-position PnL using the same scale as the training environment."""
        return self.environment.estimate_open_pnl(price)

    def _compute_action_confidence(self, q_values):
        """Estimate confidence from the gap between the best and second-best actions."""
        q_array = np.array(q_values, dtype=np.float32)
        if q_array.size == 0:
            return 0.0
        sorted_values = np.sort(q_array)
        best = float(sorted_values[-1])
        second_best = float(sorted_values[-2]) if q_array.size > 1 else best
        scale = max(np.max(np.abs(q_array)), 1.0)
        return max(0.0, min(1.0, (best - second_best) / scale))

    def _shape_training_reward(self, action, price, decision, base_reward):
        """Reward the model for patient entries and many small positive exits."""
        reward = float(base_reward)
        context = decision['context']
        confidence = decision['confidence']
        in_position = self.environment.position != 0

        if not in_position:
            if action == 0 and context['trend_strength'] < 1.0:
                reward += self.training_wait_reward
            elif action in (1, 2) and decision['reason'] == "waiting_for_better_setup":
                reward -= self.training_bad_entry_penalty
            return reward

        estimated_pnl = self._estimate_environment_pnl(price)
        position_direction = 1 if self.environment.position > 0 else -1
        trend_supports_trade = context['direction'] in {0, position_direction}

        if action == 0:
            if estimated_pnl >= self.min_profit_target:
                reward += self.training_small_profit_bonus
            elif estimated_pnl < 0:
                reward -= min(abs(estimated_pnl) / 10.0, 1.0) * self.training_loss_penalty_scale
        else:
            if trend_supports_trade and confidence >= self.hold_confidence_threshold:
                reward += self.training_trend_hold_reward
            if estimated_pnl > 0 and estimated_pnl < self.min_profit_target:
                reward += self.training_wait_reward * 0.5
            if self.position_candles_held >= self.max_hold_candles and estimated_pnl <= 0:
                reward -= self.training_overhold_penalty
            elif abs(estimated_pnl) < 0.05 and self.position_candles_held >= 2:
                reward -= self.training_stagnation_penalty

        return reward

    def analyze_market_context(self, df, index):
        """Read recent candles to estimate short-term trend, strength, and reversal risk."""
        close = df['close']
        current_close = float(close.iloc[index])
        sma_20 = float(self.indicators.calculate_sma(close, 20).iloc[index])
        ema_12 = float(self.indicators.calculate_ema(close, 12).iloc[index])
        rsi = float(self.indicators.calculate_rsi(close, 14).iloc[index])
        macd_line, signal_line, histogram = self.indicators.calculate_macd(close)
        macd_hist = float(histogram.iloc[index]) if not pd.isna(histogram.iloc[index]) else 0.0

        recent_window = close.iloc[max(0, index - 3):index + 1]
        recent_change = 0.0
        if len(recent_window) >= 2 and recent_window.iloc[0] != 0:
            recent_change = ((recent_window.iloc[-1] / recent_window.iloc[0]) - 1.0) * 100

        bullish_score = 0.0
        bearish_score = 0.0

        if not pd.isna(sma_20):
            if current_close > sma_20:
                bullish_score += 1.0
            elif current_close < sma_20:
                bearish_score += 1.0

        if not pd.isna(ema_12):
            if current_close > ema_12:
                bullish_score += 0.8
            elif current_close < ema_12:
                bearish_score += 0.8

        if recent_change > 0:
            bullish_score += min(abs(recent_change) * 2.0, 1.2)
        elif recent_change < 0:
            bearish_score += min(abs(recent_change) * 2.0, 1.2)

        if macd_hist > 0:
            bullish_score += min(abs(macd_hist) / 0.01, 1.0)
        elif macd_hist < 0:
            bearish_score += min(abs(macd_hist) / 0.01, 1.0)

        if not pd.isna(rsi):
            if rsi >= 58:
                bullish_score += 0.8
            elif rsi <= 42:
                bearish_score += 0.8

        net_score = bullish_score - bearish_score
        direction = 1 if net_score > 0.35 else -1 if net_score < -0.35 else 0
        reversal_risk = 0.0
        if direction > 0 and not pd.isna(rsi) and rsi >= 72:
            reversal_risk += 0.5
        if direction < 0 and not pd.isna(rsi) and rsi <= 28:
            reversal_risk += 0.5
        if abs(net_score) < 0.75:
            reversal_risk += 0.25

        return {
            'direction': direction,
            'trend_strength': abs(net_score),
            'trend_score': net_score,
            'recent_change_pct': recent_change,
            'rsi': rsi if not pd.isna(rsi) else 50.0,
            'macd_hist': macd_hist,
            'reversal_risk': min(reversal_risk, 1.0)
        }

    def select_managed_action(self, df, index, training=False):
        """Blend model output with candle structure and profit-protection rules."""
        features = self.get_features(df, index)
        if features is None:
            return None, None

        if training and np.random.rand() <= self.agent.epsilon:
            random_action = random.randrange(self.agent.action_size)
            return random_action, {
                'features': features,
                'q_values': np.zeros(self.agent.action_size, dtype=np.float32),
                'confidence': 0.0,
                'context': self.analyze_market_context(df, index),
                'reason': 'epsilon_exploration'
            }

        q_values = self.agent.predict_q_values(features)
        model_action = int(np.argmax(q_values))
        confidence = self._compute_action_confidence(q_values)
        context = self.analyze_market_context(df, index)
        direction = context['direction']
        reason = "hold_by_default"
        action = 0

        if self.environment.position == 0:
            aligned_action = 1 if direction > 0 else 2 if direction < 0 else 0
            expected_edge = confidence + (context['trend_strength'] / 6.0) - context['reversal_risk']
            if (
                aligned_action != 0
                and model_action == aligned_action
                and confidence >= self.entry_confidence_threshold
                and context['trend_strength'] >= 1.0
                and expected_edge >= self.min_expected_edge
            ):
                action = aligned_action
                reason = "aligned_model_and_trend"
            elif aligned_action != 0 and context['trend_strength'] >= self.strong_trend_threshold:
                action = aligned_action
                reason = "strong_candle_trend_override"
            else:
                reason = "waiting_for_better_setup"
        else:
            price = float(df['close'].iloc[index])
            estimated_pnl = self._estimate_environment_pnl(price)
            position_direction = 1 if self.environment.position > 0 else -1
            trend_supports_trade = direction == position_direction or direction == 0
            profit_locked = estimated_pnl >= (self.min_profit_target + self.profit_lock_buffer)
            profit_target_hit = estimated_pnl >= self.min_profit_target
            loss_too_large = estimated_pnl <= self.max_loss_tolerance
            held_long_enough = self.position_candles_held >= self.min_hold_candles
            held_too_long = self.position_candles_held >= self.max_hold_candles

            if profit_target_hit and held_long_enough and (not trend_supports_trade or confidence < self.hold_confidence_threshold):
                action = 0
                reason = "secured_small_profit"
            elif profit_locked and held_long_enough and context['reversal_risk'] >= 0.45:
                action = 0
                reason = "protecting_locked_profit"
            elif loss_too_large and held_long_enough and not trend_supports_trade:
                action = 0
                reason = "cutting_losing_trade"
            elif held_too_long and estimated_pnl > -0.05:
                action = 0
                reason = "time_exit_after_several_candles"
            else:
                action = self.environment.position
                action = 1 if action > 0 else 2
                reason = "holding_for_more_confirmation"

        return action, {
            'features': features,
            'q_values': q_values,
            'confidence': confidence,
            'context': context,
            'reason': reason
        }

    def get_features(self, df, index):
        """Extract features for AI"""
        if index < 50:
            return None
        
        try:
            close = df['close'].iloc[index]
            
            sma_20 = self.indicators.calculate_sma(df['close'], 20).iloc[index]
            sma_50 = self.indicators.calculate_sma(df['close'], 50).iloc[index]
            ema_12 = self.indicators.calculate_ema(df['close'], 12).iloc[index]
            rsi = self.indicators.calculate_rsi(df['close'], 14).iloc[index]
            
            macd_line, signal_line, histogram = self.indicators.calculate_macd(df['close'])
            macd_val = macd_line.iloc[index] if not pd.isna(macd_line.iloc[index]) else 0
            hist_val = histogram.iloc[index] if not pd.isna(histogram.iloc[index]) else 0
            
            upper_bb, middle_bb, lower_bb = self.indicators.calculate_bollinger_bands(df['close'])
            upper_val = upper_bb.iloc[index] if not pd.isna(upper_bb.iloc[index]) else close
            lower_val = lower_bb.iloc[index] if not pd.isna(lower_bb.iloc[index]) else close
            
            volume = df['volume'].iloc[index] / 1000 if 'volume' in df.columns else 1
            price_change = df['close'].pct_change().iloc[index] * 100 if index > 0 else 0
            
            features = np.array([
                close / 1.2,
                sma_20 / 1.2 if not pd.isna(sma_20) else close / 1.2,
                sma_50 / 1.2 if not pd.isna(sma_50) else close / 1.2,
                ema_12 / 1.2 if not pd.isna(ema_12) else close / 1.2,
                rsi / 100 if not pd.isna(rsi) else 0.5,
                macd_val / 0.01 if abs(macd_val) > 0 else 0,
                hist_val / 0.01 if abs(hist_val) > 0 else 0,
                (close - lower_val) / (upper_val - lower_val) if (upper_val - lower_val) > 0 else 0.5,
                volume,
                price_change if not pd.isna(price_change) else 0
            ])
            
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            return features
            
        except Exception as e:
            logging.debug("Feature extraction failed at index %s: %s", index, e)
            return None

    def build_feature_matrix(self, df):
        """Precompute all model features once for a dataset."""
        close = df['close']

        sma_20 = self.indicators.calculate_sma(close, 20)
        sma_50 = self.indicators.calculate_sma(close, 50)
        ema_12 = self.indicators.calculate_ema(close, 12)
        rsi = self.indicators.calculate_rsi(close, 14)
        macd_line, _, histogram = self.indicators.calculate_macd(close)
        upper_bb, _, lower_bb = self.indicators.calculate_bollinger_bands(close)
        volume = (df['volume'] / 1000) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
        price_change = close.pct_change() * 100

        safe_upper = upper_bb.fillna(close)
        safe_lower = lower_bb.fillna(close)
        band_range = (safe_upper - safe_lower).replace(0, np.nan)

        features_df = pd.DataFrame({
            'close': close / 1.2,
            'sma_20': sma_20.fillna(close) / 1.2,
            'sma_50': sma_50.fillna(close) / 1.2,
            'ema_12': ema_12.fillna(close) / 1.2,
            'rsi': rsi.fillna(50) / 100,
            'macd': macd_line.where(macd_line.abs() > 0, 0).fillna(0) / 0.01,
            'hist': histogram.where(histogram.abs() > 0, 0).fillna(0) / 0.01,
            'bb_pos': ((close - safe_lower) / band_range).fillna(0.5),
            'volume': volume.fillna(1.0),
            'price_change': price_change.fillna(0)
        })

        features = np.nan_to_num(
            features_df.to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )
        valid_mask = np.arange(len(df)) >= 50
        return features, valid_mask
