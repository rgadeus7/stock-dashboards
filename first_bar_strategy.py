import pandas as pd
import pytz
from datetime import datetime, time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Set up logging - only show errors
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingRLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        """
        Initialize the Reinforcement Learning agent for trading.
        
        Args:
            learning_rate (float): How quickly the agent updates its Q-values
            discount_factor (float): How much future rewards are valued
            exploration_rate (float): Probability of taking random actions
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.scaler = StandardScaler()
        
        # Define possible stop loss and target levels
        self.stop_loss_levels = [5, 7, 10, 12, 15]  # Points below entry
        self.target_levels = ['first_bar_high', 'first_bar_close', 'first_bar_high_plus_5', 'first_bar_high_plus_10']
        
        # Define factor thresholds for trade filtering
        self.factor_thresholds = {
            'rsi': [30, 40, 50, 60, 70],  # RSI levels
            'ema9': [0, 1, 2, 3, 4],      # Points above/below EMA9
            'ema20': [0, 1, 2, 3, 4],     # Points above/below EMA20
            'atr': [0.5, 1.0, 1.5, 2.0, 2.5],  # ATR multiplier
            'volume': [0.5, 0.8, 1.0, 1.2, 1.5]  # Volume ratio
        }
        
        # Define EMA lookback periods
        self.ema_lookback_periods = [2, 5, 7, 9, 20]
        
        # Track performance of each action
        self.action_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'total_pl': 0})
        
    def get_state_features(self, first_bar, second_bar, daily_bias=None):
        """Extract relevant features for the state"""
        # Calculate price ranges and movements
        first_bar_range = first_bar['High'] - first_bar['Low']
        second_bar_range = second_bar['High'] - second_bar['Low']
        
        # Calculate price changes
        price_change = second_bar['Close'] - first_bar['Close']
        price_change_pct = (price_change / first_bar['Close']) * 100
        
        # Calculate volume changes
        volume_change = second_bar['Volume'] - first_bar['Volume']
        volume_change_pct = (volume_change / first_bar['Volume']) * 100 if first_bar['Volume'] > 0 else 0
        
        # Calculate bar characteristics
        first_bar_body = abs(first_bar['Close'] - first_bar['Open'])
        second_bar_body = abs(second_bar['Close'] - second_bar['Open'])
        
        # Calculate relative positions
        second_bar_low_to_first_low = (second_bar['Low'] - first_bar['Low']) / first_bar['Low'] * 100
        second_bar_high_to_first_high = (second_bar['High'] - first_bar['High']) / first_bar['High'] * 100
        
        # Calculate potential profit targets
        first_bar_high_target = first_bar['High']
        first_bar_close_target = first_bar['Close']
        first_bar_high_plus_5 = first_bar['High'] + 5
        first_bar_high_plus_10 = first_bar['High'] + 10
        
        # Calculate factor values
        rsi = first_bar['RSI']
        ema9 = first_bar['EMA9']
        ema20 = first_bar['EMA20']
        atr = first_bar['ATR']
        
        # Calculate volume ratio (use the volume directly)
        volume_ratio = first_bar['Volume'] / 1000  # Normalize by 1000 to get a reasonable ratio
        
        # Calculate RSI bucket
        rsi_bucket = 0  # Default bucket
        if 50 <= rsi < 70:
            rsi_bucket = 1  # Optimal RSI range
        elif rsi >= 70:
            rsi_bucket = 2  # Overbought
        elif rsi < 50:
            rsi_bucket = 3  # Below optimal range
        
        # Calculate EMA conditions
        ema_conditions = {}
        for period in self.ema_lookback_periods:
            ema_conditions[f'above_ema9_{period}'] = 1 if first_bar['Close'] > ema9 else 0
        
        features = {
            'first_bar_range': first_bar_range,
            'first_bar_body': first_bar_body,
            'second_bar_range': second_bar_range,
            'second_bar_body': second_bar_body,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume_change': volume_change,
            'volume_change_pct': volume_change_pct,
            'second_bar_low_to_first_low': second_bar_low_to_first_low,
            'second_bar_high_to_first_high': second_bar_high_to_first_high,
            'first_bar_high_target': first_bar_high_target,
            'first_bar_close_target': first_bar_close_target,
            'first_bar_high_plus_5': first_bar_high_plus_5,
            'first_bar_high_plus_10': first_bar_high_plus_10,
            'first_bar_low': first_bar['Low'],
            'rsi': rsi,
            'rsi_bucket': rsi_bucket,
            'ema9': ema9,
            'ema20': ema20,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'volume': first_bar['Volume']
        }
        
        # Add EMA conditions to features
        features.update(ema_conditions)
        
        if daily_bias is not None:
            features['daily_bias'] = daily_bias
            
        return features
        
    def get_state_key(self, features):
        """Convert features to a state key for the Q-table"""
        # Discretize continuous features into bins
        def discretize(value, bins=5):
            if value < -2: return -2
            if value > 2: return 2
            return round(value)
            
        # Convert second bar low to first bar low difference to points
        second_bar_low_diff = features['second_bar_low_to_first_low'] * features['first_bar_low'] / 100
        
        # Discretize the difference into point levels
        if second_bar_low_diff <= 0:
            entry_level = 0  # At or above first bar low
        elif second_bar_low_diff <= 5:
            entry_level = 1  # 0-5 points below
        elif second_bar_low_diff <= 10:
            entry_level = 2  # 5-10 points below
        elif second_bar_low_diff <= 20:
            entry_level = 3  # 10-20 points below
        else:
            entry_level = 4  # More than 20 points below
            
        # Discretize factor values
        rsi_level = min(len(self.factor_thresholds['rsi']) - 1, 
                       sum(1 for t in self.factor_thresholds['rsi'] if features['rsi'] > t))
        
        ema9_level = min(len(self.factor_thresholds['ema9']) - 1,
                        sum(1 for t in self.factor_thresholds['ema9'] 
                            if features['first_bar_low'] - features['ema9'] > t))
        
        ema20_level = min(len(self.factor_thresholds['ema20']) - 1,
                         sum(1 for t in self.factor_thresholds['ema20'] 
                             if features['first_bar_low'] - features['ema20'] > t))
        
        atr_level = min(len(self.factor_thresholds['atr']) - 1,
                       sum(1 for t in self.factor_thresholds['atr'] 
                           if features['atr'] > t * features['first_bar_low'] / 100))
        
        volume_level = min(len(self.factor_thresholds['volume']) - 1,
                          sum(1 for t in self.factor_thresholds['volume'] 
                              if features['volume_ratio'] > t))
            
        state_key = (
            discretize(features['first_bar_range'] / 100),  # Normalize by 100
            discretize(features['first_bar_body'] / 100),
            discretize(features['second_bar_range'] / 100),
            discretize(features['second_bar_body'] / 100),
            discretize(features['price_change_pct']),
            discretize(features['volume_change_pct']),
            entry_level,  # Use points-based entry level
            discretize(features['second_bar_high_to_first_high']),
            rsi_level,
            ema9_level,
            ema20_level,
            atr_level,
            volume_level
        )
        
        if 'daily_bias' in features:
            state_key += (features['daily_bias'],)
            
        return state_key
        
    def choose_action(self, state_key, valid_actions):
        """Choose an action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Convert valid_actions to a list of tuples if it's not already
            if isinstance(valid_actions, list):
                return valid_actions[np.random.randint(len(valid_actions))]
            return valid_actions
            
        # Get Q-values for the state
        q_values = [self.q_table[state_key][action] for action in valid_actions]
        return valid_actions[np.argmax(q_values)]
        
    def get_action_parameters(self, action, features):
        """Get stop loss and target levels for the chosen action"""
        # Action is a tuple of (stop_loss_index, target_index)
        stop_loss = self.stop_loss_levels[action[0]]
        target_type = self.target_levels[action[1]]
        
        # Get target price based on type
        if target_type == 'first_bar_high':
            target_price = features['first_bar_high_target']
        elif target_type == 'first_bar_close':
            target_price = features['first_bar_close_target']
        elif target_type == 'first_bar_high_plus_5':
            target_price = features['first_bar_high_plus_5']
        else:  # first_bar_high_plus_10
            target_price = features['first_bar_high_plus_10']
            
        return stop_loss, target_price
        
    def update_q_value(self, state_key, action, reward, next_state_key, next_valid_actions):
        """Update Q-value using Q-learning update rule"""
        # Get maximum Q-value for next state
        next_q_values = [self.q_table[next_state_key][action] for action in next_valid_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
    def get_reward(self, trade_result, p_l, stop_loss, target_price, entry_price):
        """Calculate reward based on trade outcome and risk/reward ratio"""
        # Calculate risk/reward ratio
        risk = stop_loss
        reward = target_price - entry_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Base reward on P/L
        if trade_result == 'Target Hit':
            return p_l * 2 * risk_reward_ratio  # Double reward for target hits
        elif trade_result == 'Stop Loss':
            return p_l * 0.5  # Half penalty for stop losses
        else:
            return p_l * risk_reward_ratio  # Normal reward for market closes
            
    def save_model(self, filename):
        """Save the Q-table to a file"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
            
    def load_model(self, filename):
        """Load the Q-table from a file"""
        import pickle
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))

    def analyze_recommendations(self):
        """Analyze and display the RL agent's recommendations"""
        # Convert Q-table to a more analyzable format
        action_performance = defaultdict(list)
        
        # Only analyze states that have been visited
        visited_states = {state for state in self.q_table.keys() if any(q_value != 0 for q_value in self.q_table[state].values())}
        
        for state in visited_states:
            for action, q_value in self.q_table[state].items():
                if q_value != 0:  # Only include actions that have been taken
                    action_performance[action].append(q_value)
        
        # Calculate statistics for each action
        action_stats = {}
        for action, q_values in action_performance.items():
            if q_values:
                action_stats[action] = {
                    'mean_q': np.mean(q_values),
                    'max_q': np.max(q_values),
                    'count': len(q_values)
                }
        
        # Sort actions by mean Q-value
        sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['mean_q'], reverse=True)
        
        print("\nRL Agent Recommendations:")
        print("=" * 50)
        print("Top 5 Action Combinations (Stop Loss, Target):")
        print("-" * 30)
        
        for action, stats in sorted_actions[:5]:
            stop_loss = self.stop_loss_levels[action[0]]
            target = self.target_levels[action[1]]
            print(f"Stop Loss: {stop_loss} points, Target: {target}")
            print(f"Mean Q-value: {stats['mean_q']:.2f}")
            print(f"Max Q-value: {stats['max_q']:.2f}")
            print(f"Number of times chosen: {stats['count']}")
            print("-" * 30)
            
        # Analyze performance by RSI bucket
        print("\nRSI Bucket Analysis:")
        print("-" * 30)
        rsi_buckets = {
            0: "Default",
            1: "RSI 50-70 (Optimal)",
            2: "RSI >= 70 (Overbought)",
            3: "RSI < 50 (Below Optimal)"
        }
        
        for bucket, description in rsi_buckets.items():
            bucket_states = [state for state in visited_states if state[8] == bucket]  # RSI bucket is at index 8
            if bucket_states:
                mean_q = np.mean([self.q_table[state][action] for state in bucket_states 
                                for action in self.q_table[state] if self.q_table[state][action] != 0])
                total_trades = sum(1 for state in bucket_states 
                                 for action in self.q_table[state] if self.q_table[state][action] != 0)
                print(f"{description}:")
                print(f"  Mean Q-value: {mean_q:.2f}")
                print(f"  Total Trades: {total_trades}")
                print("-" * 30)
        
        # Analyze performance by EMA conditions
        print("\nEMA Condition Analysis:")
        print("-" * 30)
        
        # Get all states where price is above EMA9
        ema_states = [state for state in visited_states if state[9] == 1]  # EMA9 condition is at index 9
        
        if ema_states:
            mean_q = np.mean([self.q_table[state][action] for state in ema_states 
                            for action in self.q_table[state] if self.q_table[state][action] != 0])
            total_trades = sum(1 for state in ema_states 
                             for action in self.q_table[state] if self.q_table[state][action] != 0)
            print("Price above EMA9:")
            print(f"  Mean Q-value: {mean_q:.2f}")
            print(f"  Total Trades: {total_trades}")
            print("-" * 30)
        
        # Analyze performance by EMA20 conditions
        ema20_states = [state for state in visited_states if state[10] == 1]  # EMA20 condition is at index 10
        
        if ema20_states:
            mean_q = np.mean([self.q_table[state][action] for state in ema20_states 
                            for action in self.q_table[state] if self.q_table[state][action] != 0])
            total_trades = sum(1 for state in ema20_states 
                             for action in self.q_table[state] if self.q_table[state][action] != 0)
            print("Price above EMA20:")
            print(f"  Mean Q-value: {mean_q:.2f}")
            print(f"  Total Trades: {total_trades}")
            print("-" * 30)
        
        return action_stats
        
    def update_action_stats(self, action, p_l):
        """Update statistics for each action"""
        self.action_stats[action]['count'] += 1
        self.action_stats[action]['total_pl'] += p_l
        if p_l > 0:
            self.action_stats[action]['wins'] += 1
            
    def get_action_performance(self):
        """Get performance statistics for each action"""
        performance = {}
        for action, stats in self.action_stats.items():
            if stats['count'] > 0:
                performance[action] = {
                    'count': stats['count'],
                    'win_rate': (stats['wins'] / stats['count']) * 100,
                    'avg_pl': stats['total_pl'] / stats['count']
                }
        return performance

class FirstBarStrategy:
    def __init__(self, data_file, daily_data_file=None, entry_offset=0.01, use_daily_bias=False, use_rl=False, stop_loss=10):
        """
        Initialize the First Bar Strategy.
        
        Args:
            data_file (str): Path to the CSV file containing SPX 60m data
            daily_data_file (str, optional): Path to the CSV file containing SPX daily data
            entry_offset (float): Offset from the low price for entry (default: 0.01)
            use_daily_bias (bool): Whether to use daily bias for trade entry (default: False)
            use_rl (bool): Whether to use reinforcement learning (default: False)
            stop_loss (float): Fixed stop loss in points (default: 10)
        """
        self.data_file = data_file
        self.daily_data_file = daily_data_file
        self.entry_offset = entry_offset
        self.use_daily_bias = use_daily_bias
        self.use_rl = use_rl
        self.stop_loss = stop_loss
        self.rl_agent = TradingRLAgent() if use_rl else None
        self.cst = pytz.timezone('America/Chicago')
        self.utc = pytz.UTC
        
    def load_data(self):
        """Load and process the SPX data"""
        try:
            # Read the CSV file
            df = pd.read_csv(self.data_file, skiprows=2)  # Skip the header rows
            
            # Convert Datetime column to datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Convert to CST
            df['Datetime'] = df['Datetime'].dt.tz_convert(self.cst)
            
            # Convert price columns to float
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert Volume to numeric
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Sort by datetime
            df = df.sort_values('Datetime')
            
            # Add date column for grouping
            df['Date'] = df['Datetime'].dt.date
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def load_daily_data(self):
        """Load and process the daily SPX data"""
        try:
            # Read the CSV file
            df = pd.read_csv(self.daily_data_file, skiprows=2)  # Skip the header rows
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert to CST (handle both naive and timezone-aware dates)
            if df['Date'].dt.tz is None:
                df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert(self.cst)
            else:
                df['Date'] = df['Date'].dt.tz_convert(self.cst)
            
            # Convert price columns to float
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Add date column for grouping (date only, no time)
            df['Date'] = df['Date'].dt.date
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Print first few rows to verify data
            print("\nDaily Data Sample:")
            print(df.head())
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading daily data: {str(e)}")
            # Print the first few lines of the file to debug
            try:
                with open(self.daily_data_file, 'r') as f:
                    print("\nFirst few lines of daily data file:")
                    for i, line in enumerate(f):
                        if i < 5:  # Print first 5 lines
                            print(line.strip())
            except Exception as read_error:
                print(f"Error reading file: {str(read_error)}")
            return None
            
    def get_first_bar(self, df):
        """Get the first 60-minute bar of each day (8:30-9:30 CST)"""
        # Filter for bars between 8:30 and 9:30 CST
        first_bars = df[
            (df['Datetime'].dt.time >= time(8, 30)) & 
            (df['Datetime'].dt.time <= time(9, 30))
        ]
        
        # Group by date and get the first bar
        first_bars = first_bars.groupby('Date').first().reset_index()
        
        return first_bars
        
    def get_daily_bias(self, daily_df, date):
        """Get the daily bias for a given date"""
        # Get the current day's data
        curr_day = daily_df[daily_df['Date'] == date]
        if curr_day.empty:
            return 0, "No daily data"
            
        curr_day = curr_day.iloc[0]
        
        # Get the previous day's data
        prev_day = daily_df[daily_df['Date'] < date].iloc[-1] if len(daily_df[daily_df['Date'] < date]) > 0 else None
        
        if prev_day is None:
            return 0, "No previous day data"
            
        # Check for bullish bias conditions
        if curr_day['Close'] > prev_day['High']:
            return 1, "Close Above Prev High"
        elif curr_day['Close'] < prev_day['Low']:
            return -1, "Close Below Prev Low"
        elif curr_day['Close'] < prev_day['High'] and curr_day['Close'] > prev_day['Low'] and curr_day['High'] > prev_day['High'] and curr_day['Low'] > prev_day['Low']:
            return -1, "Failed to Close Above Prev High"
        elif curr_day['Close'] > prev_day['Low'] and curr_day['Close'] < prev_day['High'] and curr_day['High'] < prev_day['High'] and curr_day['Low'] < prev_day['Low']:
            return 1, "Failed to Close Below Prev Low"
        elif curr_day['High'] <= prev_day['High'] and curr_day['Low'] >= prev_day['Low']:
            p_up = prev_day['Close'] >= prev_day['Open']
            return 1 if p_up else -1, f"Inside Bar - Bias {'Up' if p_up else 'Down'}"
        else:
            return 0, "Outside Bar but Closed Inside"
            
    def analyze_trades(self, df, first_bars, daily_df=None):
        """Analyze potential trades based on first bar and daily bias"""
        results = []
        total_potential_trades = 0
        filtered_trades = 0
        
        # Calculate technical indicators for the entire dataframe first
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        
        ema9 = EMAIndicator(close=df['Close'], window=9)
        df['EMA9'] = ema9.ema_indicator()
        
        ema20 = EMAIndicator(close=df['Close'], window=20)
        df['EMA20'] = ema20.ema_indicator()
        
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        
        # Get first bars after calculating indicators
        first_bars = self.get_first_bar(df)
        
        for _, first_bar in first_bars.iterrows():
            date = first_bar['Date']
            first_bar_high = first_bar['High']
            first_bar_low = first_bar['Low']
            first_bar_close = first_bar['Close']
            first_bar_open = first_bar['Open']
            first_bar_volume = first_bar['Volume']
            
            # Calculate first bar characteristics
            first_bar_range = first_bar_high - first_bar_low
            first_bar_body = abs(first_bar_close - first_bar_open)
            first_bar_body_ratio = first_bar_body / first_bar_range if first_bar_range > 0 else 0
            
            # Get technical indicators for the first bar
            first_bar_rsi = first_bar['RSI']
            first_bar_ema9 = first_bar['EMA9']
            first_bar_ema20 = first_bar['EMA20']
            first_bar_atr = first_bar['ATR']
            
            # Check RSI conditions only
            if not (50 <= first_bar_rsi < 70):  # RSI must be between 50 and 70
                filtered_trades += 1
                continue
            
            # Check daily bias if enabled
            daily_bias = None
            if self.use_daily_bias:
                if daily_df is None:
                    logger.error("Daily bias is enabled but no daily data provided")
                    continue
                    
                daily_bias, bias_reason = self.get_daily_bias(daily_df, date)
            else:
                bias_reason = "Daily Bias Not Used"
            
            # Get all bars for this day after the first bar
            day_bars = df[df['Date'] == date]
            day_bars = day_bars[day_bars['Datetime'] > first_bar['Datetime']]
            
            # Skip if we don't have at least 2 bars after the first bar
            if len(day_bars) < 2:
                continue
                
            # Get the second bar
            second_bar = day_bars.iloc[0]
            
            # Count potential trade
            if second_bar['Low'] <= first_bar_low:
                total_potential_trades += 1
                
                # Use RL to decide stop loss and target
                if self.use_rl:
                    # Get state features
                    state_features = self.rl_agent.get_state_features(first_bar, second_bar, daily_bias)
                    state_key = self.rl_agent.get_state_key(state_features)
                    
                    # Choose action (combination of stop loss and target)
                    valid_actions = [(i, j) for i in range(len(self.rl_agent.stop_loss_levels)) 
                                   for j in range(len(self.rl_agent.target_levels))]
                    action = self.rl_agent.choose_action(state_key, valid_actions)
                    
                    # Get stop loss and target levels
                    stop_loss, target_price = self.rl_agent.get_action_parameters(action, state_features)
                else:
                    stop_loss = self.stop_loss
                    target_price = first_bar_high
                
                # Calculate entry price (first bar low - offset)
                entry_price = first_bar_low - self.entry_offset
                
                # Calculate stop loss price (enforce maximum 5 points)
                stop_loss = min(stop_loss, 5)  # Never allow stop loss more than 5 points
                stop_loss_price = entry_price - stop_loss
                
                # Initialize trade variables
                trade_open = True
                trade_result = None
                p_l = 0
                entry_time = second_bar['Datetime']
                exit_time = None
                exit_price = None
                
                # Track the trade through the day
                for _, bar in day_bars.iterrows():
                    if not trade_open:
                        break
                        
                    # Check for stop loss
                    if bar['Low'] <= stop_loss_price:
                        trade_open = False
                        trade_result = 'Stop Loss'
                        p_l = stop_loss_price - entry_price
                        exit_time = bar['Datetime']
                        exit_price = stop_loss_price
                        break
                        
                    # Check for target
                    if bar['High'] >= target_price:
                        trade_open = False
                        trade_result = 'Target Hit'
                        p_l = target_price - entry_price
                        exit_time = bar['Datetime']
                        exit_price = target_price
                        break
                
                # If trade is still open at end of day
                if trade_open:
                    trade_result = 'Market Close'
                    p_l = day_bars.iloc[-1]['Close'] - entry_price
                    exit_time = day_bars.iloc[-1]['Datetime']
                    exit_price = day_bars.iloc[-1]['Close']
                
                # Record the trade
                results.append({
                    'Date': date,
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Stop Loss': stop_loss_price,
                    'Target': target_price,
                    'Result': trade_result,
                    'P/L': p_l,
                    'Daily Bias': daily_bias,
                    'Bias Reason': bias_reason,
                    'RSI': first_bar_rsi,
                    'EMA9': first_bar_ema9,
                    'EMA20': first_bar_ema20,
                    'ATR': first_bar_atr
                })
                
                # Update RL agent if enabled
                if self.use_rl:
                    reward = self.rl_agent.get_reward(trade_result, p_l, stop_loss, target_price, entry_price)
                    next_state_features = self.rl_agent.get_state_features(second_bar, day_bars.iloc[1], daily_bias)
                    next_state_key = self.rl_agent.get_state_key(next_state_features)
                    next_valid_actions = [(i, j) for i in range(len(self.rl_agent.stop_loss_levels)) 
                                       for j in range(len(self.rl_agent.target_levels))]
                    self.rl_agent.update_q_value(state_key, action, reward, next_state_key, next_valid_actions)
                    self.rl_agent.update_action_stats(action, p_l)
        
        return pd.DataFrame(results), total_potential_trades, filtered_trades
        
    def generate_summary(self, results: pd.DataFrame) -> Dict:
        """Generate detailed summary statistics"""
        if results.empty:
            return {}
            
        # Basic statistics
        total_trades = len(results)
        target_hits = len(results[results['Result'] == 'Target Hit'])
        market_closes = len(results[results['Result'] == 'Market Close'])
        stop_losses = len(results[results['Result'] == 'Stop Loss'])
        
        # P/L statistics
        pl_stats = results['P/L'].describe()
        win_rate = len(results[results['P/L'] > 0]) / total_trades * 100
        
        # Calculate total points
        total_points = results['P/L'].sum()
        
        # Analyze winning trade characteristics
        winning_trades = results[results['P/L'] > 0]
        losing_trades = results[results['P/L'] <= 0]
        
        winning_stats = {
            'RSI': {
                'Mean': winning_trades['RSI'].mean(),
                'Std': winning_trades['RSI'].std(),
                'Min': winning_trades['RSI'].min(),
                'Max': winning_trades['RSI'].max()
            },
            'EMA9': {
                'Mean': winning_trades['EMA9'].mean(),
                'Std': winning_trades['EMA9'].std()
            },
            'EMA20': {
                'Mean': winning_trades['EMA20'].mean(),
                'Std': winning_trades['EMA20'].std()
            },
            'ATR': {
                'Mean': winning_trades['ATR'].mean(),
                'Std': winning_trades['ATR'].std()
            }
        }
        
        # P/L by exit type
        target_hit_pl = results[results['Result'] == 'Target Hit']['P/L']
        market_close_pl = results[results['Result'] == 'Market Close']['P/L']
        stop_loss_pl = results[results['Result'] == 'Stop Loss']['P/L']
        
        target_hit_stats = {
            'Count': len(target_hit_pl),
            'Win Rate': f"{len(target_hit_pl[target_hit_pl > 0]) / len(target_hit_pl) * 100:.2f}%" if len(target_hit_pl) > 0 else "N/A",
            'Average P/L': f"{target_hit_pl.mean():.2f}" if len(target_hit_pl) > 0 else "N/A",
            'Max Profit': f"{target_hit_pl.max():.2f}" if len(target_hit_pl) > 0 else "N/A",
            'Max Loss': f"{target_hit_pl.min():.2f}" if len(target_hit_pl) > 0 else "N/A",
            'Total Points': f"{target_hit_pl.sum():.2f}" if len(target_hit_pl) > 0 else "N/A"
        }
        
        market_close_stats = {
            'Count': len(market_close_pl),
            'Win Rate': f"{len(market_close_pl[market_close_pl > 0]) / len(market_close_pl) * 100:.2f}%" if len(market_close_pl) > 0 else "N/A",
            'Average P/L': f"{market_close_pl.mean():.2f}" if len(market_close_pl) > 0 else "N/A",
            'Max Profit': f"{market_close_pl.max():.2f}" if len(market_close_pl) > 0 else "N/A",
            'Max Loss': f"{market_close_pl.min():.2f}" if len(market_close_pl) > 0 else "N/A",
            'Total Points': f"{market_close_pl.sum():.2f}" if len(market_close_pl) > 0 else "N/A"
        }
        
        stop_loss_stats = {
            'Count': len(stop_loss_pl),
            'Win Rate': f"{len(stop_loss_pl[stop_loss_pl > 0]) / len(stop_loss_pl) * 100:.2f}%" if len(stop_loss_pl) > 0 else "N/A",
            'Average P/L': f"{stop_loss_pl.mean():.2f}" if len(stop_loss_pl) > 0 else "N/A",
            'Max Profit': f"{stop_loss_pl.max():.2f}" if len(stop_loss_pl) > 0 else "N/A",
            'Max Loss': f"{stop_loss_pl.min():.2f}" if len(stop_loss_pl) > 0 else "N/A",
            'Total Points': f"{stop_loss_pl.sum():.2f}" if len(stop_loss_pl) > 0 else "N/A"
        }
        
        # Time-based statistics
        if 'Entry Time' in results.columns and 'Exit Time' in results.columns:
            avg_holding_time = (results['Exit Time'] - results['Entry Time']).mean()
            holding_time_str = str(avg_holding_time)
        else:
            holding_time_str = "N/A"
        
        return {
            'Total Trades': total_trades,
            'Target Hits': target_hits,
            'Market Closes': market_closes,
            'Stop Losses': stop_losses,
            'Win Rate': f"{win_rate:.2f}%",
            'Average P/L': f"{pl_stats['mean']:.2f}",
            'Max Profit': f"{pl_stats['max']:.2f}",
            'Max Loss': f"{pl_stats['min']:.2f}",
            'Total Points': f"{total_points:.2f}",
            'Average Holding Time': holding_time_str,
            'Target Hit Statistics': target_hit_stats,
            'Market Close Statistics': market_close_stats,
            'Stop Loss Statistics': stop_loss_stats,
            'P/L Statistics': pl_stats,
            'Winning Trade Characteristics': winning_stats
        }
        
    def plot_results(self, results: pd.DataFrame):
        """Create visualizations of the trading results"""
        if results.empty:
            return
            
        # Set style
        plt.style.use('default')
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. P/L Distribution
        sns.histplot(data=results, x='P/L', bins=30, ax=ax1, color='skyblue')
        ax1.set_title('P/L Distribution', fontsize=12, pad=10)
        ax1.axvline(x=0, color='r', linestyle='--', label='Break Even')
        ax1.set_xlabel('Profit/Loss')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Cumulative P/L
        results['Cumulative P/L'] = results['P/L'].cumsum()
        ax2.plot(results['Date'], results['Cumulative P/L'], color='green', linewidth=2)
        ax2.set_title('Cumulative P/L Over Time', fontsize=12, pad=10)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative P/L')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Win/Loss by Result Type
        result_counts = results['Result'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Green for Target Hit, Red for Market Close
        ax3.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', colors=colors)
        ax3.set_title('Trade Results Distribution', fontsize=12, pad=10)
        
        # 4. P/L by Result Type
        sns.boxplot(data=results, x='Result', y='P/L', ax=ax4, palette=['#2ecc71', '#e74c3c'])
        ax4.set_title('P/L by Result Type', fontsize=12, pad=10)
        ax4.set_xlabel('Result Type')
        ax4.set_ylabel('Profit/Loss')
        
        # Add overall title
        plt.suptitle('First Bar Trading Strategy Analysis', fontsize=16, y=1.02)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('trading_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_analysis(self):
        """Run the complete analysis"""
        # Load data
        df = self.load_data()
        if df is None:
            return None
            
        # Load daily data if needed
        daily_df = None
        if self.use_daily_bias:
            if self.daily_data_file is None:
                logger.error("Daily bias is enabled but no daily data file provided")
                return None
            daily_df = self.load_daily_data()
            if daily_df is None:
                return None
            
        # Analyze trades
        results, total_potential_trades, filtered_trades = self.analyze_trades(df, self.get_first_bar(df), daily_df)
        
        if not results.empty:
            # Generate summary
            summary = self.generate_summary(results)
            
            # Print total points summary first
            print("\nTotal Points Summary:")
            print("=" * 50)
            print(f"Total Points Across All Trades: {summary['Total Points']}")
            print(f"Total Trades: {summary['Total Trades']}")
            print(f"Total Potential Trades: {total_potential_trades}")
            print("=" * 50)
            
            # Print detailed summary
            print("\nTrading Strategy Summary:")
            print("=" * 50)
            for key, value in summary.items():
                if key not in ['P/L Statistics', 'Target Hit Statistics', 'Market Close Statistics', 'Stop Loss Statistics', 'Total Points']:
                    print(f"{key}: {value}")
            
            print("\nTarget Hit Statistics:")
            print("-" * 30)
            for key, value in summary['Target Hit Statistics'].items():
                print(f"{key}: {value}")
                
            print("\nMarket Close Statistics:")
            print("-" * 30)
            for key, value in summary['Market Close Statistics'].items():
                print(f"{key}: {value}")
                
            print("\nStop Loss Statistics:")
            print("-" * 30)
            for key, value in summary['Stop Loss Statistics'].items():
                print(f"{key}: {value}")
            
            # Create visualizations
            self.plot_results(results)
            
            # Save results to CSV
            results.to_csv('first_bar_strategy_results.csv', index=False)
            print("\nResults saved to first_bar_strategy_results.csv")
            print("Visualizations saved to trading_results.png")
            
            # Save RL model if enabled
            if self.use_rl:
                self.rl_agent.save_model('trading_rl_model.pkl')
                print("RL model saved to trading_rl_model.pkl")
                
                # Analyze and display RL recommendations
                self.rl_agent.analyze_recommendations()
        
        return results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='First Bar Trading Strategy Analysis')
    parser.add_argument('--data_file', type=str, default="data/60m/SPX_60m.csv",
                      help='Path to the 60-minute data file')
    parser.add_argument('--daily_data_file', type=str, default="data/daily/SPX_daily.csv",
                      help='Path to the daily data file')
    parser.add_argument('--entry_offset', type=float, default=0.01,
                      help='Offset from the low price for entry')
    parser.add_argument('--use_daily_bias', action='store_true',
                      help='Whether to use daily bias for trade entry')
    parser.add_argument('--use_rl', action='store_true', default=True,
                      help='Whether to use reinforcement learning')
    parser.add_argument('--stop_loss', type=float, default=5,
                      help='Fixed stop loss in points')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize strategy
    strategy = FirstBarStrategy(
        data_file=args.data_file,
        daily_data_file=args.daily_data_file,
        entry_offset=args.entry_offset,
        use_daily_bias=args.use_daily_bias,
        use_rl=args.use_rl,
        stop_loss=args.stop_loss
    )
    
    # Run analysis
    results = strategy.run_analysis() 