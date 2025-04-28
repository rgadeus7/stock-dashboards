import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Optional, Tuple
import logging
import json
from file_manager import FileManager
from ticker_manager import TickerManager
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class MarketAnalyzer:
    def __init__(self, file_manager: FileManager, ticker_manager: TickerManager, config_path: str = 'config/tickers.yaml'):
        """Initialize the MarketAnalyzer with FileManager and TickerManager instances"""
        self.file_manager = file_manager
        self.ticker_manager = ticker_manager
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Create analysis_output directory if it doesn't exist
        self.analysis_output_dir = os.path.join(os.path.dirname(self.file_manager.data_dir), 'analysis_output')
        os.makedirs(self.analysis_output_dir, exist_ok=True)
        
        # Load ticker configuration
        self.ticker_config = self.ticker_manager.get_ticker_config()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('market_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    def _get_yahoo_interval(self, interval: str) -> str:
        """Map interval name to Yahoo Finance interval code"""
        interval_map = {
            '2h': '2h',
            '4h': '4h',
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo',
            '60m': '60m',
            '15m': '15m',
            '30m': '30m'
        }
        return interval_map.get(interval, '1d')  # Default to daily if not found
    
    def load_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data for a specific symbol and interval"""
        try:
            # Map interval to directory name
            interval_dir_map = {
                '1d': 'daily',
                '1wk': 'weekly',
                '1mo': 'monthly',
                '2h': '2h',
                '4h': '4h',
                '15m': '15m',
                '30m': '30m',
                '60m': '60m'
            }
            
            # Get the directory name for the interval
            interval_dir = interval_dir_map.get(interval, interval)
            
            # Construct the correct file path with interval subdirectory
            file_path = os.path.join(self.file_manager.data_dir, interval_dir, f"{symbol}_{interval_dir}.csv")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"No data file found for {symbol} at {interval} interval")
                return None
                
            # Read the CSV file, skipping the header rows
            data = pd.read_csv(file_path, skiprows=3)
            
            # Check if we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns in data file for {symbol}: {missing_columns}")
                return None
            
            # Convert numeric columns to float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    # Remove any non-numeric characters and convert to float
                    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Convert the date column to datetime
            try:
                # First, clean the date column
                data['Date'] = data['Date'].astype(str).str.strip()
                
                # Check if this is a minute-based interval
                is_minute_interval = interval in ['15m', '30m', '60m']
                
                if is_minute_interval:
                    # For minute-based intervals, use timezone-aware format
                    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
                else:
                    # For daily/weekly/monthly data
                    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
                
                # Drop rows with NaT dates
                data = data[data['Date'].notna()]
                
                if data.empty:
                    self.logger.error(f"No valid dates found in data for {symbol}")
                    return None
            
            except Exception as e:
                self.logger.error(f"Date parsing error for {symbol}: {str(e)}")
                return None
            
            # Set the date column as index
            data.set_index('Date', inplace=True)
            
            # Sort by date
            data.sort_index(inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} at {interval} interval: {str(e)}")
            return None
    
    def calculate_pivot_points(self, df: pd.DataFrame, left: int = 4, right: int = 2) -> pd.DataFrame:
        """Calculate pivot high and low points"""
        df['pivot_high'] = np.nan
        df['pivot_low'] = np.nan
        
        for i in range(left, len(df) - right):
            # Check for pivot high
            if all(df['High'].iloc[i] > df['High'].iloc[i-left:i]) and \
               all(df['High'].iloc[i] > df['High'].iloc[i+1:i+right+1]):
                df.loc[df.index[i], 'pivot_high'] = df['High'].iloc[i]
            
            # Check for pivot low
            if all(df['Low'].iloc[i] < df['Low'].iloc[i-left:i]) and \
               all(df['Low'].iloc[i] < df['Low'].iloc[i+1:i+right+1]):
                df.loc[df.index[i], 'pivot_low'] = df['Low'].iloc[i]
        
        # Forward fill the last non-NaN values using ffill()
        df['pivot_high'] = df['pivot_high'].ffill()
        df['pivot_low'] = df['pivot_low'].ffill()
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        try:
            # Calculate RSI
            df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
            # Calculate SMA
            df['sma_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
            df['sma_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
            # Calculate Bollinger Bands
            bollinger = BollingerBands(df['Close'], window=20, window_dev=2)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            # Calculate Pivot Points
            df = self.calculate_pivot_points(df)
            # Calculate percentage distances from current price
            current_price = df['Close']
            # SMA percentages
            df['sma_20_pct'] = ((current_price - df['sma_20']) / current_price) * 100
            df['sma_50_pct'] = ((current_price - df['sma_50']) / current_price) * 100
            df['sma_200_pct'] = ((current_price - df['sma_200']) / current_price) * 100
            # Bollinger Bands percentages
            df['bb_upper_pct'] = ((current_price - df['bb_upper']) / current_price) * 100
            df['bb_lower_pct'] = ((current_price - df['bb_lower']) / current_price) * 100
            # Pivot points percentages
            df['pivot_high_pct'] = ((current_price - df['pivot_high']) / current_price) * 100
            df['pivot_low_pct'] = ((current_price - df['pivot_low']) / current_price) * 100
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def analyze_market_health(self, df: pd.DataFrame, lookback_days: int = 90) -> Dict:
        """Analyze overall market health based on technical indicators
        
        Args:
            df: DataFrame containing market data
            lookback_days: Number of days of historical data to include (default: 90)
        """
        # Get the most recent data
        latest = df.iloc[-1]
        
        # Calculate trend strength
        trend_strength = 0
        if latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
            trend_strength = 1  # Strong uptrend
        elif latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
            trend_strength = -1  # Strong downtrend
        
        # Calculate momentum
        momentum = 0
        if latest['rsi'] > 70:
            momentum = 1  # Overbought
        elif latest['rsi'] < 30:
            momentum = -1  # Oversold
        
        # Calculate volatility
        volatility = latest['bb_upper'] / latest['bb_middle'] * 100
        
        # Get historical OHLC data
        historical_data = df.tail(lookback_days)
        daily_data = []
        for date, row in historical_data.iterrows():
            daily_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2)
            })
        
        return {
            'trend_strength': trend_strength,
            'momentum': momentum,
            'volatility': round(volatility, 2),
            'rsi': round(latest['rsi'], 2),
            'sma_20': round(latest['sma_20'], 2),
            'sma_50': round(latest['sma_50'], 2),
            'sma_200': round(latest['sma_200'], 2),
            'bb_upper': round(latest['bb_upper'], 2),
            'bb_lower': round(latest['bb_lower'], 2),
            'pivot_high': round(latest['pivot_high'], 2),
            'pivot_low': round(latest['pivot_low'], 2),
            'sma_20_pct': round(latest['sma_20_pct'], 2),
            'sma_50_pct': round(latest['sma_50_pct'], 2),
            'sma_200_pct': round(latest['sma_200_pct'], 2),
            'bb_upper_pct': round(latest['bb_upper_pct'], 2),
            'bb_lower_pct': round(latest['bb_lower_pct'], 2),
            'pivot_high_pct': round(latest['pivot_high_pct'], 2),
            'pivot_low_pct': round(latest['pivot_low_pct'], 2),
            'current_price': round(latest['Close'], 2),
            'historical_data': {
                'lookback_days': lookback_days,
                'data': daily_data
            }
        }
    
    def analyze_ticker(self, symbol: str, interval: str = '1d', lookback_days: int = 90) -> Dict:
        """Analyze market health for a specific ticker
        
        Args:
            symbol: Ticker symbol to analyze
            interval: Time interval for analysis (default: '1d')
            lookback_days: Number of days of historical data to include (default: 90)
        """
        try:
            # Load existing data instead of downloading again
            df = self.load_data(symbol, interval)
            
            if df is None:
                error_msg = f"No data available for {symbol} at {interval} interval"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Analyze market health
            health_metrics = self.analyze_market_health(df, lookback_days)
            
            # Add ticker info
            ticker_info = self.ticker_manager.get_ticker_info(symbol)
            health_metrics['ticker_info'] = {
                'symbol': symbol,
                'name': ticker_info.get('name', symbol),
                'description': ticker_info.get('description', ''),
                'interval': interval,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'data_start': df.index[0].strftime('%Y-%m-%d'),
                'data_end': df.index[-1].strftime('%Y-%m-%d')
            }
            
            # Save results
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"market_analysis_{symbol}_{interval}_{date_str}.json"
            file_path = os.path.join(self.analysis_output_dir, filename)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(health_metrics, f, indent=4)
            
            self.logger.info(f"Saved {symbol} market analysis to {file_path}")
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise
    
    def analyze_all_tickers(self) -> Dict[str, Dict]:
        """Analyze market health for all tickers using their configured timeframes"""
        results = {}
        available_tickers = self.ticker_manager.get_available_tickers()
        total_tickers = len(available_tickers)
        
        print(f"\nStarting analysis of {total_tickers} tickers...")
        
        for i, symbol in enumerate(available_tickers, 1):
            try:
                print(f"\nProcessing ticker {i}/{total_tickers}: {symbol}")
                
                # Get analysis timeframes for this ticker
                ticker_info = self.ticker_config['tickers'].get(symbol, {})
                analysis_timeframes = ticker_info.get('analysis_timeframes', ['daily'])
                
                results[symbol] = {}
                for timeframe in analysis_timeframes:
                    # Map timeframe to interval
                    interval = self._get_yahoo_interval(timeframe)
                    if interval:
                        # Print the file name being used
                        interval_dir = self._get_interval_dir(interval)
                        file_name = f"{symbol}_{interval_dir}.csv"
                        print(f"  Analyzing {timeframe} timeframe")
                        print(f"  Using data file: {file_name}")
                        
                        # Load data first to check if it exists
                        df = self.load_data(symbol, interval)
                        if df is None or df.empty:
                            print(f"  Warning: No data available for {symbol} at {timeframe} timeframe")
                            continue
                            
                        # Only proceed with analysis if we have data
                        results[symbol][timeframe] = self.analyze_market(symbol, interval)
                    else:
                        self.logger.warning(f"Invalid timeframe {timeframe} for {symbol}")
                        
                print(f"  Completed analysis for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Save summary of all tickers' analysis
        date_str = datetime.now().strftime("%Y-%m-%d")
        summary_filename = "market_analysis_summary.json"
        summary_path = os.path.join(self.analysis_output_dir, summary_filename)
        
        summary_data = {
            'analysis_date': date_str,
            'tickers': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        self.logger.info(f"Saved market analysis summary to {summary_path}")
        return results

    def _get_interval_dir(self, interval: str) -> str:
        """Map interval to directory name"""
        interval_dir_map = {
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly',
            '2h': '2h',
            '4h': '4h',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly'
        }
        return interval_dir_map.get(interval, interval)

    def analyze_price_action(self, df: pd.DataFrame) -> str:
        """Analyze price action and return the bias reason"""
        # Get the last 2 days of data
        last_2_days = df.iloc[-2:]
        prev_day = last_2_days.iloc[0]
        curr_day = last_2_days.iloc[1]
        
        # Calculate price action bias reason
        if curr_day['Close'] > prev_day['High']:
            return "Close Above Prev High"
        elif curr_day['Close'] < prev_day['Low']:
            return "Close Below Prev Low"
        elif curr_day['Close'] < prev_day['High'] and curr_day['Close'] > prev_day['Low'] and curr_day['High'] > prev_day['High'] and curr_day['Low'] > prev_day['Low']:
            return "Failed to Close Above Prev High"
        elif curr_day['Close'] > prev_day['Low'] and curr_day['Close'] < prev_day['High'] and curr_day['High'] < prev_day['High'] and curr_day['Low'] < prev_day['Low']:
            return "Failed to Close Below Prev Low"
        elif curr_day['High'] <= prev_day['High'] and curr_day['Low'] >= prev_day['Low']:
            p_up = prev_day['Close'] >= prev_day['Open']
            return f"Inside Bar - Bias {'Up' if p_up else 'Down'}"
        else:
            return "Outside Bar but Closed Inside"

    def analyze_market(self, symbol: str, interval: str) -> Dict:
        """
        Analyze market data for a specific symbol and interval
        
        Args:
            symbol: The symbol to analyze
            interval: The timeframe interval
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load data
            df = self.load_data(symbol, interval)
            if df is None or df.empty:
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get latest data
            latest = df.iloc[-1]
            
            # Calculate bias
            bias, bias_reason = self.calculate_bias(df)
            
            # Calculate percentage changes
            sma_20_pct = ((latest['Close'] - latest['sma_20']) / latest['sma_20']) * 100
            sma_50_pct = ((latest['Close'] - latest['sma_50']) / latest['sma_50']) * 100
            sma_200_pct = ((latest['Close'] - latest['sma_200']) / latest['sma_200']) * 100
            bb_upper_pct = ((latest['Close'] - latest['bb_upper']) / latest['bb_upper']) * 100
            bb_lower_pct = ((latest['Close'] - latest['bb_lower']) / latest['bb_lower']) * 100
            pivot_high_pct = ((latest['Close'] - latest['pivot_high']) / latest['pivot_high']) * 100
            pivot_low_pct = ((latest['Close'] - latest['pivot_low']) / latest['pivot_low']) * 100
            
            return {
                'symbol': symbol,
                'interval': interval,
                'current_price': latest['Close'],
                'rsi': latest['rsi'],
                'bias': bias,
                'bias_reason': bias_reason,
                'sma_20': latest['sma_20'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'bb_upper': latest['bb_upper'],
                'bb_lower': latest['bb_lower'],
                'pivot_high': latest['pivot_high'],
                'pivot_low': latest['pivot_low'],
                'sma_20_pct': sma_20_pct,
                'sma_50_pct': sma_50_pct,
                'sma_200_pct': sma_200_pct,
                'bb_upper_pct': bb_upper_pct,
                'bb_lower_pct': bb_lower_pct,
                'pivot_high_pct': pivot_high_pct,
                'pivot_low_pct': pivot_low_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} at {interval} timeframe: {str(e)}")
            return None

    def calculate_bias(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Calculate market bias based on price action and indicators
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            Tuple of (bias, reason) where bias is 'bullish', 'bearish', or 'neutral'
        """
        latest = df.iloc[-1]
        
        # Get price action bias first
        price_action_reason = self.analyze_price_action(df)
        
        # Check RSI with more granular conditions
        if latest['rsi'] > 80:
            return 'bearish', f'RSI Extremely Overbought - {price_action_reason}'
        elif latest['rsi'] > 70:
            return 'bearish', f'RSI Overbought - {price_action_reason}'
        elif latest['rsi'] < 20:
            return 'bullish', f'RSI Extremely Oversold - {price_action_reason}'
        elif latest['rsi'] < 30:
            return 'bullish', f'RSI Oversold - {price_action_reason}'
            
        # Check SMA alignment with more conditions
        if latest['Close'] > latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
            if latest['sma_20_pct'] > 0 and latest['sma_50_pct'] > 0:
                return 'bullish', f'Strong Uptrend (SMA Alignment Above) - {price_action_reason}'
            else:
                return 'bullish', f'Uptrend (SMA Alignment) - {price_action_reason}'
        elif latest['Close'] < latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
            if latest['sma_20_pct'] < 0 and latest['sma_50_pct'] < 0:
                return 'bearish', f'Strong Downtrend (SMA Alignment Below) - {price_action_reason}'
            else:
                return 'bearish', f'Downtrend (SMA Alignment) - {price_action_reason}'
            
        # Check Bollinger Bands with more conditions
        if latest['Close'] > latest['bb_upper']:
            if latest['bb_upper_pct'] < -2:
                return 'bearish', f'Price Significantly Above Upper Bollinger Band - {price_action_reason}'
            else:
                return 'bearish', f'Price Above Upper Bollinger Band - {price_action_reason}'
        elif latest['Close'] < latest['bb_lower']:
            if latest['bb_lower_pct'] > 2:
                return 'bullish', f'Price Significantly Below Lower Bollinger Band - {price_action_reason}'
            else:
                return 'bullish', f'Price Below Lower Bollinger Band - {price_action_reason}'
            
        # Check Pivot Points with more conditions
        if latest['Close'] > latest['pivot_high']:
            if latest['pivot_high_pct'] < -2:
                return 'bullish', f'Price Significantly Above Pivot High - {price_action_reason}'
            else:
                return 'bullish', f'Price Above Pivot High - {price_action_reason}'
        elif latest['Close'] < latest['pivot_low']:
            if latest['pivot_low_pct'] > 2:
                return 'bearish', f'Price Significantly Below Pivot Low - {price_action_reason}'
            else:
                return 'bearish', f'Price Below Pivot Low - {price_action_reason}'
            
        # Check for potential reversal patterns
        if latest['rsi'] > 60 and latest['Close'] < latest['sma_20']:
            return 'bearish', f'Potential Bearish Reversal (RSI High, Price Below SMA20) - {price_action_reason}'
        elif latest['rsi'] < 40 and latest['Close'] > latest['sma_20']:
            return 'bullish', f'Potential Bullish Reversal (RSI Low, Price Above SMA20) - {price_action_reason}'
            
        # If no clear technical bias, use price action only
        if "Up" in price_action_reason:
            return 'bullish', price_action_reason
        elif "Down" in price_action_reason:
            return 'bearish', price_action_reason
            
        return 'neutral', price_action_reason

if __name__ == "__main__":
    from file_manager import FileManager
    from ticker_manager import TickerManager
    
    # Initialize managers
    file_manager = FileManager()
    ticker_manager = TickerManager(file_manager)  # Pass file_manager to TickerManager
    
    # Initialize analyzer
    analyzer = MarketAnalyzer(file_manager, ticker_manager)
    
    try:
        # Analyze all tickers
        results = analyzer.analyze_all_tickers()
        
        # Print results
        print("\nMarket Analysis Results:")
        print("=" * 50)
        for symbol, data in results.items():
            print(f"\n{symbol}:")
            for timeframe, analysis in data.items():
                print(f"\n{timeframe} timeframe:")
                print(f"Current Price: {analysis['current_price']:.2f}")
                print(f"RSI: {analysis['rsi']:.2f}")
                print(f"Bias: {analysis['bias']} - {analysis['bias_reason']}")
                print("-" * 30)
                
    except Exception as e:
        print(f"Error running market analysis: {str(e)}") 