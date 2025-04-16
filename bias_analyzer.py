import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Optional
import logging
import json
from file_manager import FileManager
from ticker_manager import TickerManager

class BiasAnalyzer:
    def __init__(self, file_manager: FileManager, ticker_manager: TickerManager):
        """Initialize the BiasAnalyzer with FileManager and TickerManager instances"""
        self.file_manager = file_manager
        self.ticker_manager = ticker_manager
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
                logging.FileHandler('bias_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    def calculate_bias_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bias-specific technical indicators"""
        # Calculate price momentum
        df['Price_Momentum'] = df['Close'].pct_change(periods=20)
        
        # Calculate volume momentum
        df['Volume_Momentum'] = df['Volume'].pct_change(periods=20)
        
        # Calculate price volatility
        df['Price_Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Calculate volume volatility
        df['Volume_Volatility'] = df['Volume'].rolling(window=20).std() / df['Volume'].rolling(window=20).mean()
        
        # Calculate price trend strength
        df['Trend_Strength'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
        
        # Calculate volume trend strength
        df['Volume_Trend'] = (df['Volume'] - df['Volume'].rolling(window=20).mean()) / df['Volume'].rolling(window=20).std()
        
        return df
    
    def analyze_market_bias(self, df: pd.DataFrame) -> Dict:
        """Analyze market bias based on technical indicators and price action"""
        # Get the last 2 days of data
        last_2_days = df.iloc[-2:]
        prev_day = last_2_days.iloc[0]
        curr_day = last_2_days.iloc[1]
        
        # Calculate price action bias
        price_action_bias = 0
        bias_reason = ""
        
        if curr_day['Close'] > prev_day['High']:
            price_action_bias = 1
            bias_reason = "Close Above Prev High"
        elif curr_day['Close'] < prev_day['Low']:
            price_action_bias = -1
            bias_reason = "Close Below Prev Low"
        elif curr_day['Close'] < prev_day['High'] and curr_day['Close'] > prev_day['Low'] and curr_day['High'] > prev_day['High'] and curr_day['Low'] > prev_day['Low']:
            price_action_bias = -1
            bias_reason = "Failed to Close Above Prev High"
        elif curr_day['Close'] > prev_day['Low'] and curr_day['Close'] < prev_day['High'] and curr_day['High'] < prev_day['High'] and curr_day['Low'] < prev_day['Low']:
            price_action_bias = 1
            bias_reason = "Failed to Close Below Prev Low"
        elif curr_day['High'] <= prev_day['High'] and curr_day['Low'] >= prev_day['Low']:
            p_up = prev_day['Close'] >= prev_day['Open']
            price_action_bias = 1 if p_up else -1
            bias_reason = f"Inside Bar - Bias {'Up' if p_up else 'Down'}"
        else:
            price_action_bias = 0
            bias_reason = "Outside Bar but Closed Inside"
        
        # Calculate volume bias
        volume_bias = 0
        volume_change = 0
        if prev_day['Volume'] > 0:  # Check for zero volume
            volume_change = (curr_day['Volume'] - prev_day['Volume']) / prev_day['Volume']
            if curr_day['Volume'] > prev_day['Volume'] * 1.5:
                volume_bias = 1  # Strong volume increase
            elif curr_day['Volume'] < prev_day['Volume'] * 0.5:
                volume_bias = -1  # Strong volume decrease
        
        # Calculate momentum bias
        momentum_bias = 0
        price_change = (curr_day['Close'] - prev_day['Close']) / prev_day['Close']
        if price_change > 0.02:  # 2% increase
            momentum_bias = 1  # Strong positive momentum
        elif price_change < -0.02:  # 2% decrease
            momentum_bias = -1  # Strong negative momentum
        
        # Calculate volatility bias
        volatility_bias = 0
        daily_range = (curr_day['High'] - curr_day['Low']) / curr_day['Close']
        if daily_range > 0.03:  # 3% daily range
            volatility_bias = 1  # High volatility
        elif daily_range < 0.01:  # 1% daily range
            volatility_bias = -1  # Low volatility
        
        # Calculate overall bias score
        bias_score = (price_action_bias + volume_bias + momentum_bias + volatility_bias) / 4
        
        return {
            'price_action_bias': price_action_bias,
            'bias_reason': bias_reason,
            'volume_bias': volume_bias,
            'momentum_bias': momentum_bias,
            'volatility_bias': volatility_bias,
            'overall_bias': bias_score,
            'price_change': price_change,
            'daily_range': daily_range,
            'volume_change': volume_change,
            'current_price': curr_day['Close'],
            'prev_day': {
                'open': prev_day['Open'],
                'high': prev_day['High'],
                'low': prev_day['Low'],
                'close': prev_day['Close'],
                'volume': prev_day['Volume']
            },
            'current_day': {
                'open': curr_day['Open'],
                'high': curr_day['High'],
                'low': curr_day['Low'],
                'close': curr_day['Close'],
                'volume': curr_day['Volume']
            }
        }
    
    def analyze_ticker(self, symbol: str, interval: str = '1d') -> Dict:
        """Analyze bias for a specific ticker"""
        try:
            # Get ticker data
            df = self.ticker_manager.get_ticker_data(symbol, interval)
            
            # Calculate bias indicators
            df = self.calculate_bias_indicators(df)
            
            # Analyze bias
            bias_metrics = self.analyze_market_bias(df)
            
            # Add ticker info
            ticker_info = self.ticker_manager.get_ticker_info(symbol)
            bias_metrics['ticker_info'] = {
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
            filename = f"bias_analysis_{symbol}_{interval}_{date_str}.json"
            file_path = os.path.join(self.analysis_output_dir, filename)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(bias_metrics, f, indent=4)
            
            self.logger.info(f"Saved {symbol} bias analysis to {file_path}")
            return bias_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise
    
    def _get_yahoo_interval(self, timeframe: str) -> Optional[str]:
        """Map timeframe name to Yahoo Finance interval code"""
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        return interval_map.get(timeframe.lower())
    
    def analyze_all_tickers(self) -> Dict[str, Dict]:
        """Analyze market bias for all tickers using their configured timeframes"""
        results = {}
        for symbol in self.ticker_manager.get_available_tickers():
            try:
                # Get analysis timeframes for this ticker
                ticker_info = self.ticker_config['tickers'].get(symbol, {})
                analysis_timeframes = ticker_info.get('analysis_timeframes', ['daily'])
                
                results[symbol] = {}
                for timeframe in analysis_timeframes:
                    # Map timeframe to interval
                    interval = self._get_yahoo_interval(timeframe)
                    if interval:
                        # Analyze the ticker
                        bias_metrics = self.analyze_ticker(symbol, interval)
                        
                        # Map the interval back to timeframe for the summary
                        results[symbol][timeframe] = bias_metrics
                    else:
                        self.logger.warning(f"Invalid timeframe {timeframe} for {symbol}")
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Save summary of all tickers' analysis
        date_str = datetime.now().strftime("%Y-%m-%d")
        summary_filename = f"bias_analysis_summary_{date_str}.json"
        summary_path = os.path.join(self.analysis_output_dir, summary_filename)
        
        summary_data = {
            'analysis_date': date_str,
            'tickers': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        self.logger.info(f"Saved bias analysis summary to {summary_path}")
        return results

if __name__ == "__main__":
    from file_manager import FileManager
    from ticker_manager import TickerManager
    
    # Initialize managers
    file_manager = FileManager()
    ticker_manager = TickerManager(file_manager)
    analyzer = BiasAnalyzer(file_manager, ticker_manager)
    
    try:
        # Run analysis for all tickers
        print("\nAnalyzing market bias for all tickers...")
        results = analyzer.analyze_all_tickers()
        
        # Print summary for each ticker
        for symbol, result in results.items():
            print(f"\n=== Market Bias Analysis for {symbol} ===")
            print(f"Name: {result['ticker_info']['name']}")
            print(f"Period: {result['ticker_info']['data_start']} to {result['ticker_info']['data_end']}")
            
            print("\nBias Indicators:")
            print(f"Current Price: {result['current_price']:.2f}")
            print(f"Price Change: {result['price_change']:.2%}")
            print(f"Volume Change: {result['volume_change']:.2%}")
            print(f"Daily Range: {result['daily_range']:.2%}")
            
            print("\nMarket Bias:")
            print(f"Price Action Bias: {'Bullish' if result['price_action_bias'] > 0 else 'Bearish' if result['price_action_bias'] < 0 else 'Neutral'}")
            print(f"Bias Reason: {result['bias_reason']}")
            print(f"Volume Bias: {'Increasing' if result['volume_bias'] > 0 else 'Decreasing' if result['volume_bias'] < 0 else 'Neutral'}")
            print(f"Momentum Bias: {'Positive' if result['momentum_bias'] > 0 else 'Negative' if result['momentum_bias'] < 0 else 'Neutral'}")
            print(f"Volatility Bias: {'High' if result['volatility_bias'] > 0 else 'Low' if result['volatility_bias'] < 0 else 'Moderate'}")
            print(f"Overall Bias: {'Bullish' if result['overall_bias'] > 0.25 else 'Bearish' if result['overall_bias'] < -0.25 else 'Neutral'}")
        
        print("\nAnalysis completed successfully!")
        print(f"Individual ticker analysis saved to {analyzer.analysis_output_dir}")
        date_str = datetime.now().strftime('%Y-%m-%d')
        print(f"Summary analysis saved to {os.path.join(analyzer.analysis_output_dir, f'bias_analysis_summary_{date_str}.json')}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}") 