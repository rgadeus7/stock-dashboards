import json
import os
from datetime import datetime
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional
import pandas as pd
from file_manager import FileManager

class SwingAnalyzer:
    def __init__(self, file_manager: FileManager):
        """Initialize the SwingAnalyzer with a FileManager instance"""
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('swing_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    def analyze_all_symbols(self) -> Dict:
        """Analyze all symbols in the configuration"""
        results = {}
        try:
            self.logger.info("Starting swing analysis...")
            
            # Get the latest market analysis file
            latest_file = self.file_manager.get_latest_file('market_analysis_')
            if not latest_file:
                self.logger.error("No market analysis file found")
                return results
            
            self.logger.info(f"Loading market analysis from {latest_file}")
            
            # Load the market analysis data
            with open(latest_file, 'r') as f:
                market_data = json.load(f)
            
            for symbol, intervals in market_data.items():
                self.logger.info(f"Analyzing {symbol}...")
                for interval, analysis in intervals.items():
                    try:
                        if isinstance(analysis, dict) and 'prices' in analysis:
                            # Run swing analysis
                            swing_result = self.find_swing_points(analysis)
                            if symbol not in results:
                                results[symbol] = {}
                            results[symbol][interval] = swing_result
                            self.logger.info(f"Completed swing analysis for {symbol} at {interval} interval")
                        else:
                            self.logger.warning(f"Invalid analysis data for {symbol} at {interval} interval")
                            
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol} at {interval} interval: {str(e)}")
                        if symbol not in results:
                            results[symbol] = {}
                        results[symbol][interval] = {'error': str(e)}
            
            # Save results
            if results:
                self.save_results(results)
                self.logger.info("Swing analysis completed successfully")
            else:
                self.logger.warning("No swing analysis results were generated")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in swing analysis: {str(e)}")
            return results
    
    def find_swing_points(self, data: Dict) -> Dict:
        """Find swing points in the price data"""
        try:
            # Extract price data
            prices = data.get('prices', [])
            if not prices:
                return {'error': 'No price data available'}
            
            # Convert to DataFrame
            df = pd.DataFrame(prices)
            
            # Calculate moving averages for trend context
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Determine overall trend
            overall_trend = 'neutral'
            if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
                overall_trend = 'up'
            elif df['MA20'].iloc[-1] < df['MA50'].iloc[-1]:
                overall_trend = 'down'
            
            # Find swing highs and lows with trend context
            swing_highs = []
            swing_lows = []
            
            # Use a 5-point window for more reliable swing points
            window = 5
            for i in range(window, len(df) - window):
                # Check for swing high
                is_swing_high = True
                for j in range(1, window + 1):
                    if df['High'].iloc[i] <= df['High'].iloc[i-j] or df['High'].iloc[i] <= df['High'].iloc[i+j]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append({
                        'date': df['Date'].iloc[i],
                        'price': float(df['High'].iloc[i]),
                        'trend': overall_trend
                    })
                
                # Check for swing low
                is_swing_low = True
                for j in range(1, window + 1):
                    if df['Low'].iloc[i] >= df['Low'].iloc[i-j] or df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append({
                        'date': df['Date'].iloc[i],
                        'price': float(df['Low'].iloc[i]),
                        'trend': overall_trend
                    })
            
            # Filter swing points based on trend
            if overall_trend == 'up':
                # In uptrend, focus on higher lows
                swing_lows = [low for low in swing_lows if low['price'] > df['MA20'].iloc[-1]]
            elif overall_trend == 'down':
                # In downtrend, focus on lower highs
                swing_highs = [high for high in swing_highs if high['price'] < df['MA20'].iloc[-1]]
            
            # Determine current trend based on swing points
            trend = 'neutral'
            if swing_highs and swing_lows:
                last_high = swing_highs[-1]['price']
                last_low = swing_lows[-1]['price']
                
                # Consider the overall trend in determining the current trend
                if overall_trend == 'up' and last_high > last_low:
                    trend = 'up'
                elif overall_trend == 'down' and last_high < last_low:
                    trend = 'down'
                else:
                    trend = 'neutral'
            
            return {
                'trend': trend,
                'overall_trend': overall_trend,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'last_swing_high': swing_highs[-1] if swing_highs else None,
                'last_swing_low': swing_lows[-1] if swing_lows else None,
                'swing_high_count': len(swing_highs),
                'swing_low_count': len(swing_lows),
                'ma20': float(df['MA20'].iloc[-1]),
                'ma50': float(df['MA50'].iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error finding swing points: {str(e)}")
            return {'error': str(e)}
    
    def save_results(self, results: Dict) -> str:
        """Save swing analysis results to JSON file"""
        try:
            # Create filename with current date
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"swing_analysis_{date_str}.json"
            
            # Save to analysis directory
            file_path = os.path.join(self.file_manager.analysis_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Saved swing analysis results to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving swing analysis results: {str(e)}")
            return ""

if __name__ == "__main__":
    from file_manager import FileManager
    
    # Initialize FileManager
    file_manager = FileManager()
    
    # Initialize SwingAnalyzer with FileManager
    analyzer = SwingAnalyzer(file_manager)
    analyzer.analyze_all_symbols() 