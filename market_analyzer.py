import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Optional
import logging
import json
from file_manager import FileManager

class MarketAnalyzer:
    def __init__(self, file_manager: FileManager):
        """Initialize the MarketAnalyzer with a FileManager instance"""
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
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
            '1mo': '1mo'
        }
        return interval_map.get(interval, '1d')  # Default to daily if not found
    
    def load_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data for a specific symbol and interval"""
        try:
            file_path = os.path.join(self.file_manager.data_dir, interval, f"{symbol}_{interval}.csv")
            
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
                
                # Try common date formats
                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']
                for fmt in date_formats:
                    try:
                        data['Date'] = pd.to_datetime(data['Date'], format=fmt, errors='coerce')
                        # Check if we have any valid dates
                        if data['Date'].notna().any():
                            break
                    except ValueError:
                        continue
                
                # If we still have NaT values, try without format
                if data['Date'].isna().any():
                    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                
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
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given data"""
        try:
            # Moving Averages
            data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
            
            # RSI (requires at least 14 periods for calculation)
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
            else:
                data['RSI'] = np.nan
                self.logger.warning("Insufficient data for RSI calculation (requires at least 14 periods)")
            
            # MACD (requires at least 26 periods for calculation)
            if len(data) >= 26:
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            else:
                data['EMA12'] = np.nan
                data['EMA26'] = np.nan
                data['MACD'] = np.nan
                data['Signal'] = np.nan
                self.logger.warning("Insufficient data for MACD calculation (requires at least 26 periods)")
            
            return data
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            # Return the data with NaN values for indicators if calculation fails
            for col in ['MA20', 'MA50', 'MA200', 'RSI', 'EMA12', 'EMA26', 'MACD', 'Signal']:
                if col not in data.columns:
                    data[col] = np.nan
            return data
    
    def analyze_symbol(self, symbol: str, interval: str = 'daily') -> Dict:
        """Analyze a single symbol at the specified interval"""
        try:
            data = self.load_data(symbol, interval)
            if data is None or data.empty:
                self.logger.warning(f"No data available for {symbol} at {interval} interval")
                return {}
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Determine trend
            trend = self._determine_trend(data)
            
            # Find support and resistance levels
            support_resistance = self._find_support_resistance(data)
            
            # Calculate volatility
            volatility = self._calculate_volatility(data)
            
            # Prepare analysis results
            analysis = {
                'symbol': symbol,
                'interval': interval,
                'trend': trend,
                'support_resistance': support_resistance,
                'volatility': volatility,
                'last_price': float(data['Close'].iloc[-1]),
                'last_update': data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(data.index[-1]) else None,
                'prices': [{
                    'Date': date.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(date) else None,
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': float(row['Volume'])
                } for date, row in data.iterrows() if pd.notna(date)]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} at {interval} interval: {str(e)}")
            return {}
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine the current trend based on moving averages"""
        latest = data.iloc[-1]
        
        if latest['Close'] > latest['MA20'] > latest['MA50'] > latest['MA200']:
            return "Strong Uptrend"
        elif latest['Close'] > latest['MA20'] > latest['MA50']:
            return "Uptrend"
        elif latest['Close'] < latest['MA20'] < latest['MA50'] < latest['MA200']:
            return "Strong Downtrend"
        elif latest['Close'] < latest['MA20'] < latest['MA50']:
            return "Downtrend"
        else:
            return "Sideways"
    
    def _find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Find recent support and resistance levels"""
        recent_data = data['Close'].tail(window)
        return {
            'support': float(recent_data.min()),
            'resistance': float(recent_data.max())
        }
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate recent volatility"""
        returns = data['Close'].pct_change()
        return float(returns.tail(window).std() * np.sqrt(252) * 100)  # Annualized volatility in percentage
    
    def analyze_all_symbols(self) -> Dict:
        """Analyze all symbols from the configuration"""
        try:
            config = self.file_manager.load_config()
            if not config or 'tickers' not in config:
                self.logger.error("No tickers configuration found")
                return {}
            
            analysis_results = {}
            for symbol, settings in config['tickers'].items():
                for interval in settings.get('analysis_timeframes', ['daily']):
                    analysis = self.analyze_symbol(symbol, interval)
                    if analysis:
                        if symbol not in analysis_results:
                            analysis_results[symbol] = {}
                        analysis_results[symbol][interval] = analysis
            
            # Save analysis results
            self.save_analysis_to_json(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing all symbols: {str(e)}")
            return {}
    
    def save_analysis_to_json(self, analysis_results: Dict) -> str:
        """Save analysis results to a JSON file"""
        try:
            # Create filename with current date
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"market_analysis_{date_str}.json"
            
            # Save to analysis_output directory
            file_path = os.path.join(self.file_manager.analysis_dir, filename)
            
            # Convert datetime objects to strings
            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.strftime("%Y-%m-%d %H:%M:%S")
                return obj
            
            # Save the analysis results
            with open(file_path, 'w') as f:
                json.dump(analysis_results, f, default=datetime_handler, indent=4)
            
            self.logger.info(f"Analysis results saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            return ""
    
    def run_analysis(self):
        """Run the complete market analysis process"""
        try:
            self.logger.info("Starting market analysis...")
            results = self.analyze_all_symbols()
            if results:
                self.logger.info("Market analysis completed successfully")
            else:
                self.logger.warning("No analysis results were generated")
        except Exception as e:
            self.logger.error(f"Error running market analysis: {str(e)}")

if __name__ == "__main__":
    from file_manager import FileManager
    
    # Initialize FileManager
    file_manager = FileManager()
    
    # Initialize MarketAnalyzer with FileManager
    analyzer = MarketAnalyzer(file_manager)
    analyzer.run_analysis() 