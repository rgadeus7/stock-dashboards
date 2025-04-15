import os
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
import logging
import yaml
from file_manager import FileManager
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, file_manager: FileManager):
        """Initialize the DataCollector with a FileManager instance"""
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
        self.setup_data_directories()
        
        # Define period limitations for different intervals
        self.period_limits = {
            '15m': '60d',    # 60 days
            '30m': '60d',    # 60 days
            '60m': '730d',   # 2 years (Yahoo Finance's limit for hourly data)
            'daily': None,   # No limit
            'weekly': None,  # No limit
            'monthly': None  # No limit
        }
        
        # Define default periods for each interval
        self.default_periods = {
            '15m': '60d',
            '30m': '60d',
            '60m': '730d',   # 2 years for hourly data
            'daily': '5y',
            'weekly': '5y',
            'monthly': '5y'
        }
    
    def setup_data_directories(self):
        """Create necessary data directories"""
        try:
            intervals = ['daily', 'weekly', 'monthly', '15m', '30m', '60m']  # Changed '60m' to '1h'
            for interval in intervals:
                dir_path = os.path.join(self.file_manager.data_dir, interval)
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created/verified directory: {dir_path}")
        except Exception as e:
            self.logger.error(f"Error setting up data directories: {str(e)}")
            raise
    
    def collect_data(self, config: Dict, interval: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """Collect data for a symbol and interval"""
        try:
            yahoo_interval = self._get_yahoo_interval(interval)
            if not yahoo_interval:
                return None
            
            # Get the Yahoo Finance symbol from config
            yahoo_symbol = config.get('symbol', '')
            if not yahoo_symbol:
                return None
            
            try:
                # For intraday data (15m, 30m, 60m), use specific start and end dates
                if interval in ['15m', '30m', '60m']:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=60)  # Last 60 days for intraday data
                    data = yf.download(yahoo_symbol, start=start_date, end=end_date, interval=yahoo_interval)
                else:
                    # For daily, weekly, monthly data, use period parameter
                    data = yf.download(yahoo_symbol, period=period, interval=yahoo_interval)
                    
            except Exception as e:
                return None
            
            if data.empty:
                return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Check if the index column is named 'Datetime' instead of 'Date'
            if 'Datetime' in data.columns and 'Date' not in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return None
            
            # Ensure all required columns are present
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                return None
            
            data = data[required_columns]
            
            # Convert date column to datetime
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Sort by date
            data = data.sort_values('Date')
            
            return data
            
        except Exception as e:
            return None
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Save collected data to CSV file"""
        try:
            if data is None or data.empty:
                self.logger.warning(f"No data to save for {symbol} at {interval} interval")
                return False
            
            file_path = os.path.join(self.file_manager.data_dir, interval, f"{symbol}_{interval}.csv")
            
            # Add header information
            with open(file_path, 'w') as f:
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Interval: {interval}\n")
                f.write("\n")  # Empty line
                data.to_csv(f, index=False)
                
            self.logger.info(f"Saved data to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol} at {interval} interval: {str(e)}")
            return False
    
    def collect_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Collect data for all symbols and intervals"""
        try:
            # Load configuration
            config = self.file_manager.load_config()
            if not config:
                print("Failed to load configuration")
                return {}
            
            results = {}
            failed_data = []
            # Look for symbols under 'tickers' key instead of 'symbols'
            symbols = config.get('tickers', {})
            
            for symbol, symbol_config in symbols.items():
                symbol_results = {}
                
                # Get data collection settings from symbol's configuration
                data_collection = symbol_config.get('data_collection', {})
                intervals = data_collection.get('intervals', ['daily'])
                
                # Get the period from symbol's configuration
                period = data_collection.get('period', '5y')
                
                for interval in intervals:
                    # Get interval-specific period if defined
                    interval_period = symbol_config.get('intervals', {}).get(interval, {}).get('period', period)
                    
                    data = self.collect_data(symbol_config, interval, interval_period)
                    
                    if data is not None:
                        symbol_results[interval] = data
                        # Save the data
                        self.file_manager.save_data(symbol, interval, data)
                    else:
                        failed_data.append(f"{symbol} {interval}")
                
                if symbol_results:
                    results[symbol] = symbol_results
            
            if failed_data:
                print("Failed to load data for:", ", ".join(failed_data))
            
            return results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return {}
    
    def _get_yahoo_interval(self, interval: str) -> Optional[str]:
        """Convert interval name to Yahoo Finance interval code"""
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h'  # Convert 60m to 1h for Yahoo Finance API only
        }
        
        # Log the interval being processed
        if interval not in interval_map:
            self.logger.warning(f"Unsupported interval: {interval}. Supported intervals are: {list(interval_map.keys())}")
            return None
            
        return interval_map.get(interval)

if __name__ == "__main__":
    # Disable yfinance progress bar
    yf.pdr_override = lambda: None
    
    # Test the data collector
    file_manager = FileManager()
    collector = DataCollector(file_manager)
    
    try:
        # Load configuration
        config = file_manager.load_config()
        if not config:
            print("Failed to load configuration")
            exit(1)
            
        # Collect data
        collector.collect_all_data()
        
    except Exception as e:
        print(f"Error: {str(e)}") 