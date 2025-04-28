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
            '1m': '7d',     # 7 days for 1-minute data
            '2m': '60d',    # 60 days for 2-minute data
            '5m': '60d',    # 60 days for 5-minute data
            '15m': '60d',   # 60 days for 15-minute data
            '30m': '60d',   # 60 days for 30-minute data
            '60m': '730d',  # 2 years for 60-minute data
            '90m': '60d',   # 60 days for 90-minute data
            '1h': '730d',   # 2 years for 1-hour data
            '4h': '730d',   # 2 years for 4-hour data
        }
        
        # Remove default_periods since we've consolidated all periods in period_limits
        self.default_periods = {}
    
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
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Format the symbol for Yahoo Finance API.
        Adds '^' prefix for index symbols.
        
        Args:
            symbol (str): The original symbol
            
        Returns:
            str: The formatted symbol for Yahoo Finance
        """
        # List of known index symbols
        index_symbols = ['SPX', 'SPY', 'QQQ', 'DIA', 'VIX', 'NDX', 'DJI']
        
        if symbol in index_symbols:
            return f'^{symbol}'
        return symbol

    def collect_data(self, symbol, interval='1d', start=None, end=None):
        """
        Collect historical data for a given symbol and interval.
        
        Args:
            symbol (str): The stock symbol to collect data for
            interval (str): Data interval. Supported values depend on configuration
            start (str, optional): Start date in YYYY-MM-DD format
            end (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Historical data with UTC timestamps
            
        Raises:
            ValueError: If the interval is not supported or if data validation fails
        """
        try:
            # Format symbol for Yahoo Finance
            formatted_symbol = self._format_symbol(symbol)
            
            # Convert interval to Yahoo Finance format
            yf_interval = self._get_yahoo_interval(interval)
            if not yf_interval:
                raise ValueError(f"Unsupported interval: {interval}")
            
            # Get period only for intraday intervals
            period = None
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h']:
                if not start and not end:
                    period = self.get_period_for_interval(interval)
                
            # Download data
            self.logger.info(f"Collecting {interval} data for {symbol} (using {formatted_symbol})")
            df = yf.download(
                formatted_symbol,
                start=start,
                end=end,
                interval=yf_interval,  # Use converted interval
                period=period,
                progress=False
            )
            
            # Log the raw data structure
            self.logger.info(f"Raw data structure:")
            self.logger.info(f"Index type: {type(df.index)}")
            self.logger.info(f"Columns: {df.columns}")
            self.logger.info(f"First few rows:\n{df.head()}")
            self.logger.info(f"Data types:\n{df.dtypes}")
            
            # Basic validation
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            if len(df) < 2:
                raise ValueError(f"Insufficient data points for {symbol} (minimum 2 required)")
            
            # Handle timezone conversion
            if df.index.tz is None:
                # If naive, localize to UTC
                df.index = df.index.tz_localize('UTC')
            else:
                # If already timezone-aware, convert to UTC
                df.index = df.index.tz_convert('UTC')
            
            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='first')]
            
            # Log the processed data structure
            self.logger.info(f"Processed data structure:")
            self.logger.info(f"Index type: {type(df.index)}")
            self.logger.info(f"Columns: {df.columns}")
            self.logger.info(f"First few rows:\n{df.head()}")
            
            self.logger.info(f"Successfully collected {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Save collected data to CSV file"""
        try:
            if data is None or data.empty:
                self.logger.warning(f"No data to save for {symbol} at {interval} interval")
                return False
            
            # Log DataFrame info before saving
            self.logger.info(f"DataFrame info before saving:")
            self.logger.info(f"Index type: {type(data.index)}")
            self.logger.info(f"Columns: {data.columns}")
            self.logger.info(f"First few rows:\n{data.head()}")
            
            file_path = os.path.join(self.file_manager.data_dir, interval, f"{symbol}_{interval}.csv")
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            
            # Add header information
            with open(file_path, 'w') as f:
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Interval: {interval}\n")
                f.write("\n")  # Empty line
                
                # Save the data with timestamp column
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
                print(f"\nProcessing {symbol}...")  # Added for visibility
                symbol_results = {}
                
                # Get data collection settings from symbol's configuration
                data_collection = symbol_config.get('data_collection', {})
                intervals = data_collection.get('intervals', ['daily'])
                
                print(f"Intervals to collect: {intervals}")  # Added for visibility
                
                for interval in intervals:
                    print(f"\nCollecting {interval} data for {symbol}")  # Added for visibility
                    data = self.collect_data(symbol, interval)
                    
                    if data is not None:
                        symbol_results[interval] = data
                        # Save the data using our own save_data method
                        success = self.save_data(data, symbol, interval)
                        if success:
                            print(f"Successfully saved {interval} data for {symbol}")
                        else:
                            print(f"Failed to save {interval} data for {symbol}")
                    else:
                        failed_data.append(f"{symbol} {interval}")
                        print(f"Failed to collect {interval} data for {symbol}")
                
                if symbol_results:
                    results[symbol] = symbol_results
            
            if failed_data:
                print("\nFailed to load data for:", ", ".join(failed_data))
            
            return results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return {}
    
    def _get_yahoo_interval(self, interval: str) -> Optional[str]:
        """
        Convert interval name to Yahoo Finance interval code.
        
        Args:
            interval (str): The interval to convert (e.g., 'daily', 'weekly', 'monthly', '15m', etc.)
            
        Returns:
            Optional[str]: The corresponding Yahoo Finance interval code, or None if unsupported
            
        Note:
            Valid Yahoo Finance intervals are:
            - Intraday: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
            - Daily and above: 1d, 5d, 1wk, 1mo, 3mo
        """
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo',
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '90m': '90m',
            '1h': '1h',
            '4h': '4h'
        }
        
        # Log the interval being processed
        if interval not in interval_map:
            self.logger.warning(f"Unsupported interval: {interval}. Supported intervals are: {list(interval_map.keys())}")
            return None
            
        return interval_map.get(interval)

    def get_period_for_interval(self, interval):
        """
        Get the appropriate period for the given interval based on configuration.
        Only used for intraday intervals.
        
        Args:
            interval (str): The data interval (e.g., '1m', '15m', etc.)
            
        Returns:
            str: The period to use for data collection
            
        Raises:
            ValueError: If the interval is not supported
        """
        # Check if interval is supported
        if interval in self.period_limits:
            return self.period_limits[interval]
        else:
            supported_intervals = list(self.period_limits.keys())
            raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {supported_intervals}")

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