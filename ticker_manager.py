import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import logging

class TickerManager:
    def __init__(self, file_manager):
        """Initialize TickerManager with FileManager instance"""
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Load ticker configuration
        self.ticker_config = self.load_ticker_config()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ticker_manager.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_ticker_config(self) -> Dict:
        """Load ticker configuration from YAML file"""
        try:
            config_path = os.path.join(os.path.dirname(self.file_manager.data_dir), 'config', 'tickers.yaml')
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading ticker configuration: {str(e)}")
            return {}
    
    def get_ticker_config(self) -> Dict:
        """Get the ticker configuration"""
        return self.ticker_config
    
    def get_ticker_data(self, symbol: str, interval: str = '1d', period: str = '20y') -> pd.DataFrame:
        """Download ticker data using yfinance"""
        try:
            # Get symbol from config if it exists
            ticker_info = self.ticker_config['tickers'].get(symbol, {})
            yf_symbol = ticker_info.get('symbol', symbol)
            
            # Download data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Save to appropriate directory based on interval
            interval_dir = {
                '1d': 'daily',
                '1wk': 'weekly',
                '1mo': 'monthly'
            }.get(interval, 'daily')
            
            save_dir = os.path.join(self.file_manager.data_dir, interval_dir)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save file with date range in name
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            filename = f"{symbol}_{start_date}_to_{end_date}_{interval}.csv"
            save_path = os.path.join(save_dir, filename)
            
            # Reset index to make Date a column and save
            df.reset_index().to_csv(save_path, index=False)
            self.logger.info(f"Saved {symbol} data to {save_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading {symbol} data: {str(e)}")
            raise
    
    def get_all_tickers_data(self, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Download data for all tickers in config"""
        results = {}
        period = self.ticker_config.get('data_collection', {}).get('period', '20y')
        
        for symbol in self.ticker_config['tickers'].keys():
            try:
                df = self.get_ticker_data(symbol, interval, period)
                results[symbol] = df
            except Exception as e:
                self.logger.error(f"Error getting data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers from configuration"""
        return list(self.ticker_config.get('tickers', {}).keys())
    
    def get_ticker_info(self, symbol: str) -> Dict:
        """Get information for a specific ticker"""
        return self.ticker_config.get('tickers', {}).get(symbol, {}) 