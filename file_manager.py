import os
import glob
from datetime import datetime, timedelta
import json
import shutil
import logging
from typing import List, Optional, Union, Dict
import yaml
import pandas as pd

class FileManager:
    def __init__(self):
        """Initialize FileManager with project directories"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.analysis_dir = os.path.join(self.base_dir, 'analysis_output')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        for directory in [self.data_dir, self.analysis_dir, self.reports_dir, self.config_dir]:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created/verified directory: {directory}")
    
    def get_data_file_path(self, filename: str) -> str:
        """Get full path for a data file"""
        return os.path.join(self.data_dir, filename)
    
    def get_analysis_file_path(self, filename: str) -> str:
        """Get full path for an analysis file"""
        return os.path.join(self.analysis_dir, filename)
    
    def get_config_file_path(self, filename: str) -> str:
        """Get full path for a config file"""
        return os.path.join(self.config_dir, filename)
    
    def get_latest_analysis_file(self) -> Optional[str]:
        """Get the path to the latest analysis file"""
        try:
            analysis_files = [f for f in os.listdir(self.analysis_dir) if f.endswith('.json')]
            if not analysis_files:
                return None
            
            latest_file = max(analysis_files, key=lambda x: os.path.getctime(os.path.join(self.analysis_dir, x)))
            return os.path.join(self.analysis_dir, latest_file)
        except Exception as e:
            self.logger.error(f"Error getting latest analysis file: {str(e)}")
            return None
    
    def cleanup_old_files(self, cutoff_time: Union[datetime, int]) -> List[str]:
        """Remove files older than the cutoff time and return list of removed files
        
        Args:
            cutoff_time: Either a datetime object or number of hours to keep
        """
        removed_files = []
        try:
            # Convert hours to datetime if needed
            if isinstance(cutoff_time, int):
                cutoff_time = datetime.now() - timedelta(hours=cutoff_time)
            
            for directory in [self.data_dir, self.analysis_dir, self.reports_dir]:
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            removed_files.append(file_path)
                            self.logger.info(f"Removed old file: {file_path}")
            return removed_files
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {str(e)}")
            return removed_files
    
    def save_file(self, content: str, filename: str, directory: str) -> bool:
        """Save content to a file in the specified directory"""
        try:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Saved file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving file {filename}: {str(e)}")
            return False
    
    def load_file(self, filename: str, directory: str) -> Optional[str]:
        """Load content from a file in the specified directory"""
        try:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            self.logger.error(f"Error loading file {filename}: {str(e)}")
            return None
    
    def get_latest_file(self, prefix: str) -> Optional[str]:
        """Get the latest file with given prefix"""
        pattern = os.path.join(self.analysis_dir, f"{prefix}*.json")
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)  # Returns full path
    
    def get_all_files(self, prefix: str = None) -> List[str]:
        """Get all files in directory, optionally filtered by prefix"""
        pattern = os.path.join(self.analysis_dir, f"{prefix}*.json" if prefix else "*.json")
        return glob.glob(pattern)
    
    def save_json(self, data: dict, filename: str) -> str:
        """Save data to JSON file with date-only filename, overriding same-day files"""
        # Get today's date in YYYYMMDD format
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{filename}_{date_str}.json"
        file_path = os.path.join(self.analysis_dir, filename)
        
        # Remove existing file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Save new file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return file_path
    
    def load_json(self, file_path: str) -> dict:
        """Load data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def setup_data_directories(self):
        """Create necessary data directories"""
        try:
            intervals = ['daily', 'weekly', 'monthly', '15m', '30m', '60m']  # Using 60m instead of 1h
            for interval in intervals:
                dir_path = os.path.join(self.data_dir, interval)
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created/verified directory: {dir_path}")
        except Exception as e:
            self.logger.error(f"Error setting up data directories: {str(e)}")
            raise

    def load_config(self) -> Optional[Dict]:
        """Load configuration from tickers.yaml"""
        try:
            config_path = os.path.join(self.config_dir, 'tickers.yaml')
            if not os.path.exists(config_path):
                self.logger.error(f"Configuration file not found at {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return None

    def save_data(self, symbol: str, interval: str, data: pd.DataFrame) -> bool:
        """Save collected data to CSV file"""
        try:
            if data is None or data.empty:
                return False
            
            file_path = os.path.join(self.data_dir, interval, f"{symbol}_{interval}.csv")
            
            # Add header information
            with open(file_path, 'w') as f:
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Interval: {interval}\n")
                f.write("\n")  # Empty line
                data.to_csv(f, index=False)
                
            return True
            
        except Exception as e:
            return False

if __name__ == "__main__":
    manager = FileManager()
    manager.cleanup_old_files(24)  # Keep files from last 24 hours 