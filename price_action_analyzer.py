import os
import pandas as pd
from datetime import datetime
import logging
import json
from file_manager import FileManager
from ticker_manager import TickerManager

class PriceActionAnalyzer:
    def __init__(self, file_manager: FileManager, ticker_manager: TickerManager):
        """Initialize the PriceActionAnalyzer with FileManager and TickerManager instances"""
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
                logging.FileHandler('price_action_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
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
    
    def analyze_ticker(self, symbol: str, interval: str = '1d') -> Dict:
        """Analyze price action for a specific ticker"""
        try:
            # Load existing data
            df = self.load_data(symbol, interval)
            
            if df is None:
                error_msg = f"No data available for {symbol} at {interval} interval"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Analyze price action
            bias_reason = self.analyze_price_action(df)
            
            # Create result dictionary
            result = {
                'bias_reason': bias_reason,
                'current_price': round(df['Close'].iloc[-1], 2),
                'ticker_info': {
                    'symbol': symbol,
                    'interval': interval,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'data_start': df.index[0].strftime('%Y-%m-%d'),
                    'data_end': df.index[-1].strftime('%Y-%m-%d')
                }
            }
            
            # Save results
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"price_action_analysis_{symbol}_{interval}_{date_str}.json"
            file_path = os.path.join(self.analysis_output_dir, filename)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            self.logger.info(f"Saved {symbol} price action analysis to {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise
    
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
    
    def analyze_all_tickers(self) -> Dict[str, Dict]:
        """Analyze price action for all tickers using their configured timeframes"""
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
                        results[symbol][timeframe] = self.analyze_ticker(symbol, interval)
                    else:
                        self.logger.warning(f"Invalid timeframe {timeframe} for {symbol}")
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Save summary of all tickers' analysis
        date_str = datetime.now().strftime("%Y-%m-%d")
        summary_filename = f"price_action_analysis_summary_{date_str}.json"
        summary_path = os.path.join(self.analysis_output_dir, summary_filename)
        
        summary_data = {
            'analysis_date': date_str,
            'tickers': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        self.logger.info(f"Saved price action analysis summary to {summary_path}")
        return results
    
    def _get_yahoo_interval(self, timeframe: str) -> Optional[str]:
        """Map timeframe name to Yahoo Finance interval code"""
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        return interval_map.get(timeframe.lower())

if __name__ == "__main__":
    from file_manager import FileManager
    from ticker_manager import TickerManager
    
    # Initialize managers
    file_manager = FileManager()
    ticker_manager = TickerManager(file_manager)
    analyzer = PriceActionAnalyzer(file_manager, ticker_manager)
    
    try:
        # Run analysis for all tickers
        print("\nAnalyzing price action for all tickers...")
        results = analyzer.analyze_all_tickers()
        
        # Print summary for each ticker
        for symbol, result in results.items():
            print(f"\n=== Price Action Analysis for {symbol} ===")
            for timeframe, data in result.items():
                print(f"\n{timeframe.capitalize()} Timeframe:")
                print(f"Bias Reason: {data['bias_reason']}")
                print(f"Current Price: {data['current_price']:.2f}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}") 