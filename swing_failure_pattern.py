import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

class SwingFailurePattern:
    def __init__(self, 
                 swing_length: int = 5,
                 enable_bullish: bool = True,
                 enable_bearish: bool = True):
        """
        Initialize the Swing Failure Pattern detector
        
        Args:
            swing_length: Number of bars to look back for swing detection
            enable_bullish: Enable bullish SFP detection
            enable_bearish: Enable bearish SFP detection
        """
        self.swing_length = swing_length
        self.enable_bullish = enable_bullish
        self.enable_bearish = enable_bearish
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('swing_failure_pattern.log'),
                logging.StreamHandler()
            ]
        )
    
    def find_pivot_highs(self, df: pd.DataFrame) -> pd.Series:
        """Find pivot highs in the price data"""
        highs = df['High']
        pivot_highs = pd.Series(False, index=df.index)
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            window = highs.iloc[i-self.swing_length:i+self.swing_length+1]
            if highs.iloc[i] == window.max():
                pivot_highs.iloc[i] = True
                
        return pivot_highs
    
    def find_pivot_lows(self, df: pd.DataFrame) -> pd.Series:
        """Find pivot lows in the price data"""
        lows = df['Low']
        pivot_lows = pd.Series(False, index=df.index)
        
        for i in range(self.swing_length, len(df) - self.swing_length):
            window = lows.iloc[i-self.swing_length:i+self.swing_length+1]
            if lows.iloc[i] == window.min():
                pivot_lows.iloc[i] = True
                
        return pivot_lows
    
    def detect_bearish_sfp(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bearish Swing Failure Patterns"""
        if not self.enable_bearish:
            return []
            
        patterns = []
        pivot_highs = self.find_pivot_highs(df)
        
        for i in range(len(df)):
            if not pivot_highs.iloc[i]:
                continue
                
            swing_high = df['High'].iloc[i]
            swing_bar = i
            
            # Look for potential SFP
            for j in range(i+1, len(df)):
                if (df['High'].iloc[j] > swing_high and 
                    df['Open'].iloc[j] < swing_high and 
                    df['Close'].iloc[j] < swing_high):
                    
                    # Find opposite point
                    oppos_low = swing_high
                    oppos_bar = j
                    
                    for k in range(1, j - swing_bar):
                        if df['Low'].iloc[j-k] < oppos_low:
                            oppos_low = df['Low'].iloc[j-k]
                            oppos_bar = j - k
                    
                    # Check for confirmation
                    if df['Close'].iloc[j] < oppos_low:
                        patterns.append({
                            'type': 'bearish',
                            'swing_high': swing_high,
                            'swing_bar': swing_bar,
                            'oppos_low': oppos_low,
                            'oppos_bar': oppos_bar,
                            'confirmation_bar': j,
                            'confirmed': True
                        })
                    else:
                        patterns.append({
                            'type': 'bearish',
                            'swing_high': swing_high,
                            'swing_bar': swing_bar,
                            'oppos_low': oppos_low,
                            'oppos_bar': oppos_bar,
                            'confirmation_bar': j,
                            'confirmed': False
                        })
                        
        return patterns
    
    def detect_bullish_sfp(self, df: pd.DataFrame) -> List[Dict]:
        """Detect bullish Swing Failure Patterns"""
        if not self.enable_bullish:
            return []
            
        patterns = []
        pivot_lows = self.find_pivot_lows(df)
        
        for i in range(len(df)):
            if not pivot_lows.iloc[i]:
                continue
                
            swing_low = df['Low'].iloc[i]
            swing_bar = i
            
            # Look for potential SFP
            for j in range(i+1, len(df)):
                if (df['Low'].iloc[j] < swing_low and 
                    df['Open'].iloc[j] > swing_low and 
                    df['Close'].iloc[j] > swing_low):
                    
                    # Find opposite point
                    oppos_high = swing_low
                    oppos_bar = j
                    
                    for k in range(1, j - swing_bar):
                        if df['High'].iloc[j-k] > oppos_high:
                            oppos_high = df['High'].iloc[j-k]
                            oppos_bar = j - k
                    
                    # Check for confirmation
                    if df['Close'].iloc[j] > oppos_high:
                        patterns.append({
                            'type': 'bullish',
                            'swing_low': swing_low,
                            'swing_bar': swing_bar,
                            'oppos_high': oppos_high,
                            'oppos_bar': oppos_bar,
                            'confirmation_bar': j,
                            'confirmed': True
                        })
                    else:
                        patterns.append({
                            'type': 'bullish',
                            'swing_low': swing_low,
                            'swing_bar': swing_bar,
                            'oppos_high': oppos_high,
                            'oppos_bar': oppos_bar,
                            'confirmation_bar': j,
                            'confirmed': False
                        })
                        
        return patterns
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all Swing Failure Patterns in the data"""
        bearish_patterns = self.detect_bearish_sfp(df)
        bullish_patterns = self.detect_bullish_sfp(df)
        
        return bearish_patterns + bullish_patterns

if __name__ == "__main__":
    # Example usage
    sfp = SwingFailurePattern(
        swing_length=5,
        enable_bullish=True,
        enable_bearish=True
    )
    
    # Load your price data into a DataFrame with columns: 'Open', 'High', 'Low', 'Close'
    # df = pd.read_csv('your_data.csv')
    
    # Detect patterns
    # patterns = sfp.detect_patterns(df)
    # print(f"Found {len(patterns)} Swing Failure Patterns") 