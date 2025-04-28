import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator
import json
from datetime import datetime
import os

# === CONFIG ===
CSV_PATH = r"C:\Users\Swathi Gogula\Desktop\myapps\stocks\stock-dashboards\data\daily\SPX_daily.csv"
LOG_ENABLED = True
NUM_BINS = 5  # Number of bins to create
OUTPUT_DIR = r"C:\Users\Swathi Gogula\Desktop\myapps\stocks\stock-dashboards\data\signals"

def create_dynamic_bins(df, signal_type):
    """Create dynamic bins based on the data range for buy or sell signals"""
    if signal_type == 'buy':
        # For buy signals, we look at how far below the lower BB
        values = df['%_from_LowerBB'].dropna()
    else:  # sell
        # For sell signals, we look at how far above the upper BB
        values = df['%_from_UpperBB'].dropna()
    
    if len(values) == 0:
        return [], []
    
    # Get min and max values
    min_val = values.min()
    max_val = values.max()
    
    # Add a small buffer to ensure all values are included
    buffer = (max_val - min_val) * 0.01
    min_val -= buffer
    max_val += buffer
    
    # Create bins with equal width
    bin_width = (max_val - min_val) / NUM_BINS
    bins = [min_val + i * bin_width for i in range(NUM_BINS + 1)]
    
    # Create labels for the bins
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}%" for i in range(NUM_BINS)]
    
    return bins, labels

def load_data():
    """Load and prepare the data"""
    # Read CSV with proper header handling
    df = pd.read_csv(CSV_PATH, skiprows=3)  # Skip 3 rows to get to the actual data
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Match the actual column order
    
    # Convert date and numeric columns
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean numeric columns by removing commas and converting to float
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Sort by date and set as index
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    
    # Drop rows with NaN values in price columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    return df

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Bollinger Bands
    indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_upper'] = indicator_bb.bollinger_hband()
    df['bb_middle'] = indicator_bb.bollinger_mavg()
    df['bb_lower'] = indicator_bb.bollinger_lband()
    
    # Moving Averages
    df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Only calculate SMA200 if we have enough data points
    if len(df) >= 200:
        df['SMA200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    else:
        print(f"Warning: Not enough data points for SMA200 calculation. Required: 200, Available: {len(df)}")
        df['SMA200'] = np.nan
    
    return df

def calculate_forward_returns(df, signal_type):
    """Calculate forward returns based on data frequency"""
    # Determine data frequency
    if 'weekly' in CSV_PATH.lower():
        # For weekly data, calculate returns for next 4 weeks
        periods = [1, 2, 3, 4]
    elif 'monthly' in CSV_PATH.lower():
        # For monthly data, calculate returns for next 4 months
        periods = [1, 2, 3, 4]
    else:
        # For shorter timeframes (hourly/daily), calculate returns for 5, 10, 15 bars
        periods = [5, 10, 15]
    
    # Calculate forward returns for each period
    for period in periods:
        df[f'Return_{period}'] = df['Close'].shift(-period) / df['Close'] * 100 - 100
    
    return df, periods

def analyze_signals(df):
    """Analyze trading signals based on Bollinger Bands and store in JSON format"""
    signals = []
    bin_summaries = {
        'buy': {},
        'sell': {}
    }
    
    # Calculate forward returns
    df, periods = calculate_forward_returns(df, 'all')
    
    # Buy signals
    buy_signals = df[df['Buy_Signal']].copy()
    if not buy_signals.empty:
        buy_signals['%_from_LowerBB'] = ((buy_signals['Close'] - buy_signals['bb_lower']) / buy_signals['bb_lower']) * 100
        # Drop NaN values before creating bins
        buy_signals = buy_signals.dropna(subset=['%_from_LowerBB'])
        if not buy_signals.empty:
            bins, labels = create_dynamic_bins(buy_signals, 'buy')
            if bins and labels:  # Only proceed if we have valid bins
                buy_signals['BB_Distance_Bin'] = pd.cut(buy_signals['%_from_LowerBB'], bins=bins, labels=labels, include_lowest=True)
                # Drop any rows that didn't get assigned to a bin
                buy_signals = buy_signals.dropna(subset=['BB_Distance_Bin'])
                
                # Initialize bin summaries for buy signals
                for label in labels:
                    bin_summaries['buy'][label] = {
                        'count': 0,
                        'avg_returns': {f'return_{period}': 0 for period in periods},
                        'max_returns': {f'return_{period}': 0 for period in periods},
                        'max_losses': {f'return_{period}': 0 for period in periods},
                        'signals': []
                    }
                
                for _, row in buy_signals.iterrows():
                    signal = {
                        'date': row.name.strftime('%Y-%m-%d'),
                        'type': 'buy',
                        'price': {
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close'])
                        },
                        'indicators': {
                            'bb_lower': float(row['bb_lower']),
                            'bb_middle': float(row['bb_middle']),
                            'bb_upper': float(row['bb_upper']),
                            'sma20': float(row['SMA20']),
                            'sma50': float(row['SMA50']),
                            'sma200': float(row['SMA200']) if not pd.isna(row['SMA200']) else None
                        },
                        'distance_bin': str(row['BB_Distance_Bin']),
                        'forward_returns': {
                            f'return_{period}': float(row[f'Return_{period}']) if not pd.isna(row[f'Return_{period}']) else None
                            for period in periods
                        }
                    }
                    signals.append(signal)
                    
                    # Update bin summary
                    bin_label = str(row['BB_Distance_Bin'])
                    bin_summaries['buy'][bin_label]['count'] += 1
                    bin_summaries['buy'][bin_label]['signals'].append(signal)
    
    # Sell signals
    sell_signals = df[df['Sell_Signal']].copy()
    if not sell_signals.empty:
        sell_signals['%_from_UpperBB'] = ((sell_signals['Close'] - sell_signals['bb_upper']) / sell_signals['bb_upper']) * 100
        # Drop NaN values before creating bins
        sell_signals = sell_signals.dropna(subset=['%_from_UpperBB'])
        if not sell_signals.empty:
            bins, labels = create_dynamic_bins(sell_signals, 'sell')
            if bins and labels:  # Only proceed if we have valid bins
                sell_signals['BB_Distance_Bin'] = pd.cut(sell_signals['%_from_UpperBB'], bins=bins, labels=labels, include_lowest=True)
                # Drop any rows that didn't get assigned to a bin
                sell_signals = sell_signals.dropna(subset=['BB_Distance_Bin'])
                
                # Initialize bin summaries for sell signals
                for label in labels:
                    bin_summaries['sell'][label] = {
                        'count': 0,
                        'avg_returns': {f'return_{period}': 0 for period in periods},
                        'max_returns': {f'return_{period}': 0 for period in periods},
                        'max_losses': {f'return_{period}': 0 for period in periods},
                        'signals': []
                    }
                
                for _, row in sell_signals.iterrows():
                    signal = {
                        'date': row.name.strftime('%Y-%m-%d'),
                        'type': 'sell',
                        'price': {
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close'])
                        },
                        'indicators': {
                            'bb_lower': float(row['bb_lower']),
                            'bb_middle': float(row['bb_middle']),
                            'bb_upper': float(row['bb_upper']),
                            'sma20': float(row['SMA20']),
                            'sma50': float(row['SMA50']),
                            'sma200': float(row['SMA200']) if not pd.isna(row['SMA200']) else None
                        },
                        'distance_bin': str(row['BB_Distance_Bin']),
                        'forward_returns': {
                            f'return_{period}': float(row[f'Return_{period}']) if not pd.isna(row[f'Return_{period}']) else None
                            for period in periods
                        }
                    }
                    signals.append(signal)
                    
                    # Update bin summary
                    bin_label = str(row['BB_Distance_Bin'])
                    bin_summaries['sell'][bin_label]['count'] += 1
                    bin_summaries['sell'][bin_label]['signals'].append(signal)
    
    # Calculate average returns, max returns, and max losses for each bin
    for signal_type in ['buy', 'sell']:
        for bin_label, bin_data in bin_summaries[signal_type].items():
            if bin_data['count'] > 0:
                for period in periods:
                    returns = [s['forward_returns'][f'return_{period}'] 
                             for s in bin_data['signals'] 
                             if s['forward_returns'][f'return_{period}'] is not None]
                    if returns:
                        bin_data['avg_returns'][f'return_{period}'] = sum(returns) / len(returns)
                        bin_data['max_returns'][f'return_{period}'] = max(returns)
                        bin_data['max_losses'][f'return_{period}'] = min(returns)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed signals
    signals_file = os.path.join(OUTPUT_DIR, f'trading_signals.json')
    with open(signals_file, 'w') as f:
        json.dump(signals, f, indent=4)
    
    # Save bin summaries
    summary_file = os.path.join(OUTPUT_DIR, f'bin_summaries.json')
    with open(summary_file, 'w') as f:
        json.dump(bin_summaries, f, indent=4)
    
    print(f"\n✅ Signals saved to: {signals_file}")
    print(f"✅ Bin summaries saved to: {summary_file}")
    
    return signals, bin_summaries

def generate_html_report(bin_summaries, output_dir):
    """Generate HTML report from bin summaries"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bollinger Bands Signal Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .signal-type { margin-bottom: 30px; }
            .bin { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .bin-header { background-color: #f5f5f5; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }
            .returns-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            .returns-table th, .returns-table td { padding: 8px; text-align: center; border: 1px solid #ddd; }
            .returns-table th { background-color: #f5f5f5; }
            .positive { color: green; }
            .negative { color: red; }
            .signal-details { margin-top: 20px; }
            .signal-row { border: 1px solid #eee; padding: 10px; margin: 5px 0; }
            .signal-date { font-weight: bold; }
            .signal-price { color: #666; }
            .signal-returns { margin-top: 5px; }
            .max-return { color: #2e7d32; }
            .max-loss { color: #c62828; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bollinger Bands Signal Analysis</h1>
    """
    
    # Add buy signals section
    html_content += """
        <div class="signal-type">
            <h2>Buy Signals</h2>
    """
    for bin_label, bin_data in bin_summaries['buy'].items():
        if bin_data['count'] > 0:
            html_content += f"""
            <div class="bin">
                <div class="bin-header">
                    <h3>Distance Bin: {bin_label}</h3>
                    <p>Total Signals: {bin_data['count']}</p>
                </div>
                <table class="returns-table">
                    <tr>
                        <th>Time Period</th>
                        <th>Average Return</th>
                        <th>Max Return</th>
                        <th>Max Loss</th>
                    </tr>
            """
            for period in bin_data['avg_returns'].keys():
                avg_return = bin_data['avg_returns'][period]
                max_return = bin_data['max_returns'][period]
                max_loss = bin_data['max_losses'][period]
                avg_class = "positive" if avg_return > 0 else "negative"
                html_content += f"""
                    <tr>
                        <td>{period}</td>
                        <td class="{avg_class}">{avg_return:.2f}%</td>
                        <td class="max-return">{max_return:.2f}%</td>
                        <td class="max-loss">{max_loss:.2f}%</td>
                    </tr>
                """
            html_content += """
                </table>
                <div class="signal-details">
                    <h4>Individual Signals:</h4>
            """
            for signal in bin_data['signals']:
                html_content += f"""
                    <div class="signal-row">
                        <div class="signal-date">Date: {signal['date']}</div>
                        <div class="signal-price">
                            Price: {signal['price']['close']:.2f} | 
                            BB Lower: {signal['indicators']['bb_lower']:.2f} | 
                            BB Middle: {signal['indicators']['bb_middle']:.2f} | 
                            BB Upper: {signal['indicators']['bb_upper']:.2f}
                        </div>
                        <div class="signal-returns">
                            Returns: {', '.join([f"{k}: {v:.2f}%" for k, v in signal['forward_returns'].items() if v is not None])}
                        </div>
                    </div>
                """
            html_content += """
                </div>
            </div>
            """
    
    # Add sell signals section
    html_content += """
        </div>
        <div class="signal-type">
            <h2>Sell Signals</h2>
    """
    for bin_label, bin_data in bin_summaries['sell'].items():
        if bin_data['count'] > 0:
            html_content += f"""
            <div class="bin">
                <div class="bin-header">
                    <h3>Distance Bin: {bin_label}</h3>
                    <p>Total Signals: {bin_data['count']}</p>
                </div>
                <table class="returns-table">
                    <tr>
                        <th>Time Period</th>
                        <th>Average Return</th>
                        <th>Max Return</th>
                        <th>Max Loss</th>
                    </tr>
            """
            for period in bin_data['avg_returns'].keys():
                avg_return = bin_data['avg_returns'][period]
                max_return = bin_data['max_returns'][period]
                max_loss = bin_data['max_losses'][period]
                avg_class = "positive" if avg_return > 0 else "negative"
                html_content += f"""
                    <tr>
                        <td>{period}</td>
                        <td class="{avg_class}">{avg_return:.2f}%</td>
                        <td class="max-return">{max_return:.2f}%</td>
                        <td class="max-loss">{max_loss:.2f}%</td>
                    </tr>
                """
            html_content += """
                </table>
                <div class="signal-details">
                    <h4>Individual Signals:</h4>
            """
            for signal in bin_data['signals']:
                html_content += f"""
                    <div class="signal-row">
                        <div class="signal-date">Date: {signal['date']}</div>
                        <div class="signal-price">
                            Price: {signal['price']['close']:.2f} | 
                            BB Lower: {signal['indicators']['bb_lower']:.2f} | 
                            BB Middle: {signal['indicators']['bb_middle']:.2f} | 
                            BB Upper: {signal['indicators']['bb_upper']:.2f}
                        </div>
                        <div class="signal-returns">
                            Returns: {', '.join([f"{k}: {v:.2f}%" for k, v in signal['forward_returns'].items() if v is not None])}
                        </div>
                    </div>
                """
            html_content += """
                </div>
            </div>
            """
    
    html_content += """
        </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_file = os.path.join(output_dir, 'signal_analysis_report.html')
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {html_file}")
    return html_file

def main():
    # Load data
    df = load_data()
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Define buy and sell signals
    df['Buy_Signal'] = (df['Close'] < df['bb_lower']) & (df['Close'].shift(1) >= df['bb_lower'].shift(1))
    df['Sell_Signal'] = (df['Close'] > df['bb_upper']) & (df['Close'].shift(1) <= df['bb_upper'].shift(1))
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Filter out consecutive signals of the same type
    # Create a column to track the last signal type
    df['Last_Signal'] = None
    
    # Initialize the first signal
    if df['Buy_Signal'].iloc[0]:
        df.loc[df.index[0], 'Last_Signal'] = 'buy'
    elif df['Sell_Signal'].iloc[0]:
        df.loc[df.index[0], 'Last_Signal'] = 'sell'
    
    # Iterate through the dataframe to filter consecutive signals
    for i in range(1, len(df)):
        current_idx = df.index[i]
        prev_idx = df.index[i-1]
        
        if df['Buy_Signal'].iloc[i]:
            if df.loc[prev_idx, 'Last_Signal'] != 'buy':
                df.loc[current_idx, 'Last_Signal'] = 'buy'
                df.loc[current_idx, 'Buy_Signal'] = True
            else:
                df.loc[current_idx, 'Buy_Signal'] = False
        elif df['Sell_Signal'].iloc[i]:
            if df.loc[prev_idx, 'Last_Signal'] != 'sell':
                df.loc[current_idx, 'Last_Signal'] = 'sell'
                df.loc[current_idx, 'Sell_Signal'] = True
            else:
                df.loc[current_idx, 'Sell_Signal'] = False
        else:
            df.loc[current_idx, 'Last_Signal'] = df.loc[prev_idx, 'Last_Signal']
    
    # Analyze signals and store in JSON
    signals, bin_summaries = analyze_signals(df)
    
    # Generate HTML report
    html_file = generate_html_report(bin_summaries, OUTPUT_DIR)
    
    # Print summary
    print("\n📊 Trading Signals Summary:")
    print(f"Total Buy Signals: {len([s for s in signals if s['type'] == 'buy'])}")
    print(f"Total Sell Signals: {len([s for s in signals if s['type'] == 'sell'])}")
    
    # Print bin summaries
    print("\n🔍 Bin Summaries:")
    for signal_type in ['buy', 'sell']:
        print(f"\n{signal_type.upper()} Signals:")
        for bin_label, bin_data in bin_summaries[signal_type].items():
            if bin_data['count'] > 0:
                print(f"\n  {bin_label}:")
                print(f"    Count: {bin_data['count']}")
                print("    Average Returns:")
                for period, avg_return in bin_data['avg_returns'].items():
                    print(f"      {period}: {avg_return:.2f}%")
    
    print(f"\n✅ HTML report generated: {html_file}")

if __name__ == "__main__":
    main()
