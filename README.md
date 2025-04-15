# Market Health Analyzer

A comprehensive tool for analyzing market health using multiple indicators and data sources.

## Project Structure
```
stock-dashboards/
├── config/                # Configuration files
│   └── tickers.yaml      # Ticker configuration
├── data/                  # Data storage
│   ├── daily/            # Daily data
│   ├── weekly/           # Weekly data
│   └── monthly/          # Monthly data
├── requirements.txt      # Project dependencies
├── data_collector.py     # Data collection script
└── README.md            # Project documentation
```

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Run All Components at Once
Use the orchestrator script to run the entire pipeline:
```bash
# Run data collection and analysis
python run_analysis.py --collect-data --analyze --swing --summary

# Launch the dashboard in your browser
streamlit run dashboard.py
```

You can also run specific components:
```bash
# Just collect data
python run_analysis.py --collect-data

# Collect data and analyze
python run_analysis.py --collect-data --analyze

# Run everything except dashboard
python run_analysis.py --collect-data --analyze --swing --summary
```

### Option 2: Run Components Individually
Run the following commands in sequence:

1. Collect market data:
```bash
python data_collector.py
```

2. Analyze market data:
```bash
python market_analyzer.py
```

3. Perform technical analysis:
```bash
python swing_analyzer.py
```

4. Generate summary report:
```bash
python summary_generator.py
```

5. Generate detailed report:
```bash
python report_generator.py
```

6. Launch the dashboard in your browser:
```bash
streamlit run dashboard.py
```

## Configuration

The `config/tickers.yaml` file contains all ticker information and data collection settings. You can modify this file to:
- Add or remove tickers
- Change data collection periods
- Modify intervals
- Add ticker descriptions

Example configuration:
```yaml
tickers:
  SPX:
    symbol: ^GSPC
    name: S&P 500 Index
    description: "Broad market index representing 500 large companies"
  
data_collection:
  period: 20y
  intervals:
    - daily: 1d
    - weekly: 1wk
    - monthly: 1mo
```

## Data Collection
The system currently collects data for:
- S&P 500 Index (SPX)
- Volatility Index (VIX)
- Homebuilders ETF (XHB)

Data is collected for:
- Daily intervals
- Weekly intervals
- Monthly intervals

All data spans 20 years of historical data.

## Next Steps
- Implement market health indicators
- Create visualization dashboard
- Add more data sources
- Implement real-time updates 