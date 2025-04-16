import os
import json
from datetime import datetime
import logging
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator:
    def __init__(self, analysis_output_dir: str):
        """Initialize the ReportGenerator with the analysis output directory"""
        self.analysis_output_dir = analysis_output_dir
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Create reports directory if it doesn't exist
        self.reports_dir = os.path.join(os.path.dirname(analysis_output_dir), 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('report_generation.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_summary_data(self) -> Dict:
        """Load the market analysis summary data"""
        summary_path = os.path.join(self.analysis_output_dir, 'market_analysis_summary.json')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def generate_html_report(self, summary_data: Dict) -> str:
        """Generate HTML report from summary data in a table format for all symbols"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                h2 {{
                    margin-top: 40px;
                    margin-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .signal-buy {{ background-color: #d4edda; color: #155724; font-weight: bold; }}
                .signal-sell {{ background-color: #f8d7da; color: #721c24; font-weight: bold; }}
                .signal-neutral {{ background-color: #fff3cd; color: #856404; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Market Analysis Report</h1>
                    <p>Generated on: {summary_data['analysis_date']}</p>
                </div>
        """
        
        for symbol, tf_data in summary_data['tickers'].items():
            html_content += f"<h2>{symbol} Summary</h2>"
            html_content += """
            <table>
                <tr>
                    <th>Timeframe</th>
                    <th>Price</th>
                    <th>RSI</th>
                    <th>Bias Reason</th>
                    <th>SMA (20)</th>
                    <th>SMA (50)</th>
                    <th>SMA (200)</th>
                    <th>BB Upper</th>
                    <th>BB Lower</th>
                    <th>Pivot High</th>
                    <th>Pivot Low</th>
                </tr>
            """
            
            # Get all available timeframes for this symbol
            timeframes = list(tf_data.keys())
            
            for timeframe in timeframes:
                data = tf_data[timeframe]
                def pct(val):
                    return f"<span class='{'positive' if val > 0 else 'negative'}'>({'+' if val > 0 else ''}{val:.2f}%)</span>"
                html_content += f"""
                <tr>
                    <td>{timeframe.capitalize()}</td>
                    <td>{data['current_price']:.2f}</td>
                    <td>{data['rsi']:.2f}</td>
                    <td>{data['bias_reason']}</td>
                    <td>{data['sma_20']:.2f} {pct(data['sma_20_pct'])}</td>
                    <td>{data['sma_50']:.2f} {pct(data['sma_50_pct'])}</td>
                    <td>{data['sma_200']:.2f} {pct(data['sma_200_pct'])}</td>
                    <td>{data['bb_upper']:.2f} {pct(data['bb_upper_pct'])}</td>
                    <td>{data['bb_lower']:.2f} {pct(data['bb_lower_pct'])}</td>
                    <td>{data['pivot_high']:.2f} {pct(data['pivot_high_pct'])}</td>
                    <td>{data['pivot_low']:.2f} {pct(data['pivot_low_pct'])}</td>
                </tr>
                """
            html_content += "</table>"
        html_content += """
            </div>
        </body>
        </html>
        """
        return html_content
    
    def generate_reports(self) -> None:
        """Generate all reports and dashboards"""
        try:
            # Load summary data
            summary_data = self.load_summary_data()
            
            # Generate HTML report
            html_content = self.generate_html_report(summary_data)
            
            # Create report filename with date
            date_str = summary_data['analysis_date']
            report_filename = f"market_report_{date_str}.html"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            self.logger.info(f"Saved HTML report to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize report generator
    analyzer = ReportGenerator('analysis_output')
    
    try:
        # Generate reports
        print("\nGenerating reports and dashboards...")
        analyzer.generate_reports()
        print("\nReports generated successfully!")
        
    except Exception as e:
        print(f"\nError during report generation: {str(e)}") 