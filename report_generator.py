import os
import json
from datetime import datetime
from file_manager import FileManager

class ReportGenerator:
    def __init__(self, file_manager: FileManager):
        """Initialize the ReportGenerator with a FileManager instance"""
        self.file_manager = file_manager
    
    def generate_report(self, summary: dict) -> str:
        """Generate HTML report from summary data"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Health Report - {summary['date']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .symbol-section {{ 
                    margin-bottom: 40px; 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    border-radius: 8px;
                    background: #fff;
                }}
                .interval-section {{ 
                    margin: 15px 0; 
                    padding: 15px; 
                    background: #f8f9fa; 
                    border-radius: 5px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                }}
                .metric {{
                    background: #fff;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .trend-up {{ color: #28a745; }}
                .trend-down {{ color: #dc3545; }}
                .trend-neutral {{ color: #ffc107; }}
                .levels {{
                    display: flex;
                    gap: 20px;
                    margin-top: 10px;
                }}
                .support-levels {{ color: #28a745; }}
                .resistance-levels {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Market Health Report - {summary['date']}</h1>
        """
        
        # Group all data by symbol
        for symbol in summary['market_overview'].keys():
            html += f"""
            <div class='symbol-section'>
                <h2>{symbol}</h2>
            """
            
            # Get all intervals for this symbol
            intervals = summary['market_overview'][symbol].keys()
            
            for interval in intervals:
                html += f"""
                <div class='interval-section'>
                    <h3>{interval}</h3>
                    <div class='metric-grid'>
                """
                
                # Market Overview Metrics
                market_data = summary['market_overview'][symbol][interval]
                html += f"""
                        <div class='metric'>
                            <h4>Market Overview</h4>
                            <p>Last Price: {market_data['last_price']}</p>
                            <p>Trend: <span class='trend-{market_data['trend'].lower().split()[0]}'>{market_data['trend']}</span></p>
                            <p>Volatility: {market_data['volatility']}</p>
                        </div>
                """
                
                # Swing Analysis Metrics
                swing_data = summary['swing_analysis'][symbol][interval]
                html += f"""
                        <div class='metric'>
                            <h4>Swing Analysis</h4>
                            <p>Trend: <span class='trend-{swing_data['trend']}'>{swing_data['trend']}</span></p>
                            <p>Overall Trend: <span class='trend-{swing_data['overall_trend']}'>{swing_data['overall_trend']}</span></p>
                            <p>MA20: {swing_data['ma20']}</p>
                            <p>MA50: {swing_data['ma50']}</p>
                        </div>
                """
                
                # Key Levels
                levels = summary['key_levels'][symbol][interval]
                html += f"""
                        <div class='metric'>
                            <h4>Key Levels</h4>
                            <div class='levels'>
                                <div class='support-levels'>
                                    <strong>Support:</strong><br>
                                    {'<br>'.join([f"• {level}" for level in levels['support']])}
                                </div>
                                <div class='resistance-levels'>
                                    <strong>Resistance:</strong><br>
                                    {'<br>'.join([f"• {level}" for level in levels['resistance']])}
                                </div>
                            </div>
                        </div>
                """
                
                html += """
                    </div>
                </div>
                """
            
            html += """
            </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, html_content: str, date_str: str = None) -> str:
        """Save the HTML report to a file"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        filename = f"market_report_{date_str}.html"
        file_path = os.path.join(self.file_manager.reports_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        return file_path

if __name__ == "__main__":
    # Initialize FileManager
    file_manager = FileManager()
    
    # Create ReportGenerator
    report_generator = ReportGenerator(file_manager)
    
    # Load the summary file
    summary_file = os.path.join(file_manager.reports_dir, 'market_summary_2025-04-14.json')
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Generate and save the report
    html_content = report_generator.generate_report(summary)
    report_path = report_generator.save_report(html_content, summary['date'])
    
    print(f"Report generated and saved to: {report_path}") 