import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
from file_manager import FileManager

class SummaryGenerator:
    def __init__(self, file_manager: FileManager):
        """Initialize the SummaryGenerator with a FileManager instance"""
        self.file_manager = file_manager
        self.logger = logging.getLogger(__name__)
    
    def generate_summary(self, market_results: Dict, swing_results: Dict) -> Dict:
        """Generate a summary of market and swing analysis results"""
        try:
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_overview': {},
                'swing_analysis': {},
                'trend_analysis': {},
                'key_levels': {}
            }
            
            # Process each symbol
            for symbol, intervals in market_results.items():
                summary['market_overview'][symbol] = {}
                summary['swing_analysis'][symbol] = {}
                summary['trend_analysis'][symbol] = {}
                summary['key_levels'][symbol] = {}
                
                for interval, market_data in intervals.items():
                    # Get corresponding swing analysis
                    swing_data = swing_results.get(symbol, {}).get(interval, {})
                    
                    # Market Overview
                    summary['market_overview'][symbol][interval] = {
                        'last_price': f"{market_data.get('last_price', 0):.2f}",
                        'trend': market_data.get('trend', 'neutral'),
                        'volatility': f"{market_data.get('volatility', 0):.2f}%"
                    }
                    
                    # Swing Analysis
                    if swing_data and 'error' not in swing_data:
                        summary['swing_analysis'][symbol][interval] = {
                            'trend': swing_data.get('trend', 'neutral'),
                            'overall_trend': swing_data.get('overall_trend', 'neutral'),
                            'swing_high_count': swing_data.get('swing_high_count', 0),
                            'swing_low_count': swing_data.get('swing_low_count', 0),
                            'ma20': f"{swing_data.get('ma20', 0):.2f}",
                            'ma50': f"{swing_data.get('ma50', 0):.2f}"
                        }
                        
                        # Last swing points
                        last_high = swing_data.get('last_swing_high', {})
                        last_low = swing_data.get('last_swing_low', {})
                        
                        if last_high:
                            summary['swing_analysis'][symbol][interval]['last_high'] = {
                                'price': f"{last_high.get('price', 0):.2f}",
                                'date': last_high.get('date', '').split()[0]  # Only keep date part
                            }
                        if last_low:
                            summary['swing_analysis'][symbol][interval]['last_low'] = {
                                'price': f"{last_low.get('price', 0):.2f}",
                                'date': last_low.get('date', '').split()[0]  # Only keep date part
                            }
                    
                    # Trend Analysis
                    summary['trend_analysis'][symbol][interval] = {
                        'market_trend': market_data.get('trend', 'neutral'),
                        'swing_trend': swing_data.get('trend', 'neutral') if swing_data else 'neutral',
                        'overall_trend': swing_data.get('overall_trend', 'neutral') if swing_data else 'neutral'
                    }
                    
                    # Key Levels
                    support_resistance = market_data.get('support_resistance', {})
                    support_levels = support_resistance.get('support', [])
                    resistance_levels = support_resistance.get('resistance', [])
                    
                    # Convert single values to lists if needed
                    if isinstance(support_levels, (int, float)):
                        support_levels = [support_levels]
                    if isinstance(resistance_levels, (int, float)):
                        resistance_levels = [resistance_levels]
                    
                    summary['key_levels'][symbol][interval] = {
                        'support': [f"{level:.2f}" for level in support_levels] if isinstance(support_levels, list) else [],
                        'resistance': [f"{level:.2f}" for level in resistance_levels] if isinstance(resistance_levels, list) else []
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {'error': str(e)}
    
    def save_summary(self, summary: Dict) -> str:
        """Save the summary to a JSON file"""
        try:
            # Use only the date from the summary
            date_str = summary.get('date', datetime.now().strftime('%Y%m%d'))
            filename = f"market_summary_{date_str}.json"
            file_path = os.path.join(self.file_manager.output_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            self.logger.info(f"Summary saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {str(e)}")
            raise

if __name__ == "__main__":
    from file_manager import FileManager
    import json
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('summary_generator.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize FileManager
    file_manager = FileManager()
    
    try:
        # Load market analysis results
        market_file = file_manager.get_latest_file('market_analysis_')
        if not market_file:
            logging.error("No market analysis file found")
            exit(1)
            
        with open(market_file, 'r') as f:
            market_results = json.load(f)
        
        # Load swing analysis results
        swing_file = file_manager.get_latest_file('swing_analysis_')
        if not swing_file:
            logging.error("No swing analysis file found")
            exit(1)
            
        with open(swing_file, 'r') as f:
            swing_results = json.load(f)
        
        # Initialize SummaryGenerator with FileManager
        generator = SummaryGenerator(file_manager)
        
        # Generate and save summary
        summary = generator.generate_summary(market_results, swing_results)
        if 'error' not in summary:
            file_path = generator.save_summary(summary)
            logging.info(f"Summary generated and saved to {file_path}")
        else:
            logging.error(f"Error generating summary: {summary['error']}")
            
    except Exception as e:
        logging.error(f"Error in summary generation: {str(e)}")
        exit(1) 