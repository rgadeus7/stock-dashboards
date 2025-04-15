import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import yaml
from data_collector import DataCollector
from market_analyzer import MarketAnalyzer
from swing_analyzer import SwingAnalyzer
from summary_generator import SummaryGenerator
from dashboard import MarketDashboard
from file_manager import FileManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Market Health Analysis Pipeline')
    parser.add_argument('--collect-data', action='store_true', help='Collect market data')
    parser.add_argument('--analyze', action='store_true', help='Run market analysis')
    parser.add_argument('--swing', action='store_true', help='Run swing analysis')
    parser.add_argument('--summary', action='store_true', help='Generate summary')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--keep-hours', type=int, default=24, help='Keep data for specified hours')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize FileManager
        file_manager = FileManager()
        
        # Initialize components
        collector = DataCollector(file_manager)
        market_analyzer = MarketAnalyzer(file_manager)
        swing_analyzer = SwingAnalyzer(file_manager)
        summary_generator = SummaryGenerator(file_manager)
        dashboard = MarketDashboard(file_manager)

        # Run pipeline components based on arguments
        if args.collect_data:
            logger.info("Starting data collection...")
            collector.collect_all_data()

        if args.analyze:
            logger.info("Starting market analysis...")
            market_results = market_analyzer.analyze_all_symbols()
            if market_results:
                logger.info("Market analysis completed successfully")

        if args.swing:
            logger.info("Starting swing analysis...")
            swing_results = swing_analyzer.analyze_all_symbols()
            if swing_results:
                logger.info("Swing analysis completed successfully")

        if args.summary:
            logger.info("Generating summary...")
            summary = summary_generator.generate_summary(market_results, swing_results)
            if summary:
                summary_generator.save_summary(summary)
                logger.info("Summary generated and saved successfully")

        if args.dashboard:
            logger.info("Launching dashboard...")
            dashboard.launch()

        # Clean up old data
        cutoff_time = datetime.now() - timedelta(hours=args.keep_hours)
        file_manager.cleanup_old_files(cutoff_time)

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        logger.error("Full error details:", exc_info=True)

if __name__ == "__main__":
    main() 