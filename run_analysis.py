import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import yaml
from data_collector import DataCollector
from market_analyzer import MarketAnalyzer
from bias_analyzer import BiasAnalyzer
from summary_generator import SummaryGenerator
from dashboard import Dashboard
from file_manager import FileManager
from ticker_manager import TickerManager

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
    parser.add_argument('--bias', action='store_true', help='Run bias analysis')
    parser.add_argument('--summary', action='store_true', help='Generate summary')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--keep-hours', type=int, default=24, help='Keep data for specified hours')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize FileManager and TickerManager
        file_manager = FileManager()
        ticker_manager = TickerManager(file_manager)
        
        # Initialize components
        collector = DataCollector(file_manager)
        market_analyzer = MarketAnalyzer(file_manager, ticker_manager)
        bias_analyzer = BiasAnalyzer(file_manager, ticker_manager)
        summary_generator = SummaryGenerator(file_manager)
        dashboard = Dashboard(file_manager)

        # Run pipeline components based on arguments
        if args.collect_data:
            logger.info("Starting data collection...")
            collector.collect_all_data()

        # If summary is requested, ensure analysis is run first
        if args.summary and not (args.analyze or args.bias):
            logger.info("Summary requested but no analysis specified. Running both market and bias analysis...")
            args.analyze = True
            args.bias = True

        if args.analyze:
            logger.info("Starting market analysis...")
            market_results = market_analyzer.analyze_all_tickers()
            if market_results:
                logger.info("Market analysis completed successfully")

        if args.bias:
            logger.info("Starting bias analysis...")
            bias_results = bias_analyzer.analyze_all_tickers()
            if bias_results:
                logger.info("Bias analysis completed successfully")

        if args.summary:
            logger.info("Generating summary...")
            summary = summary_generator.generate_summary()
            if summary:
                summary_generator.save_summary(summary)
                logger.info("Summary generated and saved successfully")

        if args.dashboard:
            logger.info("Launching dashboard...")
            dashboard.run()

        # Clean up old data
        cutoff_time = datetime.now() - timedelta(hours=args.keep_hours)
        file_manager.cleanup_old_files(cutoff_time)

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        logger.error("Full error details:", exc_info=True)

if __name__ == "__main__":
    main() 