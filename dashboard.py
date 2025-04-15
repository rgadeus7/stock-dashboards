import streamlit as st
import json
import os
from datetime import datetime
from file_manager import FileManager
import plotly.graph_objects as go
import pandas as pd

class Dashboard:
    def __init__(self, file_manager: FileManager):
        """Initialize the Dashboard with a FileManager instance"""
        self.file_manager = file_manager
    
    def load_latest_summary(self) -> dict:
        """Load the latest summary file"""
        try:
            # Use the correct path to find the summary file
            summary_file = os.path.join(os.path.dirname(__file__), 'reports', 'market_summary_2025-04-14.json')
            
            if not os.path.exists(summary_file):
                st.error(f"Summary file not found at: {summary_file}")
                return {}
            
            st.write(f"Loading summary file: {summary_file}")
            with open(summary_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            st.error(f"Error loading summary file: {str(e)}")
            return {}
    
    def create_price_chart(self, symbol: str, interval: str, market_data: dict) -> go.Figure:
        """Create a price chart with technical indicators"""
        try:
            # Convert price data to DataFrame
            prices = market_data.get('prices', [])
            if not prices:
                return None
            
            df = pd.DataFrame(prices)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            )])
            
            # Add moving averages if available
            if 'ma20' in market_data:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=[market_data['ma20']] * len(df),
                    name='MA20',
                    line=dict(color='blue', width=1)
                ))
            
            if 'ma50' in market_data:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=[market_data['ma50']] * len(df),
                    name='MA50',
                    line=dict(color='orange', width=1)
                ))
            
            # Add support and resistance levels
            support_levels = market_data.get('support_resistance', {}).get('support', [])
            resistance_levels = market_data.get('support_resistance', {}).get('resistance', [])
            
            for level in support_levels:
                fig.add_hline(y=level, line_dash="dash", line_color="green", name=f"Support {level}")
            
            for level in resistance_levels:
                fig.add_hline(y=level, line_dash="dash", line_color="red", name=f"Resistance {level}")
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - {interval}",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def run(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="Market Health Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Market Health Dashboard")
        
        # Load latest summary
        summary = self.load_latest_summary()
        if not summary:
            return
        
        # Display date
        st.sidebar.header("Analysis Date")
        st.sidebar.write(summary['date'])
        
        # Symbol selection
        symbols = list(summary['market_overview'].keys())
        selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)
        
        # Interval selection
        intervals = list(summary['market_overview'][selected_symbol].keys())
        selected_interval = st.sidebar.selectbox("Select Interval", intervals)
        
        # Market Overview
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        
        market_data = summary['market_overview'][selected_symbol][selected_interval]
        col1.metric("Last Price", market_data['last_price'])
        col2.metric("Trend", market_data['trend'])
        col3.metric("Volatility", market_data['volatility'])
        
        # Price Chart
        st.header("Price Analysis")
        fig = self.create_price_chart(selected_symbol, selected_interval, summary['market_overview'][selected_symbol][selected_interval])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Swing Analysis
        st.header("Swing Analysis")
        swing_data = summary['swing_analysis'][selected_symbol][selected_interval]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trend", swing_data['trend'])
        col2.metric("Overall Trend", swing_data['overall_trend'])
        col3.metric("MA20", swing_data['ma20'])
        col4.metric("MA50", swing_data['ma50'])
        
        if 'last_high' in swing_data:
            st.write(f"Last Swing High: {swing_data['last_high']['price']} on {swing_data['last_high']['date']}")
        if 'last_low' in swing_data:
            st.write(f"Last Swing Low: {swing_data['last_low']['price']} on {swing_data['last_low']['date']}")
        
        # Key Levels
        st.header("Key Levels")
        levels = summary['key_levels'][selected_symbol][selected_interval]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Support Levels")
            for level in levels['support']:
                st.write(f"• {level}")
        
        with col2:
            st.subheader("Resistance Levels")
            for level in levels['resistance']:
                st.write(f"• {level}")

if __name__ == "__main__":
    # Initialize FileManager
    file_manager = FileManager()
    
    # Create and run dashboard
    dashboard = Dashboard(file_manager)
    dashboard.run() 