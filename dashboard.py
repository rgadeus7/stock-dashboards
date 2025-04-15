import streamlit as st
import json
import os
from datetime import datetime
from file_manager import FileManager
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

class Dashboard:
    def __init__(self, file_manager: FileManager):
        """Initialize the Dashboard with a FileManager instance"""
        self.file_manager = file_manager
    
    def load_latest_summary(self) -> dict:
        """Load the latest market_analysis_summary.json file"""
        try:
            summary_files = [f for f in os.listdir(self.file_manager.analysis_dir)
                             if f.startswith('market_analysis_summary') and f.endswith('.json')]
            if not summary_files:
                st.error("No market_analysis_summary.json files found in analysis_output directory")
                return {}
            latest_file = max(summary_files)
            summary_file = os.path.join(self.file_manager.analysis_dir, latest_file)
            st.write(f"Loading summary file: {summary_file}")
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading summary file: {str(e)}")
            return {}
    
    def create_trend_chart(self, symbol: str, interval: str, trend_data: dict) -> go.Figure:
        """Create a trend visualization chart"""
        fig = go.Figure()
        
        # Convert trend strings to numeric values
        def trend_to_value(trend_str: str) -> float:
            trend_map = {
                'strong_bullish': 1.0,
                'bullish': 0.5,
                'neutral': 0.0,
                'bearish': -0.5,
                'strong_bearish': -1.0
            }
            return trend_map.get(trend_str.lower(), 0.0)
        
        # Add trend indicators
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=trend_to_value(trend_data['overall_trend']),
            title={'text': "Overall Trend"},
            gauge={'axis': {'range': [-1, 1]}},
            domain={'row': 0, 'column': 0}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=trend_to_value(trend_data['market_trend']),
            title={'text': "Market Trend"},
            gauge={'axis': {'range': [-1, 1]}},
            domain={'row': 0, 'column': 1}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=trend_to_value(trend_data['bias_trend']),
            title={'text': "Bias Trend"},
            gauge={'axis': {'range': [-1, 1]}},
            domain={'row': 0, 'column': 2}
        ))
        
        fig.update_layout(
            grid={'rows': 1, 'columns': 3, 'pattern': 'independent'},
            title=f"Trend Analysis - {symbol} ({interval})"
        )
        
        return fig
    
    def create_bias_chart(self, symbol: str, interval: str, bias_data: dict) -> go.Figure:
        """Create a bias analysis radar chart"""
        categories = ['Overall', 'Price Action', 'Volume', 'Momentum', 'Volatility']
        values = [
            bias_data['overall_bias'],
            bias_data['price_action_bias'],
            bias_data['volume_bias'],
            bias_data['momentum_bias'],
            bias_data['volatility_bias']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Bias Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1]
                )
            ),
            title=f"Bias Analysis - {symbol} ({interval})"
        )
        
        return fig
    
    def create_key_levels_chart(self, symbol: str, levels: dict) -> go.Figure:
        """Create a key levels chart for a symbol"""
        try:
            # Get values with safe conversion
            sma_20 = float(levels.get('sma_20', 0))
            sma_50 = float(levels.get('sma_50', 0))
            sma_200 = float(levels.get('sma_200', 0))
            bb_upper = float(levels.get('bb_upper', 0))
            bb_lower = float(levels.get('bb_lower', 0))
            current_price = float(levels.get('current_price', 0))
            
            # Calculate % distance from SMA and BB
            def calculate_pct(price: float, value: float) -> str:
                if value == 0 or price == 0:
                    return ""
                pct = ((price - value) / value) * 100
                sign = "+" if pct >= 0 else ""
                return f" ({sign}{pct:.2f}%)"
            
            sma_20_pct = calculate_pct(current_price, sma_20)
            sma_50_pct = calculate_pct(current_price, sma_50)
            sma_200_pct = calculate_pct(current_price, sma_200)
            bb_upper_pct = calculate_pct(current_price, bb_upper)
            bb_lower_pct = calculate_pct(current_price, bb_lower)
            
            # Create figure
            fig = go.Figure()
            
            # Add current price line with larger font
            fig.add_trace(go.Indicator(
                mode="number",
                value=current_price,
                title={"text": "Current Price", "font": {"size": 24}},
                number={"font": {"size": 36}},
                domain={"row": 0, "column": 0}
            ))
            
            # Add SMA indicators with percentage in a grid
            fig.add_trace(go.Indicator(
                mode="number",
                value=sma_20,
                title={"text": f"SMA (20){sma_20_pct}", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 1, "column": 0}
            ))
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=sma_50,
                title={"text": f"SMA (50){sma_50_pct}", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 1, "column": 1}
            ))
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=sma_200,
                title={"text": f"SMA (200){sma_200_pct}", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 1, "column": 2}
            ))
            
            # Add BB indicators with percentage
            fig.add_trace(go.Indicator(
                mode="number",
                value=bb_upper,
                title={"text": f"BB Upper{bb_upper_pct}", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 2, "column": 0}
            ))
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=bb_lower,
                title={"text": f"BB Lower{bb_lower_pct}", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 2, "column": 1}
            ))
            
            # Add bias indicator with color coding
            bias_value = float(levels.get('bias', 0))
            bias_reason = levels.get('bias_reason', 'No bias data available')
            
            # Determine bias color
            if bias_value > 0.5:
                bias_color = "#008000"  # green
            elif bias_value < -0.5:
                bias_color = "#FF0000"  # red
            else:
                bias_color = "#808080"  # gray
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=bias_value,
                title={"text": "Bias", "font": {"size": 20}},
                number={"font": {"size": 24}},
                domain={"row": 2, "column": 2}
            ))
            
            # Add bias reason with color
            fig.add_trace(go.Indicator(
                mode="number",
                value=0,
                title={"text": f"<span style='color:{bias_color}'>{bias_reason}</span>", "font": {"size": 16}},
                number={"font": {"size": 16}},
                domain={"row": 3, "column": 0}
            ))
            
            # Update layout with better spacing and margins
            fig.update_layout(
                grid={"rows": 4, "columns": 3, "pattern": "independent"},
                height=500,  # Increased height for better visibility
                margin=dict(l=50, r=50, t=50, b=50),  # Increased margins
                showlegend=False,
                title={
                    "text": f"Key Levels - {symbol}",
                    "font": {"size": 24},
                    "x": 0.5,
                    "y": 0.95
                }
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating key levels chart for {symbol}: {str(e)}")
            return None
    
    def create_price_chart(self, symbol: str, current_price: float, daily_bb_lower: float, daily_bb_upper: float, weekly_bb_lower: float, weekly_bb_upper: float) -> go.Figure:
        """Create a price chart with Bollinger Bands"""
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=[0],
            y=[current_price],
            mode='markers',
            marker=dict(size=10, color='black'),
            name='Current Price'
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[daily_bb_lower, daily_bb_upper],
            mode='lines',
            line=dict(width=2, color='black'),
            name='Daily Bollinger Bands'
        ))
        
        fig.add_trace(go.Scatter(
            x=[1, 1],
            y=[weekly_bb_lower, weekly_bb_upper],
            mode='lines',
            line=dict(width=2, color='black'),
            name='Weekly Bollinger Bands'
        ))
        
        fig.update_layout(
            title=f"Price Chart - {symbol}",
            xaxis_title="Time",
            yaxis_title="Price"
        )
        
        return fig
    
    def run(self):
        """Run the Streamlit dashboard with visual cards and charts"""
        st.set_page_config(layout="wide")
        st.title("Market Analysis Dashboard")
        
        # Load latest summary
        summary = self.load_latest_summary()
        if not summary:
            st.error("No summary data available")
            return
        
        # Create tabs for each symbol
        symbols = list(summary.get('tickers', {}).keys())
        if not symbols:
            st.error("No symbols found in summary data")
            return
        tabs = st.tabs(symbols)
        
        for tab, symbol in zip(tabs, symbols):
            with tab:
                st.header(f"{symbol} Overview")
                tf_data = summary['tickers'][symbol]
                # Show metrics for each timeframe as cards
                cols = st.columns(3)
                for i, timeframe in enumerate(['daily', 'weekly', 'monthly']):
                    if timeframe not in tf_data:
                        continue
                    data = tf_data[timeframe]
                    with cols[i]:
                        st.metric(f"{timeframe.capitalize()} Price", f"{data['current_price']:.2f}")
                        st.metric(f"{timeframe.capitalize()} RSI", f"{data['rsi']:.2f}")
                        st.metric(f"{timeframe.capitalize()} Bias", data['bias_reason'])
                # Show key levels as Plotly indicator grid for each timeframe
                for timeframe in ['daily', 'weekly', 'monthly']:
                    if timeframe not in tf_data:
                        continue
                    st.markdown(f"**{timeframe.capitalize()} Key Levels**")
                    st.plotly_chart(self.create_key_levels_chart(symbol, tf_data[timeframe]), use_container_width=True)
                # Show daily historical price chart if available
                if 'daily' in tf_data and 'historical_data' in tf_data['daily']:
                    hist = tf_data['daily']['historical_data'].get('data', [])
                    if hist:
                        st.markdown("**Daily Historical Price**")
                        df = pd.DataFrame(hist)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        fig = px.line(df, x='date', y='close', title=f"{symbol} Daily Close Price")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No daily historical price data available.")
                else:
                    st.info("No daily historical price data available.")

if __name__ == "__main__":
    # Initialize FileManager
    file_manager = FileManager()
    
    # Create and run Dashboard
    dashboard = Dashboard(file_manager)
    dashboard.run() 