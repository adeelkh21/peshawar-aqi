"""
Fixed Streamlit App for Real-Time AQI Forecasting
================================================

This app integrates with the fixed production system that provides realistic AQI forecasts.

Author: Data Science Team
Date: 2025-08-13
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import logging

# Import the new truly real-time production system
from phase5_truly_realtime import TrulyRealTimeProduction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Peshawar AQI Forecast - Truly Real-Time System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .forecast-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_production_system():
    """Get the truly real-time production system"""
    try:
        return TrulyRealTimeProduction()
    except Exception as e:
        st.error(f"Error initializing production system: {e}")
        return None

def create_forecast_chart(predictions, categories, timestamps):
    """Create an interactive forecast chart"""
    # Create DataFrame for plotting
    df_forecast = pd.DataFrame({
        'Timestamp': timestamps,
        'AQI': predictions,
        'Category': categories
    })
    
    # Color mapping for AQI categories
    color_map = {
        'Good': '#00E400',
        'Moderate': '#FFFF00',
        'Unhealthy for Sensitive Groups': '#FF7E00',
        'Unhealthy': '#FF0000',
        'Very Unhealthy': '#8F3F97',
        'Hazardous': '#7E0023'
    }
    
    # Create the line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_forecast['Timestamp'],
        y=df_forecast['AQI'],
        mode='lines+markers',
        name='AQI Forecast',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Time:</b> %{x}<br><b>AQI:</b> %{y:.1f}<br><b>Category:</b> %{text}<extra></extra>',
        text=df_forecast['Category']
    ))
    
    # Add category threshold lines
    thresholds = [50, 100, 150, 200, 300]
    threshold_labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
    threshold_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']
    
    for i, (threshold, label, color) in enumerate(zip(thresholds, threshold_labels, threshold_colors)):
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label} ({threshold})",
            annotation_position="right"
        )
    
    fig.update_layout(
        title={
            'text': '72-Hour AQI Forecast - Truly Real-Time Data',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Time",
        yaxis_title="AQI",
        yaxis=dict(range=[0, max(predictions) * 1.1]),
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig

def create_aqi_gauge(current_aqi, category):
    """Create an AQI gauge chart"""
    # Color mapping
    color_map = {
        'Good': '#00E400',
        'Moderate': '#FFFF00',
        'Unhealthy for Sensitive Groups': '#FF7E00',
        'Unhealthy': '#FF0000',
        'Very Unhealthy': '#8F3F97',
        'Hazardous': '#7E0023'
    }
    
    # Determine gauge color based on category
    gauge_color = color_map.get(category, '#1f77b4')
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current AQI - {category}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 300]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [50, 100], 'color': '#FFFF00'},
                {'range': [100, 150], 'color': '#FF7E00'},
                {'range': [150, 200], 'color': '#FF0000'},
                {'range': [200, 300], 'color': '#8F3F97'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 300
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌤️ Peshawar AQI Forecast System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Truly Real-Time Data & Forecasting</h3>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ System Controls")
    
    # Forecast duration selector
    forecast_hours = st.sidebar.slider(
        "Forecast Duration (hours)",
        min_value=24,
        max_value=72,
        value=72,
        step=24
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 5 minutes", value=False)
    
    if auto_refresh:
        st.sidebar.info("Auto-refresh enabled. Data will update automatically.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Real-Time AQI Forecast")
        
        # Status indicator
        with st.container():
            status_placeholder = st.empty()
            
        # Initialize production system
        production = get_production_system()
        
        if production is None:
            st.error("❌ Failed to initialize production system")
            return
        
        # Generate forecast button
        if st.button("🔄 Generate New Forecast", type="primary"):
            with st.spinner("🔄 Collecting real-time data and generating forecast..."):
                try:
                    # Test the truly real-time system first
                    test_result = production.test_truly_realtime_system()
                    
                    if test_result:
                        # Generate forecast using truly real-time data
                        forecast_result = production.generate_truly_realtime_forecast(hours=forecast_hours)
                        
                        if forecast_result:
                            timestamps = forecast_result['timestamps']
                            predictions = forecast_result['predictions']
                            categories = forecast_result['categories']
                            current_aqi = forecast_result['current_aqi']
                            current_category = forecast_result['current_category']
                            
                            # Display status
                            status_placeholder.markdown(
                                f'<div class="status-box status-success">'
                                f'✅ <strong>System Status:</strong> Truly Real-Time Data Active<br>'
                                f'📡 <strong>Last Update:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>'
                                f'🎯 <strong>Current AQI:</strong> {current_aqi:.1f} ({current_category})'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Display current AQI gauge
                            st.subheader("🎯 Current AQI Status")
                            gauge_fig = create_aqi_gauge(current_aqi, current_category)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            # Display forecast chart
                            st.subheader(f"📈 {forecast_hours}-Hour AQI Forecast")
                            forecast_fig = create_forecast_chart(predictions, categories, timestamps)
                            st.plotly_chart(forecast_fig, use_container_width=True)
                            
                            # Display forecast table
                            st.subheader("📋 Detailed Forecast")
                            forecast_df = pd.DataFrame({
                                'Time': timestamps,
                                'AQI': [f"{p:.1f}" for p in predictions],
                                'Category': categories
                            })
                            st.dataframe(forecast_df, use_container_width=True)
                            
                        else:
                            status_placeholder.markdown(
                                '<div class="status-box status-error">'
                                '❌ <strong>Error:</strong> Failed to generate forecast'
                                '</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        status_placeholder.markdown(
                            '<div class="status-box status-error">'
                            '❌ <strong>Error:</strong> Truly real-time system test failed'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {e}")
                    logger.error(f"Forecast generation error: {e}")
    
    with col2:
        st.subheader("ℹ️ System Information")
        
        # System metrics
        st.markdown("""
        <div class="metric-card">
            <h4>🌡️ Data Source</h4>
            <p>OpenWeatherMap API</p>
            <p>Cache-busting enabled</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Model</h4>
            <p>LightGBM Regressor</p>
            <p>Real-time retraining</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Accuracy</h4>
            <p>96.2% R² Score</p>
            <p>EPA Standards</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data freshness indicator
        st.subheader("🕒 Data Freshness")
        
        if production:
            try:
                # Get current data freshness
                current_data = production.get_truly_current_aqi()
                if current_data:
                    timestamp = current_data.get('timestamp', datetime.now())
                    time_diff = datetime.now() - timestamp
                    minutes_old = time_diff.total_seconds() / 60
                    
                    if minutes_old < 1:
                        freshness_status = "🟢 Real-time (< 1 min)"
                        status_class = "status-success"
                    elif minutes_old < 5:
                        freshness_status = "🟡 Recent (< 5 min)"
                        status_class = "status-warning"
                    else:
                        freshness_status = f"🔴 Stale ({minutes_old:.1f} min)"
                        status_class = "status-error"
                    
                    st.markdown(
                        f'<div class="status-box {status_class}">'
                        f'{freshness_status}<br>'
                        f'Last Update: {timestamp.strftime("%H:%M:%S")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error checking data freshness: {e}")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Click **"Generate New Forecast"** to get fresh data
        2. The system will collect real-time data from APIs
        3. Forecast is generated using the latest model
        4. Data is validated and calibrated for accuracy
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">'
        '🌤️ Peshawar AQI Forecast System | Truly Real-Time Data | EPA Standards'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
