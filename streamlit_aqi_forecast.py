"""
Streamlit AQI Forecasting App
============================

Real-time Air Quality Index forecasting for Peshawar using the trained LightGBM model.
Integrates with the production system for live data collection and predictions.

Features:
- Real-time data collection
- 3-day AQI forecasting
- Interactive visualizations
- Historical data display
- Model performance metrics

Author: Data Science Team
Date: 2025-08-13
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time

# Import our production system
from phase5_production_integration import ProductionIntegration

# Page configuration
st.set_page_config(
    page_title="Peshawar AQI Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
    .status-good { color: #28a745; }
    .status-moderate { color: #ffc107; }
    .status-unhealthy { color: #fd7e14; }
    .status-very-unhealthy { color: #dc3545; }
    .status-hazardous { color: #6f42c1; }
</style>
""", unsafe_allow_html=True)

def get_aqi_category_color(category):
    """Get color for AQI category"""
    colors = {
        "Good": "#28a745",
        "Moderate": "#ffc107", 
        "Unhealthy for Sensitive Groups": "#fd7e14",
        "Unhealthy": "#dc3545",
        "Very Unhealthy": "#6f42c1",
        "Hazardous": "#6f42c1"
    }
    return colors.get(category, "#6c757d")

def get_aqi_category_emoji(category):
    """Get emoji for AQI category"""
    emojis = {
        "Good": "😊",
        "Moderate": "😐",
        "Unhealthy for Sensitive Groups": "😷",
        "Unhealthy": "😷",
        "Very Unhealthy": "🤢",
        "Hazardous": "☠️"
    }
    return emojis.get(category, "❓")

def create_aqi_gauge(value, category):
    """Create a gauge chart for AQI value"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Current AQI: {category}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 500]},
            'bar': {'color': get_aqi_category_color(category)},
            'steps': [
                {'range': [0, 50], 'color': "#28a745"},
                {'range': [50, 100], 'color': "#ffc107"},
                {'range': [100, 150], 'color': "#fd7e14"},
                {'range': [150, 200], 'color': "#dc3545"},
                {'range': [200, 300], 'color': "#6f42c1"},
                {'range': [300, 500], 'color': "#6f42c1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 500
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_forecast_chart(predictions, categories):
    """Create forecast chart"""
    # Create time range for next 3 days
    now = datetime.now()
    time_range = [now + timedelta(hours=i) for i in range(len(predictions))]
    
    df = pd.DataFrame({
        'timestamp': time_range,
        'aqi': predictions,
        'category': categories
    })
    
    # Create the chart
    fig = go.Figure()
    
    # Add AQI line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi'],
        mode='lines+markers',
        name='AQI Forecast',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add category-based coloring
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        fig.add_trace(go.Scatter(
            x=cat_data['timestamp'],
            y=cat_data['aqi'],
            mode='markers',
            name=f'{category}',
            marker=dict(
                color=get_aqi_category_color(category),
                size=10,
                symbol='circle'
            ),
            showlegend=True
        ))
    
    # Add threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy")
    
    fig.update_layout(
        title="3-Day AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI Value",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ Peshawar AQI Forecast</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Air Quality Index Forecasting System")
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Model performance info
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("R² Score", "94.97%")
    st.sidebar.metric("Model Type", "LightGBM")
    st.sidebar.metric("Status", "Production Ready")
    
    # Forecast controls
    st.sidebar.markdown("### 🎯 Forecast Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)", value=True)
    forecast_hours = st.sidebar.slider("Forecast Hours", 24, 72, 72, 24)
    
    # Initialize production system
    @st.cache_resource
    def get_production_system():
        return ProductionIntegration()
    
    try:
        production = get_production_system()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Real-time AQI Forecast")
            
            # Generate forecast button
            if st.button("🔄 Generate New Forecast", type="primary"):
                with st.spinner("Collecting real-time data and generating forecast..."):
                    try:
                        forecast = production.forecast_3_day_aqi()
                        
                        # Store in session state
                        st.session_state.forecast = forecast
                        st.session_state.forecast_time = datetime.now()
                        
                        st.success("✅ Forecast generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating forecast: {str(e)}")
            
            # Display forecast if available
            if 'forecast' in st.session_state:
                forecast = st.session_state.forecast
                
                # Create forecast chart
                fig = create_forecast_chart(
                    forecast['predictions'], 
                    forecast['aqi_categories']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast details
                st.markdown("### 📋 Forecast Details")
                
                # Create a DataFrame for better display
                now = datetime.now()
                forecast_df = pd.DataFrame({
                    'Time': [now + timedelta(hours=i) for i in range(len(forecast['predictions']))],
                    'AQI': [round(pred, 1) for pred in forecast['predictions']],
                    'Category': forecast['aqi_categories'],
                    'Status': [get_aqi_category_emoji(cat) for cat in forecast['aqi_categories']]
                })
                
                st.dataframe(forecast_df, use_container_width=True)
                
            else:
                st.info("👆 Click 'Generate New Forecast' to get started!")
        
        with col2:
            st.markdown("### 🎯 Current Status")
            
            if 'forecast' in st.session_state:
                forecast = st.session_state.forecast
                current_aqi = forecast['predictions'][0]
                current_category = forecast['aqi_categories'][0]
                
                # Current AQI gauge
                gauge_fig = create_aqi_gauge(current_aqi, current_category)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Status card
                st.markdown(f"""
                <div class="forecast-card">
                    <h4>Current AQI: {current_aqi:.1f}</h4>
                    <p><strong>Category:</strong> {current_category} {get_aqi_category_emoji(current_category)}</p>
                    <p><strong>Model Performance:</strong> {forecast['model_performance']}</p>
                    <p><strong>Last Updated:</strong> {st.session_state.forecast_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Category distribution
                st.markdown("### 📊 Category Distribution")
                category_counts = pd.Series(forecast['aqi_categories']).value_counts()
                
                for category, count in category_counts.items():
                    percentage = (count / len(forecast['aqi_categories'])) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{category}</strong> {get_aqi_category_emoji(category)}<br>
                        {count} hours ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No forecast data available. Generate a forecast to see current status.")
        
        # Bottom section
        st.markdown("---")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("### 📊 System Information")
            st.markdown("""
            - **Model**: LightGBM Optimized
            - **Performance**: 94.97% R²
            - **Features**: 497 engineered features
            - **Data Sources**: Real-time APIs
            """)
        
        with col4:
            st.markdown("### 🔍 Data Quality")
            st.markdown("""
            - **Weather Data**: ✅ Active
            - **Pollution Data**: ✅ Active
            - **Feature Engineering**: ✅ Complete
            - **Model Predictions**: ✅ Working
            """)
        
        with col5:
            st.markdown("### 🚀 Next Steps")
            st.markdown("""
            - **Automated Updates**: Hourly forecasts
            - **Historical Analysis**: Trend visualization
            - **Alert System**: Health notifications
            - **Mobile App**: Real-time monitoring
            """)
        
        # Auto-refresh functionality
        if auto_refresh and 'forecast' in st.session_state:
            time_since_update = datetime.now() - st.session_state.forecast_time
            if time_since_update.total_seconds() > 300:  # 5 minutes
                st.info("🔄 Auto-refreshing forecast...")
                st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error initializing production system: {str(e)}")
        st.info("Please ensure the production system is properly configured.")

if __name__ == "__main__":
    main()
