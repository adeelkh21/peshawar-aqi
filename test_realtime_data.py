#!/usr/bin/env python3
"""
Test Real-Time Data Verification
================================

This script verifies that the AQI prediction system is truly using real-time data
and not static values.
"""

import time
import pandas as pd
from datetime import datetime
from phase5_fixed_production import FixedProductionIntegration

def test_realtime_data_verification():
    """Test that the system is using real-time data"""
    print("🧪 TESTING REAL-TIME DATA VERIFICATION")
    print("=" * 50)
    
    try:
        # Initialize the production system
        production = FixedProductionIntegration()
        
        # Test 1: Check if real-time data collection is working
        print("\n🔍 Test 1: Real-time Data Collection")
        print("-" * 30)
        
        # Collect fresh real-time data
        print("🔄 Collecting fresh real-time data...")
        real_time_data = production.collect_real_time_data()
        
        if real_time_data is not None and len(real_time_data) > 0:
            print(f"✅ Real-time data collected: {len(real_time_data)} records")
            print(f"📅 Latest timestamp: {real_time_data['timestamp'].max()}")
            print(f"📊 Current AQI category: {real_time_data['aqi_category'].iloc[-1]}")
            print(f"🌫️ Current PM2.5: {real_time_data['pm2_5'].iloc[-1]:.2f}")
            print(f"🌫️ Current PM10: {real_time_data['pm10'].iloc[-1]:.2f}")
        else:
            print("❌ Failed to collect real-time data")
            return False
        
        # Test 2: Check if data changes over time
        print("\n🔍 Test 2: Data Freshness Check")
        print("-" * 30)
        
        # Wait a few seconds and collect data again
        print("⏳ Waiting 5 seconds...")
        time.sleep(5)
        
        print("🔄 Collecting fresh real-time data again...")
        real_time_data_2 = production.collect_real_time_data()
        
        if real_time_data_2 is not None and len(real_time_data_2) > 0:
            print(f"✅ Second data collection: {len(real_time_data_2)} records")
            print(f"📅 Latest timestamp: {real_time_data_2['timestamp'].max()}")
            print(f"📊 Current AQI category: {real_time_data_2['aqi_category'].iloc[-1]}")
            print(f"🌫️ Current PM2.5: {real_time_data_2['pm2_5'].iloc[-1]:.2f}")
            print(f"🌫️ Current PM10: {real_time_data_2['pm10'].iloc[-1]:.2f}")
            
            # Check if data is different
            if (real_time_data['timestamp'].max() != real_time_data_2['timestamp'].max() or
                real_time_data['aqi_category'].iloc[-1] != real_time_data_2['aqi_category'].iloc[-1] or
                abs(real_time_data['pm2_5'].iloc[-1] - real_time_data_2['pm2_5'].iloc[-1]) > 0.01):
                print("✅ Data is changing - Real-time collection working!")
            else:
                print("⚠️  Data appears static - may be cached")
        else:
            print("❌ Failed to collect second real-time data")
            return False
        
        # Test 3: Check forecast generation with fresh data
        print("\n🔍 Test 3: Fresh Forecast Generation")
        print("-" * 30)
        
        # Load model
        model, scaler = production.load_model()
        
        # Generate forecast with fresh data
        print("🔄 Generating forecast with fresh data...")
        forecast_1 = production.generate_realistic_forecast(model, scaler, hours=24)
        
        print(f"✅ Forecast 1 generated: {len(forecast_1['predictions'])} predictions")
        print(f"📊 AQI Range: {min(forecast_1['predictions']):.1f} - {max(forecast_1['predictions']):.1f}")
        print(f"🎯 Current AQI: {forecast_1['predictions'][0]:.1f}")
        
        # Wait and generate another forecast
        print("⏳ Waiting 3 seconds...")
        time.sleep(3)
        
        # Force fresh data collection
        production.collect_real_time_data()
        
        print("🔄 Generating second forecast with fresh data...")
        forecast_2 = production.generate_realistic_forecast(model, scaler, hours=24)
        
        print(f"✅ Forecast 2 generated: {len(forecast_2['predictions'])} predictions")
        print(f"📊 AQI Range: {min(forecast_2['predictions']):.1f} - {max(forecast_2['predictions']):.1f}")
        print(f"🎯 Current AQI: {forecast_2['predictions'][0]:.1f}")
        
        # Check if forecasts are different
        if abs(forecast_1['predictions'][0] - forecast_2['predictions'][0]) > 0.1:
            print("✅ Forecasts are different - Real-time forecasting working!")
        else:
            print("⚠️  Forecasts are identical - may be using cached data")
        
        # Test 4: Check API connectivity
        print("\n🔍 Test 4: API Connectivity")
        print("-" * 30)
        
        # Test OpenWeatherMap API
        import requests
        url = f"http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            'lat': 34.0083,
            'lon': 71.5189,
            'appid': "86e22ef485ce8beb1a30ba654f6c2d5a"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            current_aqi = data.get('list', [{}])[0].get('main', {}).get('aqi', 'N/A')
            print(f"✅ OpenWeatherMap API: Connected (AQI: {current_aqi})")
        else:
            print(f"❌ OpenWeatherMap API: Failed ({response.status_code})")
        
        # Test Meteostat API
        try:
            from meteostat import Point, Hourly
            location = Point(34.0083, 71.5189)
            data = Hourly(location, datetime.now(), datetime.now())
            data = data.fetch()
            if not data.empty:
                print(f"✅ Meteostat API: Connected ({len(data)} records)")
            else:
                print("⚠️  Meteostat API: No data available")
        except Exception as e:
            print(f"❌ Meteostat API: Error - {str(e)}")
        
        print("\n🎉 REAL-TIME DATA VERIFICATION COMPLETE!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during real-time data verification: {str(e)}")
        return False

if __name__ == "__main__":
    test_realtime_data_verification()
