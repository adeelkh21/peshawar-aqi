"""
AQI Prediction System - Data Collection Pipeline
=============================================

This script collects historical weather and pollution data for AQI prediction:
- Weather data from Meteostat API
- Pollution data from OpenWeatherMap API
- 150 days of historical hourly data
- Converts categorical AQI to numerical values

Author: Data Science Team
Date: 2024-03-09
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import meteostat
import time
import warnings
import json
import logging
from logging_config import setup_logging
from data_validation import DataValidator
from api_utils import APIClient
warnings.filterwarnings('ignore')

# Using Meteostat with default configuration

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
COLLECTION_DAYS = 1  # Collect last 24 hours for hourly updates

class DataCollector:
    def __init__(self):
        # Initialize data validator
        self.validator = DataValidator()
        
        # Initialize API clients
        self.weather_api = APIClient("https://api.openweathermap.org/data/2.5")
        self.pollution_api = APIClient(
            "http://api.openweathermap.org/data/2.5",
            api_key=OPENWEATHER_API_KEY
        )
        """Initialize data collector"""
        print("🔄 Initializing AQI Data Collection Pipeline")
        print("=" * 50)
        
        # Initialize dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=COLLECTION_DAYS)
        
        # Create enhanced directory structure
        self.collection_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_version = f"v1_{self.collection_timestamp}"
        
        # Create versioned data directories
        self.data_dir = os.path.join("data", self.data_version)
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "metadata"), exist_ok=True)
        
        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {COLLECTION_DAYS} days")
        print(f"📂 Data Version: {self.data_version}")

    def fetch_weather_data(self):
        """Fetch weather data from Meteostat"""
        print("\n🌤️ Fetching Weather Data")
        print("-" * 30)
        
        try:
            # Create Point instance inside the method
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            
            # Fetch hourly data
            data = Hourly(location, self.start_date, self.end_date)
            df = data.fetch()
            
            if df is None or df.empty:
                print("❌ No weather data received!")
                return None
                
            # Reset index to get timestamp as column
            df.reset_index(inplace=True)
            
            # Rename columns for clarity
            df.rename(columns={
                'time': 'timestamp',
                'temp': 'temperature',
                'dwpt': 'dew_point',
                'rhum': 'relative_humidity',
                'prcp': 'precipitation',
                'wdir': 'wind_direction',
                'wspd': 'wind_speed',
                'pres': 'pressure'
            }, inplace=True)
            
            # Save raw data with metadata
            weather_file = os.path.join(self.data_dir, "raw", "weather_data.csv")
            df.to_csv(weather_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "weather_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Weather data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return None

    def fetch_pollution_data(self):
        """Fetch pollution data from OpenWeatherMap API"""
        print("\n🏭 Fetching Pollution Data")
        print("-" * 30)
        
        try:
            # Calculate UNIX timestamps
            end_timestamp = int(self.end_date.timestamp())
            start_timestamp = int(self.start_date.timestamp())
            total_days = COLLECTION_DAYS
            
            results = []
            
            for day in range(total_days):
                if day % 10 == 0:  # Progress update every 10 days
                    print(f"   📦 Progress: {day+1}/{total_days} days...")
                
                day_start = start_timestamp + day * 86400  # 86400 seconds = 1 day
                day_end = day_start + 86399  # Full day minus 1 second
                
                url = (
                    f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
                    f"lat={PESHAWAR_LAT}&lon={PESHAWAR_LON}&"
                    f"start={day_start}&end={day_end}&"
                    f"appid={OPENWEATHER_API_KEY}"
                )
                
                try:
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('list', []):
                            record = {
                                "timestamp": datetime.utcfromtimestamp(item['dt']),
                                "aqi_category": item['main']['aqi'],  # Original categorical AQI
                                **item['components']  # All pollutant components
                            }
                            results.append(record)
                    else:
                        print(f"   ⚠️ Warning: Failed day {day+1}, status: {response.status_code}")
                        
                    time.sleep(1.1)  # Rate limiting
                    
                except requests.exceptions.RequestException as e:
                    print(f"   ⚠️ Network error on day {day+1}: {str(e)}")
                    continue
            
            if not results:
                print("❌ No pollution data collected!")
                return None
                
            df = pd.DataFrame(results)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Save raw data with metadata
            pollution_file = os.path.join(self.data_dir, "raw", "pollution_data.csv")
            df.to_csv(pollution_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict(),
                "pollutants": [col for col in df.columns if col not in ['timestamp', 'aqi_category']]
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "pollution_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Pollution data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching pollution data: {str(e)}")
            return None

    def convert_categorical_to_numerical_aqi(self, df):
        """Convert categorical AQI to numerical values based on PM2.5 and PM10"""
        
        # PM2.5 breakpoints (μg/m³) and corresponding AQI values
        pm25_breakpoints = {
            'Good': (0.0, 12.0, 0, 50),
            'Moderate': (12.1, 35.4, 51, 100),
            'Unhealthy for Sensitive Groups': (35.5, 55.4, 101, 150),
            'Unhealthy': (55.5, 150.4, 151, 200),
            'Very Unhealthy': (150.5, 250.4, 201, 300),
            'Hazardous': (250.5, 500.4, 301, 500)
        }
        
        # PM10 breakpoints (μg/m³) and corresponding AQI values
        pm10_breakpoints = {
            'Good': (0, 54, 0, 50),
            'Moderate': (55, 154, 51, 100),
            'Unhealthy for Sensitive Groups': (155, 254, 101, 150),
            'Unhealthy': (255, 354, 151, 200),
            'Very Unhealthy': (355, 424, 201, 300),
            'Hazardous': (425, 604, 301, 500)
        }
        
        def calculate_aqi(concentration, breakpoints):
            """Calculate AQI value given concentration and breakpoint table"""
            for _, (low_conc, high_conc, low_aqi, high_aqi) in breakpoints.items():
                if low_conc <= concentration <= high_conc:
                    return np.interp(concentration, [low_conc, high_conc], [low_aqi, high_aqi])
            return 500  # Maximum AQI value
        
        # Calculate AQI based on both PM2.5 and PM10
        df['aqi_pm25'] = df['pm2_5'].apply(lambda x: calculate_aqi(x, pm25_breakpoints))
        df['aqi_pm10'] = df['pm10'].apply(lambda x: calculate_aqi(x, pm10_breakpoints))
        
        # Final AQI is the maximum of PM2.5 and PM10 AQI values
        df['aqi_numeric'] = df[['aqi_pm25', 'aqi_pm10']].max(axis=1)
        
        return df

    def merge_and_process_data(self, weather_df, pollution_df):
        """Merge weather and pollution data, process for modeling"""
        print("\n🔄 Processing and Merging Data")
        print("-" * 30)
        
        try:
            # Ensure timestamps are datetime
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])
            
            # Round timestamps to nearest hour for better matching
            weather_df['timestamp'] = weather_df['timestamp'].dt.floor('H')
            pollution_df['timestamp'] = pollution_df['timestamp'].dt.floor('H')
            
            # Merge on timestamp
            df = pd.merge(
                pollution_df,
                weather_df,
                on='timestamp',
                how='inner'
            )
            
            # Convert categorical AQI to numerical
            df = self.convert_categorical_to_numerical_aqi(df)
            
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Save processed data with metadata
            processed_file = os.path.join(self.data_dir, "processed", "merged_data.csv")
            df.to_csv(processed_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "features": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "aqi_stats": {
                    "min": float(df['aqi_numeric'].min()),
                    "max": float(df['aqi_numeric'].max()),
                    "mean": float(df['aqi_numeric'].mean()),
                    "median": float(df['aqi_numeric'].median())
                }
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "processed_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Data processing completed")
            print(f"📊 Final dataset shape: {df.shape}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"📈 Numerical AQI range: {df['aqi_numeric'].min():.1f} to {df['aqi_numeric'].max():.1f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None

    def run_pipeline(self):
        """Run complete data collection pipeline"""
        print("\n🚀 Starting Data Collection Pipeline")
        print("=" * 50)
        
        # Step 1: Fetch weather data
        weather_df = self.fetch_weather_data()
        if weather_df is None:
            return False
            
        # Step 2: Fetch pollution data
        pollution_df = self.fetch_pollution_data()
        if pollution_df is None:
            return False
            
        # Step 3: Process and merge data
        final_df = self.merge_and_process_data(weather_df, pollution_df)
        if final_df is None:
            return False
            
        print("\n✅ Data Collection Pipeline Completed Successfully!")
        print("=" * 50)
        print("📁 Files saved:")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'weather_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'pollution_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'processed', 'merged_data.csv')}")
        
        return True

def main():
    """Run data collection pipeline"""
    collector = DataCollector()
    success = collector.run_pipeline()
    
    if success:
        print("\n🎉 Ready for feature engineering!")
    else:
        print("\n❌ Pipeline failed! Check error messages above.")

if __name__ == "__main__":
    main()
