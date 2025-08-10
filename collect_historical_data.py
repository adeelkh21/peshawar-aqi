"""
AQI Prediction System - Historical Data Collection
===============================================

This script performs a one-time collection of 150 days historical data:
- Weather data from Meteostat API
- Pollution data from OpenWeatherMap API
- Stores data in a separate historical data directory
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import time
import warnings
import json
from logging_config import setup_logging
from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = "86e22ef485ce8beb1a30ba654f6c2d5a"
COLLECTION_DAYS = 150

class HistoricalDataCollector:
    def __init__(self):
        """Initialize historical data collector"""
        print("🔄 Initializing Historical Data Collection")
        print("=" * 50)
        
        # Initialize dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=COLLECTION_DAYS)
        
        # Create historical data directory
        self.data_dir = os.path.join("data", "historical", f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "metadata"), exist_ok=True)
        
        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {COLLECTION_DAYS} days")
        print(f"📂 Data Directory: {self.data_dir}")

    def fetch_weather_data(self):
        """Fetch historical weather data from Meteostat"""
        print("\n🌤️ Fetching Historical Weather Data")
        print("-" * 40)
        
        try:
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            data = Hourly(location, self.start_date, self.end_date)
            df = data.fetch()
            
            if df is None or df.empty:
                print("❌ No weather data received!")
                return None
            
            df.reset_index(inplace=True)
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
            
            # Save raw data
            weather_file = os.path.join(self.data_dir, "raw", "historical_weather.csv")
            df.to_csv(weather_file, index=False)
            
            # Save metadata
            metadata = {
                "collection_date": datetime.now().isoformat(),
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "weather_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Historical weather data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return None

    def fetch_pollution_data(self):
        """Fetch historical pollution data from OpenWeatherMap"""
        print("\n🏭 Fetching Historical Pollution Data")
        print("-" * 40)
        
        try:
            end_timestamp = int(self.end_date.timestamp())
            start_timestamp = int(self.start_date.timestamp())
            
            results = []
            retry_days = []
            
            print(f"📅 Collecting data from {self.start_date.date()} to {self.end_date.date()}")
            
            for day in range(COLLECTION_DAYS):
                if day % 10 == 0:
                    print(f"📦 Progress: {day+1}/{COLLECTION_DAYS} days ({(day+1)/COLLECTION_DAYS*100:.1f}%)")
                
                day_start = start_timestamp + day * 86400
                day_end = day_start + 86399
                
                url = (
                    f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
                    f"lat={PESHAWAR_LAT}&lon={PESHAWAR_LON}&"
                    f"start={day_start}&end={day_end}&"
                    f"appid={OPENWEATHER_API_KEY}"
                )
                
                try:
                    response = requests.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('list', []):
                            record = {
                                "timestamp": datetime.utcfromtimestamp(item['dt']),
                                "aqi_category": item['main']['aqi'],
                                **item['components']
                            }
                            results.append(record)
                    else:
                        print(f"⚠️ Failed day {day+1}, status: {response.status_code}")
                        if response.status_code == 429:  # Rate limit
                            print("Rate limit hit, waiting 60 seconds...")
                            time.sleep(60)
                        retry_days.append(day)
                    
                    time.sleep(1.2)  # Rate limiting
                    
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Network error on day {day+1}: {str(e)}")
                    retry_days.append(day)
            
            # Retry failed days
            if retry_days:
                print(f"\n🔄 Retrying {len(retry_days)} failed days...")
                for day in retry_days:
                    print(f"Retrying day {day+1}...")
                    day_start = start_timestamp + day * 86400
                    day_end = day_start + 86399
                    
                    try:
                        time.sleep(2)  # Extra delay for retries
                        response = requests.get(
                            f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
                            f"lat={PESHAWAR_LAT}&lon={PESHAWAR_LON}&"
                            f"start={day_start}&end={day_end}&"
                            f"appid={OPENWEATHER_API_KEY}",
                            timeout=15
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            for item in data.get('list', []):
                                record = {
                                    "timestamp": datetime.utcfromtimestamp(item['dt']),
                                    "aqi_category": item['main']['aqi'],
                                    **item['components']
                                }
                                results.append(record)
                            print(f"✅ Successfully retrieved day {day+1}")
                        else:
                            print(f"❌ Failed to retrieve day {day+1} on retry")
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Network error on retry for day {day+1}: {str(e)}")
            
            if not results:
                print("❌ No pollution data collected!")
                return None
            
            df = pd.DataFrame(results)
            df = df.sort_values('timestamp')
            
            # Save raw data
            pollution_file = os.path.join(self.data_dir, "raw", "historical_pollution.csv")
            df.to_csv(pollution_file, index=False)
            
            # Save metadata
            metadata = {
                "collection_date": datetime.now().isoformat(),
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict(),
                "retry_attempts": len(retry_days)
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "pollution_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            # Calculate collection statistics
            total_hours = len(df)
            expected_hours = COLLECTION_DAYS * 24
            coverage = (total_hours / expected_hours) * 100
            
            print("\n📊 Collection Summary:")
            print(f"✅ Records collected: {total_hours:,} out of {expected_hours:,} hours")
            print(f"📈 Data coverage: {coverage:.1f}%")
            print(f"🗓️ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"📋 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching pollution data: {str(e)}")
            return None

    def process_historical_data(self, weather_df, pollution_df):
        """Process and merge historical data"""
        print("\n🔄 Processing Historical Data")
        print("-" * 40)
        
        try:
            # Ensure timestamps are datetime
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])
            
            # Round timestamps to nearest hour
            weather_df['timestamp'] = weather_df['timestamp'].dt.floor('H')
            pollution_df['timestamp'] = pollution_df['timestamp'].dt.floor('H')
            
            # Merge data
            df = pd.merge(
                pollution_df,
                weather_df,
                on='timestamp',
                how='inner'
            )
            
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Save processed data
            processed_file = os.path.join(self.data_dir, "processed", "historical_merged.csv")
            df.to_csv(processed_file, index=False)
            
            # Save metadata
            metadata = {
                "processing_date": datetime.now().isoformat(),
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "features": list(df.columns),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "processed_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Historical data processing completed")
            print(f"📊 Final dataset shape: {df.shape}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None

    def run_collection(self):
        """Run historical data collection pipeline"""
        print("\n🚀 Starting Historical Data Collection")
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
        final_df = self.process_historical_data(weather_df, pollution_df)
        if final_df is None:
            return False
        
        print("\n✅ Historical Data Collection Completed Successfully!")
        print("=" * 50)
        print("📁 Files saved in:")
        print(f"   {self.data_dir}")
        
        return True

def main():
    """Run historical data collection"""
    collector = HistoricalDataCollector()
    success = collector.run_collection()
    
    if success:
        print("\n🎉 Historical data collection completed!")
    else:
        print("\n❌ Historical data collection failed! Check error messages above.")

if __name__ == "__main__":
    main()
