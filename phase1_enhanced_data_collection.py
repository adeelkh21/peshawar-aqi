"""
Enhanced AQI Data Collection Pipeline - Phase 1
==============================================

This script provides enhanced data collection with:
- Real-time data validation
- Quality assurance checks
- Error handling and recovery
- Repository structure integration
- Data freshness monitoring

Author: Data Science Team
Date: 2024-12-08
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
import logging
from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = "86e22ef485ce8beb1a30ba654f6c2d5a"
COLLECTION_DAYS = 1  # Collect last 24 hours for hourly updates

class EnhancedDataCollector:
    """Enhanced data collector with validation and quality assurance"""
    
    def __init__(self):
        """Initialize enhanced data collector"""
        print("🔄 Initializing Enhanced AQI Data Collection Pipeline")
        print("=" * 60)
        
        # Initialize dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=COLLECTION_DAYS)
        
        # Create directories in enhanced repository structure
        self.collection_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = "data_repositories"
        
        # Ensure all required directories exist
        self._create_directory_structure()
        
        # Initialize validator
        self.validator = DataValidator(self.data_dir)
        
        # Setup logging
        self._setup_logging()
        
        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {COLLECTION_DAYS} days")
        print(f"📂 Data Directory: {self.data_dir}")
        print(f"🔍 Validation: Enabled")

    def _create_directory_structure(self):
        """Create the complete directory structure"""
        directories = [
            "raw",
            "raw/metadata", 
            "processed",
            "processed/metadata",
            "processed/validation_reports",
            "features",
            "features/validation",
            "models",
            "models/trained_models",
            "models/evaluation_reports",
            "historical_data",
            "historical_data/metadata"
        ]
        
        for dir_path in directories:
            full_path = os.path.join(self.data_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)

    def _setup_logging(self):
        """Setup logging for the collection process"""
        log_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"collection_{self.collection_timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def fetch_weather_data(self):
        """Fetch weather data with enhanced error handling and validation"""
        print("\n🌤️ Fetching Weather Data")
        print("-" * 40)
        
        try:
            self.logger.info("Starting weather data collection")
            
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            data = Hourly(location, self.start_date, self.end_date)
            df = data.fetch()
            
            if df is None or df.empty:
                self.logger.error("No weather data received from Meteostat")
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
            
            # Validate weather data
            validation_report = self.validator.validate_weather_data(df)
            
            # Save raw data
            weather_file = os.path.join(self.data_dir, "raw", "weather_data.csv")
            df.to_csv(weather_file, index=False)
            
            # Save validation report
            validation_file = os.path.join(self.data_dir, "raw", "metadata", "weather_validation.json")
            self.validator.save_validation_report(validation_report, validation_file)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min().isoformat(),
                "end_date": df['timestamp'].max().isoformat(),
                "missing_values": df.isnull().sum().to_dict(),
                "validation_status": validation_report['valid'],
                "validation_errors": len(validation_report['errors']),
                "validation_warnings": len(validation_report['warnings'])
            }
            
            metadata_file = os.path.join(self.data_dir, "raw", "metadata", "weather_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            # Log results
            self.logger.info(f"Weather data collected: {len(df)} records")
            self.logger.info(f"Validation status: {'PASS' if validation_report['valid'] else 'FAIL'}")
            
            print(f"✅ Weather data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            print(f"🔍 Validation: {'✅ PASS' if validation_report['valid'] else '❌ FAIL'}")
            
            if validation_report['warnings']:
                print(f"⚠️  Warnings: {len(validation_report['warnings'])}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {str(e)}")
            print(f"❌ Error fetching weather data: {str(e)}")
            return None

    def fetch_pollution_data(self):
        """Fetch pollution data with enhanced error handling and validation"""
        print("\n🏭 Fetching Pollution Data")
        print("-" * 40)
        
        try:
            self.logger.info("Starting pollution data collection")
            
            end_timestamp = int(self.end_date.timestamp())
            start_timestamp = int(self.start_date.timestamp())
            
            url = (
                f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
                f"lat={PESHAWAR_LAT}&lon={PESHAWAR_LON}&"
                f"start={start_timestamp}&end={end_timestamp}&"
                f"appid={OPENWEATHER_API_KEY}"
            )
            
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            data = response.json()
            results = []
            
            for item in data.get('list', []):
                record = {
                    "timestamp": datetime.utcfromtimestamp(item['dt']),
                    "aqi_category": item['main']['aqi'],
                    **item['components']
                }
                results.append(record)
            
            if not results:
                self.logger.error("No pollution data collected from API")
                print("❌ No pollution data collected!")
                return None
            
            df = pd.DataFrame(results)
            df = df.sort_values('timestamp')
            
            # Validate pollution data
            validation_report = self.validator.validate_pollution_data(df)
            
            # Save raw data
            pollution_file = os.path.join(self.data_dir, "raw", "pollution_data.csv")
            df.to_csv(pollution_file, index=False)
            
            # Save validation report
            validation_file = os.path.join(self.data_dir, "raw", "metadata", "pollution_validation.json")
            self.validator.save_validation_report(validation_report, validation_file)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min().isoformat(),
                "end_date": df['timestamp'].max().isoformat(),
                "missing_values": df.isnull().sum().to_dict(),
                "aqi_distribution": df['aqi_category'].value_counts().to_dict(),
                "validation_status": validation_report['valid'],
                "validation_errors": len(validation_report['errors']),
                "validation_warnings": len(validation_report['warnings'])
            }
            
            metadata_file = os.path.join(self.data_dir, "raw", "metadata", "pollution_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            # Log results
            self.logger.info(f"Pollution data collected: {len(df)} records")
            self.logger.info(f"Validation status: {'PASS' if validation_report['valid'] else 'FAIL'}")
            
            print(f"✅ Pollution data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            print(f"🔍 Validation: {'✅ PASS' if validation_report['valid'] else '❌ FAIL'}")
            
            if validation_report['warnings']:
                print(f"⚠️  Warnings: {len(validation_report['warnings'])}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching pollution data: {str(e)}")
            print(f"❌ Error fetching pollution data: {str(e)}")
            return None

    def merge_and_process_data(self, weather_df, pollution_df):
        """Merge and process data with enhanced validation and historical data preservation"""
        print("\n🔄 Processing and Merging Data")
        print("-" * 40)
        
        try:
            self.logger.info("Starting data processing and merging with historical preservation")
            
            # Ensure timestamps are datetime
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])
            
            # Round timestamps to nearest hour
            weather_df['timestamp'] = weather_df['timestamp'].dt.floor('H')
            pollution_df['timestamp'] = pollution_df['timestamp'].dt.floor('H')
            
            # Merge on timestamp
            new_data = pd.merge(
                pollution_df,
                weather_df,
                on='timestamp',
                how='inner'
            )
            
            # Add time-based features
            new_data['hour'] = new_data['timestamp'].dt.hour
            new_data['day'] = new_data['timestamp'].dt.day
            new_data['month'] = new_data['timestamp'].dt.month
            new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
            new_data['is_weekend'] = new_data['day_of_week'].isin([5, 6]).astype(int)
            
            # CRITICAL: Load existing historical data and merge with new data
            merged_file = os.path.join(self.data_dir, "processed", "merged_data.csv")
            historical_data = None
            
            if os.path.exists(merged_file):
                try:
                    print("📚 Loading existing historical data...")
                    historical_data = pd.read_csv(merged_file)
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                    
                    # Remove any duplicate timestamps from historical data
                    historical_data = historical_data.drop_duplicates(subset=['timestamp'], keep='first')
                    
                    print(f"📊 Historical data loaded: {len(historical_data):,} records")
                    print(f"📅 Historical range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
                    
                    # Check for overlapping timestamps
                    new_timestamps = set(new_data['timestamp'])
                    historical_timestamps = set(historical_data['timestamp'])
                    overlap = new_timestamps.intersection(historical_timestamps)
                    
                    if overlap:
                        print(f"⚠️  Found {len(overlap)} overlapping timestamps - updating with new data")
                        # Remove overlapping records from historical data
                        historical_data = historical_data[~historical_data['timestamp'].isin(overlap)]
                        print(f"📊 Historical data after overlap removal: {len(historical_data):,} records")
                    
                except Exception as e:
                    self.logger.warning(f"Could not load historical data: {e}")
                    print(f"⚠️  Could not load historical data: {e}")
                    historical_data = None
            
            # Combine historical and new data
            if historical_data is not None and len(historical_data) > 0:
                print("🔄 Merging historical and new data...")
                combined_data = pd.concat([historical_data, new_data], ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                # Remove any duplicate timestamps (keep latest)
                combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
                
                print(f"📊 Combined dataset: {len(combined_data):,} records")
                print(f"📅 Combined range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
                
                # Calculate data growth
                growth = len(combined_data) - len(historical_data)
                print(f"📈 Data growth: +{growth:,} new records")
                
            else:
                print("🆕 No historical data found - creating new dataset")
                combined_data = new_data.copy()
                print(f"📊 New dataset: {len(combined_data):,} records")
            
            # Validate combined data
            validation_report = self.validator.validate_merged_data(combined_data)
            
            # Save the combined dataset
            combined_data.to_csv(merged_file, index=False)
            
            # Save validation report
            validation_file = os.path.join(self.data_dir, "processed", "validation_reports", f"merged_validation_{self.collection_timestamp}.json")
            self.validator.save_validation_report(validation_report, validation_file)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "total_records": len(combined_data),
                "new_records": len(new_data),
                "historical_records": len(historical_data) if historical_data is not None else 0,
                "start_date": combined_data['timestamp'].min().isoformat(),
                "end_date": combined_data['timestamp'].max().isoformat(),
                "data_growth": len(combined_data) - (len(historical_data) if historical_data is not None else 0),
                "validation_status": validation_report['valid'],
                "validation_errors": len(validation_report.get('errors', [])),
                "validation_warnings": len(validation_report.get('warnings', [])),
                "preservation_status": "success"
            }
            
            metadata_file = os.path.join(self.data_dir, "processed", "metadata", f"merged_metadata_{self.collection_timestamp}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            # Log results
            self.logger.info(f"Data merging completed: {len(combined_data)} total records")
            self.logger.info(f"Historical preservation: {'SUCCESS' if historical_data is not None else 'NEW_DATASET'}")
            self.logger.info(f"Validation status: {'PASS' if validation_report['valid'] else 'FAIL'}")
            
            print(f"✅ Data merging completed successfully!")
            print(f"📊 Total records: {len(combined_data):,}")
            print(f"📈 New records added: {len(new_data):,}")
            print(f"🔍 Validation: {'✅ PASS' if validation_report['valid'] else '❌ FAIL'}")
            print(f"💾 Historical data preserved: {'✅ YES' if historical_data is not None else '🆕 NEW'}")
            
            if validation_report.get('warnings'):
                print(f"⚠️  Warnings: {len(validation_report['warnings'])}")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error in data processing and merging: {str(e)}")
            print(f"❌ Error in data processing and merging: {str(e)}")
            return None

    def generate_collection_summary(self, weather_df, pollution_df, merged_df):
        """Generate comprehensive collection summary"""
        print("\n📋 Collection Summary")
        print("-" * 40)
        
        summary = {
            "collection_timestamp": self.collection_timestamp,
            "weather_data": {
                "records": len(weather_df) if weather_df is not None else 0,
                "features": list(weather_df.columns) if weather_df is not None else [],
                "date_range": {
                    "start": weather_df['timestamp'].min().isoformat() if weather_df is not None else None,
                    "end": weather_df['timestamp'].max().isoformat() if weather_df is not None else None
                }
            },
            "pollution_data": {
                "records": len(pollution_df) if pollution_df is not None else 0,
                "features": list(pollution_df.columns) if pollution_df is not None else [],
                "date_range": {
                    "start": pollution_df['timestamp'].min().isoformat() if pollution_df is not None else None,
                    "end": pollution_df['timestamp'].max().isoformat() if pollution_df is not None else None
                }
            },
            "merged_data": {
                "records": len(merged_df) if merged_df is not None else 0,
                "features": list(merged_df.columns) if merged_df is not None else [],
                "date_range": {
                    "start": merged_df['timestamp'].min().isoformat() if merged_df is not None else None,
                    "end": merged_df['timestamp'].max().isoformat() if merged_df is not None else None
                }
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.data_dir, "collection_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        # Print summary
        print(f"📅 Collection Time: {self.collection_timestamp}")
        print(f"🌤️ Weather Records: {summary['weather_data']['records']:,}")
        print(f"🏭 Pollution Records: {summary['pollution_data']['records']:,}")
        print(f"🔄 Merged Records: {summary['merged_data']['records']:,}")
        print(f"📁 Summary saved to: {summary_file}")
        
        return summary

    def run_pipeline(self):
        """Run complete enhanced data collection pipeline"""
        print("\n🚀 Starting Enhanced Data Collection Pipeline")
        print("=" * 60)
        
        self.logger.info("Starting enhanced data collection pipeline")
        
        # Step 1: Fetch weather data
        weather_df = self.fetch_weather_data()
        if weather_df is None:
            self.logger.error("Weather data collection failed")
            return False
        
        # Step 2: Fetch pollution data
        pollution_df = self.fetch_pollution_data()
        if pollution_df is None:
            self.logger.error("Pollution data collection failed")
            return False
        
        # Step 3: Process and merge data
        merged_df = self.merge_and_process_data(weather_df, pollution_df)
        if merged_df is None:
            self.logger.error("Data processing failed")
            return False
        
        # Step 4: Generate summary
        summary = self.generate_collection_summary(weather_df, pollution_df, merged_df)
        
        print("\n✅ Enhanced Data Collection Pipeline Completed Successfully!")
        print("=" * 60)
        print("📁 Files saved:")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'weather_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'pollution_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'processed', 'merged_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'collection_summary.json')}")
        
        self.logger.info("Enhanced data collection pipeline completed successfully")
        
        return True

def main():
    """Run enhanced data collection pipeline"""
    collector = EnhancedDataCollector()
    success = collector.run_pipeline()
    
    if success:
        print("\n🎉 Phase 1 Data Infrastructure Ready!")
        print("📊 Data validated and stored in repository structure")
        print("🔍 Quality checks completed")
        print("📈 Ready for Phase 2: Feature Engineering Pipeline")
    else:
        print("\n❌ Pipeline failed! Check error messages and logs above.")

if __name__ == "__main__":
    main()
