"""
AQI Prediction System - Data Merger
=================================

This script:
1. Merges all collected data (historical + hourly updates)
2. Removes duplicates
3. Maintains a single consolidated dataset
4. Updates the main data files with new data
"""

import os
import pandas as pd
import glob
from datetime import datetime
import json
import shutil

def merge_all_data():
    """Merge all data and maintain a single consolidated dataset"""
    print("🔄 Starting Data Merger")
    print("=" * 50)
    
    # Create main data directories if they don't exist
    main_dirs = ['data/raw', 'data/processed', 'data/metadata']
    for dir_path in main_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        # Find all weather and pollution data files
        weather_files = (
            glob.glob('data/historical/*/raw/historical_weather.csv') +
            glob.glob('data/v1_*/raw/weather_data.csv')
        )
        pollution_files = (
            glob.glob('data/historical/*/raw/historical_pollution.csv') +
            glob.glob('data/v1_*/raw/pollution_data.csv')
        )
        
        print(f"\n📂 Found {len(weather_files)} weather files and {len(pollution_files)} pollution files")
        
        # Merge weather data
        weather_dfs = []
        for file in weather_files:
            df = pd.read_csv(file)
            weather_dfs.append(df)
        
        if weather_dfs:
            weather_data = pd.concat(weather_dfs, ignore_index=True)
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            weather_data = weather_data.drop_duplicates(subset=['timestamp'])
            weather_data = weather_data.sort_values('timestamp')
            
            # Save consolidated weather data
            weather_data.to_csv('data/raw/weather_data.csv', index=False)
            print(f"✅ Weather data merged: {len(weather_data):,} records")
        
        # Merge pollution data
        pollution_dfs = []
        for file in pollution_files:
            df = pd.read_csv(file)
            pollution_dfs.append(df)
        
        if pollution_dfs:
            pollution_data = pd.concat(pollution_dfs, ignore_index=True)
            pollution_data['timestamp'] = pd.to_datetime(pollution_data['timestamp'])
            pollution_data = pollution_data.drop_duplicates(subset=['timestamp'])
            pollution_data = pollution_data.sort_values('timestamp')
            
            # Save consolidated pollution data
            pollution_data.to_csv('data/raw/pollution_data.csv', index=False)
            print(f"✅ Pollution data merged: {len(pollution_data):,} records")
        
        # Merge and process final dataset
        if weather_dfs and pollution_dfs:
            # Ensure timestamps are datetime
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            pollution_data['timestamp'] = pd.to_datetime(pollution_data['timestamp'])
            
            # Round timestamps to nearest hour
            weather_data['timestamp'] = weather_data['timestamp'].dt.floor('H')
            pollution_data['timestamp'] = pollution_data['timestamp'].dt.floor('H')
            
            # Merge on timestamp
            merged_data = pd.merge(
                pollution_data,
                weather_data,
                on='timestamp',
                how='inner'
            )
            
            # Add time-based features
            merged_data['hour'] = merged_data['timestamp'].dt.hour
            merged_data['day'] = merged_data['timestamp'].dt.day
            merged_data['month'] = merged_data['timestamp'].dt.month
            merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
            merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
            
            # Save processed data
            merged_data.to_csv('data/processed/merged_data.csv', index=False)
            
            # Save metadata
            metadata = {
                "last_update": datetime.now().isoformat(),
                "total_records": len(merged_data),
                "date_range": {
                    "start": merged_data['timestamp'].min(),
                    "end": merged_data['timestamp'].max()
                },
                "features": list(merged_data.columns),
                "data_sources": {
                    "weather_files": len(weather_files),
                    "pollution_files": len(pollution_files)
                },
                "missing_values": merged_data.isnull().sum().to_dict()
            }
            
            with open('data/metadata/dataset_info.json', 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print("\n📊 Final Dataset Summary:")
            print(f"Total records: {len(merged_data):,}")
            print(f"Date range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
            print(f"Features: {len(merged_data.columns)}")
            
            # Clean up old versioned directories
            cleanup_old_data()
            
            return True
            
    except Exception as e:
        print(f"❌ Error merging data: {str(e)}")
        return False

def cleanup_old_data():
    """Remove old versioned directories after successful merge"""
    try:
        # Remove historical data directories
        historical_dirs = glob.glob('data/historical/*/')
        for dir_path in historical_dirs:
            shutil.rmtree(dir_path)
        
        # Remove versioned directories
        version_dirs = glob.glob('data/v1_*/')
        for dir_path in version_dirs:
            shutil.rmtree(dir_path)
        
        print("\n🧹 Cleaned up old data directories")
        
    except Exception as e:
        print(f"⚠️ Warning: Error during cleanup: {str(e)}")

def main():
    """Run data merger"""
    success = merge_all_data()
    
    if success:
        print("\n✅ Data merger completed successfully!")
        print("📁 Final data structure:")
        print("   data/")
        print("   ├── raw/")
        print("   │   ├── weather_data.csv")
        print("   │   └── pollution_data.csv")
        print("   ├── processed/")
        print("   │   └── merged_data.csv")
        print("   └── metadata/")
        print("       └── dataset_info.json")
    else:
        print("\n❌ Data merger failed! Check error messages above.")

if __name__ == "__main__":
    main()
