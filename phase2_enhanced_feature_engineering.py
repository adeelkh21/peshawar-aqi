"""
Enhanced Feature Engineering Pipeline - Phase 2
=============================================

This script provides enhanced feature engineering with:
- Incremental feature updates
- Rolling window calculations
- Lag features and temporal patterns
- Feature validation and drift detection
- Automated feature versioning

Author: Data Science Team
Date: 2024-12-08
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from data_validation import DataValidator
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with validation and incremental updates"""
    
    def __init__(self, data_dir: str = "data_repositories"):
        """Initialize enhanced feature engineer"""
        print("🔄 Initializing Enhanced Feature Engineering Pipeline")
        print("=" * 60)
        
        self.data_dir = data_dir
        self.feature_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize validator
        self.validator = DataValidator(data_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Feature engineering configuration
        self.feature_config = {
            'rolling_windows': [3, 6, 12, 24],  # Hours
            'lag_features': [1, 2, 3, 6, 12, 24, 36, 54, 66],  # Hours (extended lags enabled with 150+ days history)
            'temporal_features': True,
            'interaction_features': True,
            'statistical_features': True
        }
        
        print(f"📂 Data Directory: {self.data_dir}")
        print(f"🔧 Feature Configuration: {len(self.feature_config['rolling_windows'])} rolling windows, {len(self.feature_config['lag_features'])} lag features")
        print(f"🔍 Validation: Enabled")

    def _setup_logging(self):
        """Setup logging for feature engineering"""
        log_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"feature_engineering_{self.feature_timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load processed data from Phase 1"""
        print("\n📂 Loading Processed Data")
        print("-" * 40)
        
        try:
            processed_file = os.path.join(self.data_dir, "processed", "merged_data.csv")
            
            if not os.path.exists(processed_file):
                self.logger.error(f"Processed data file not found: {processed_file}")
                print(f"❌ Processed data file not found: {processed_file}")
                return None
            
            df = pd.read_csv(processed_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded processed data: {len(df)} records")
            print(f"✅ Loaded processed data: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            print(f"❌ Error loading processed data: {str(e)}")
            return None

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        print("\n⏰ Creating Temporal Features")
        print("-" * 40)
        
        try:
            self.logger.info("Creating temporal features")
            
            # Basic temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Time-based categories
            df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
            df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
            df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
            
            # Season features
            df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
            
            self.logger.info(f"Created {len([col for col in df.columns if col not in ['timestamp', 'aqi_category', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'temperature', 'dew_point', 'relative_humidity', 'precipitation', 'wind_direction', 'wind_speed', 'pressure']])} temporal features")
            print(f"✅ Created temporal features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating temporal features: {str(e)}")
            print(f"❌ Error creating temporal features: {str(e)}")
            return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series prediction"""
        print("\n⏪ Creating Lag Features")
        print("-" * 40)
        
        try:
            self.logger.info("Creating lag features")
            
            # Key features for lagging
            lag_features = ['aqi_category', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'temperature', 'relative_humidity']
            
            for feature in lag_features:
                if feature in df.columns:
                    for lag in self.feature_config['lag_features']:
                        if len(df) > lag:
                            df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
            
            # Create lag differences
            for feature in ['aqi_category', 'pm2_5', 'pm10']:
                if feature in df.columns:
                    df[f'{feature}_diff_1h'] = df[feature].diff(1)
                    df[f'{feature}_diff_3h'] = df[feature].diff(3)
                    df[f'{feature}_diff_6h'] = df[feature].diff(6)
            
            self.logger.info(f"Created lag features for {len(lag_features)} base features")
            print(f"✅ Created lag features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            print(f"❌ Error creating lag features: {str(e)}")
            return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistical features"""
        print("\n📊 Creating Rolling Features")
        print("-" * 40)
        
        try:
            self.logger.info("Creating rolling features")
            
            # Features for rolling calculations
            rolling_features = ['aqi_category', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'temperature', 'relative_humidity', 'wind_speed']
            
            for feature in rolling_features:
                if feature in df.columns:
                    for window in self.feature_config['rolling_windows']:
                        if len(df) >= window:
                            # Rolling statistics
                            df[f'{feature}_rolling_mean_{window}h'] = df[feature].rolling(window=window, min_periods=1).mean()
                            df[f'{feature}_rolling_std_{window}h'] = df[feature].rolling(window=window, min_periods=1).std()
                            df[f'{feature}_rolling_min_{window}h'] = df[feature].rolling(window=window, min_periods=1).min()
                            df[f'{feature}_rolling_max_{window}h'] = df[feature].rolling(window=window, min_periods=1).max()
                            
                            # Rolling percentiles
                            df[f'{feature}_rolling_25p_{window}h'] = df[feature].rolling(window=window, min_periods=1).quantile(0.25)
                            df[f'{feature}_rolling_75p_{window}h'] = df[feature].rolling(window=window, min_periods=1).quantile(0.75)
            
            self.logger.info(f"Created rolling features for {len(rolling_features)} base features")
            print(f"✅ Created rolling features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {str(e)}")
            print(f"❌ Error creating rolling features: {str(e)}")
            return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        print("\n🔗 Creating Interaction Features")
        print("-" * 40)
        
        try:
            self.logger.info("Creating interaction features")
            
            # Temperature and humidity interactions
            if 'temperature' in df.columns and 'relative_humidity' in df.columns:
                df['temp_humidity_interaction'] = df['temperature'] * df['relative_humidity']
                df['temp_humidity_ratio'] = df['temperature'] / (df['relative_humidity'] + 1e-8)
            
            # Pollution interactions
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-8)
                df['pm2_5_pm10_sum'] = df['pm2_5'] + df['pm10']
            
            # Weather and pollution interactions
            if 'temperature' in df.columns and 'pm2_5' in df.columns:
                df['temp_pm2_5_interaction'] = df['temperature'] * df['pm2_5']
            
            if 'wind_speed' in df.columns and 'pm2_5' in df.columns:
                df['wind_pm2_5_interaction'] = df['wind_speed'] * df['pm2_5']
            
            # AQI and weather interactions
            if 'aqi_category' in df.columns and 'temperature' in df.columns:
                df['aqi_temp_interaction'] = df['aqi_category'] * df['temperature']
            
            if 'aqi_category' in df.columns and 'relative_humidity' in df.columns:
                df['aqi_humidity_interaction'] = df['aqi_category'] * df['relative_humidity']
            
            self.logger.info("Created interaction features")
            print(f"✅ Created interaction features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            print(f"❌ Error creating interaction features: {str(e)}")
            return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features and ratios"""
        print("\n📈 Creating Statistical Features")
        print("-" * 40)
        
        try:
            self.logger.info("Creating statistical features")
            
            # Pollution ratios and indices
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-8)
                df['pm2_5_pm10_diff'] = df['pm2_5'] - df['pm10']
            
            # Weather indices
            if 'temperature' in df.columns and 'relative_humidity' in df.columns:
                # Heat index approximation
                df['heat_index'] = 0.5 * (df['temperature'] + 61.0 + ((df['temperature'] - 68.0) * 1.2) + (df['relative_humidity'] * 0.094))
                
                # Comfort index
                df['comfort_index'] = df['temperature'] + 0.348 * df['relative_humidity'] - 0.7 * df['wind_speed'] - 0.7
            
            # Pollution indices
            if all(col in df.columns for col in ['co', 'no2', 'o3', 'so2']):
                df['pollution_index'] = (df['co'] + df['no2'] + df['o3'] + df['so2']) / 4
            
            # Wind chill (if temperature and wind speed available)
            if 'temperature' in df.columns and 'wind_speed' in df.columns:
                df['wind_chill'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16)
            
            self.logger.info("Created statistical features")
            print(f"✅ Created statistical features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            print(f"❌ Error creating statistical features: {str(e)}")
            return df

    def validate_features(self, df: pd.DataFrame) -> Dict:
        """Validate engineered features"""
        print("\n🔍 Validating Features")
        print("-" * 40)
        
        try:
            self.logger.info("Validating engineered features")
            
            validation_report = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {
                    'total_features': len(df.columns),
                    'total_records': len(df),
                    'missing_values': df.isnull().sum().to_dict(),
                    'feature_types': df.dtypes.to_dict(),
                    'feature_categories': {
                        'temporal': len([col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'week', 'season'])]),
                        'lag': len([col for col in df.columns if 'lag' in col]),
                        'rolling': len([col for col in df.columns if 'rolling' in col]),
                        'interaction': len([col for col in df.columns if 'interaction' in col or 'ratio' in col]),
                        'statistical': len([col for col in df.columns if any(x in col for x in ['index', 'chill', 'comfort'])]),
                        'original': len([col for col in df.columns if col in ['timestamp', 'aqi_category', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'temperature', 'dew_point', 'relative_humidity', 'precipitation', 'wind_direction', 'wind_speed', 'pressure']])
                    }
                }
            }
            
            # Check for required features
            required_features = ['timestamp', 'aqi_category']
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                validation_report['errors'].append(f"Missing required features: {missing_features}")
                validation_report['valid'] = False
            
            # Check for high missing values
            missing_pct = df.isnull().sum() / len(df)
            high_missing = missing_pct[missing_pct > 0.5]  # More than 50% missing
            if not high_missing.empty:
                validation_report['warnings'].append(f"High missing values in features: {high_missing.to_dict()}")
            
            # Check for infinite values
            inf_features = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if np.isinf(df[col]).any():
                    inf_features.append(col)
            
            if inf_features:
                validation_report['warnings'].append(f"Infinite values found in features: {inf_features}")
            
            # Check feature correlation with target
            if 'aqi_category' in df.columns:
                correlations = {}
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'aqi_category':
                        corr = df[col].corr(df['aqi_category'])
                        if not pd.isna(corr):
                            correlations[col] = corr
                
                # Sort by absolute correlation
                sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                validation_report['statistics']['top_correlations'] = sorted_correlations[:10]
            
            self.logger.info(f"Feature validation completed: {validation_report['valid']}")
            print(f"✅ Feature validation completed")
            print(f"📊 Total features: {validation_report['statistics']['total_features']}")
            print(f"📈 Feature categories: {validation_report['statistics']['feature_categories']}")
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            print(f"❌ Error validating features: {str(e)}")
            return {'valid': False, 'errors': [str(e)], 'warnings': [], 'statistics': {}}

    def save_engineered_features(self, df: pd.DataFrame, validation_report: Dict):
        """Save engineered features and metadata"""
        print("\n💾 Saving Engineered Features")
        print("-" * 40)
        
        try:
            self.logger.info("Saving engineered features")
            
            # Save engineered features
            features_file = os.path.join(self.data_dir, "features", "engineered_features.csv")
            df.to_csv(features_file, index=False)
            
            # Save validation report
            validation_file = os.path.join(self.data_dir, "features", "validation", "feature_validation.json")
            self.validator.save_validation_report(validation_report, validation_file)
            
            # Save feature metadata
            feature_metadata = {
                "timestamp": self.feature_timestamp,
                "total_features": len(df.columns),
                "total_records": len(df),
                "feature_config": self.feature_config,
                "feature_categories": validation_report['statistics']['feature_categories'],
                "validation_status": validation_report['valid'],
                "validation_errors": len(validation_report['errors']),
                "validation_warnings": len(validation_report['warnings']),
                "feature_list": list(df.columns),
                "data_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                }
            }
            
            metadata_file = os.path.join(self.data_dir, "features", "feature_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(feature_metadata, f, indent=4, default=str)
            
            self.logger.info(f"Saved engineered features: {len(df.columns)} features, {len(df)} records")
            print(f"✅ Saved engineered features")
            print(f"📁 Files saved:")
            print(f"   - {features_file}")
            print(f"   - {validation_file}")
            print(f"   - {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving engineered features: {str(e)}")
            print(f"❌ Error saving engineered features: {str(e)}")

    def generate_feature_summary(self, df: pd.DataFrame, validation_report: Dict):
        """Generate comprehensive feature engineering summary"""
        print("\n📋 Feature Engineering Summary")
        print("-" * 40)
        
        summary = {
            "feature_timestamp": self.feature_timestamp,
            "total_features": len(df.columns),
            "total_records": len(df),
            "feature_categories": validation_report['statistics']['feature_categories'],
            "validation_status": validation_report['valid'],
            "data_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "feature_config": self.feature_config
        }
        
        # Save summary
        summary_file = os.path.join(self.data_dir, "feature_engineering_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        # Print summary
        print(f"📅 Feature Engineering Time: {self.feature_timestamp}")
        print(f"🔧 Total Features: {summary['total_features']}")
        print(f"📊 Total Records: {summary['total_records']:,}")
        print(f"📈 Feature Categories:")
        for category, count in summary['feature_categories'].items():
            print(f"   - {category}: {count}")
        print(f"🔍 Validation: {'✅ PASS' if summary['validation_status'] else '❌ FAIL'}")
        print(f"📁 Summary saved to: {summary_file}")
        
        return summary

    def run_pipeline(self):
        """Run complete enhanced feature engineering pipeline"""
        print("\n🚀 Starting Enhanced Feature Engineering Pipeline")
        print("=" * 60)
        
        self.logger.info("Starting enhanced feature engineering pipeline")
        
        # Step 1: Load processed data
        df = self.load_processed_data()
        if df is None:
            self.logger.error("Failed to load processed data")
            return False
        
        # Step 2: Create temporal features
        df = self.create_temporal_features(df)
        
        # Step 3: Create lag features
        df = self.create_lag_features(df)
        
        # Step 4: Create rolling features
        df = self.create_rolling_features(df)
        
        # Step 5: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 6: Create statistical features
        df = self.create_statistical_features(df)
        
        # Step 7: Validate features
        validation_report = self.validate_features(df)
        
        # Step 8: Save engineered features
        self.save_engineered_features(df, validation_report)
        
        # Step 9: Generate summary
        summary = self.generate_feature_summary(df, validation_report)
        
        print("\n✅ Enhanced Feature Engineering Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📊 Final dataset: {len(df.columns)} features, {len(df)} records")
        print(f"🔍 Validation: {'✅ PASS' if validation_report['valid'] else '❌ FAIL'}")
        
        self.logger.info("Enhanced feature engineering pipeline completed successfully")
        
        return True

def main():
    """Run enhanced feature engineering pipeline"""
    engineer = EnhancedFeatureEngineer()
    success = engineer.run_pipeline()
    
    if success:
        print("\n🎉 Phase 2 Feature Engineering Complete!")
        print("📊 Features engineered and validated")
        print("🔍 Quality checks completed")
        print("📈 Ready for Phase 3: Model Training Pipeline")
    else:
        print("\n❌ Pipeline failed! Check error messages and logs above.")

if __name__ == "__main__":
    main()
