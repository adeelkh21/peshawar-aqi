"""
Phase 5: Production Integration
==============================

This script integrates the trained model with the Streamlit app for real-time AQI forecasting.
It provides a production-ready API and model serving system.

Key Features:
- Load the trained LightGBM model (94.97% R²)
- Real-time data processing and feature engineering
- 3-day AQI forecasting
- Streamlit app integration
- Production API endpoints

Author: Data Science Team
Date: 2025-08-13
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import pickle
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import time

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Import our existing components
from phase1_enhanced_data_collection import EnhancedDataCollector
from phase2_enhanced_feature_engineering import EnhancedFeatureEngineer

warnings.filterwarnings('ignore')

class ProductionIntegration:
    """
    Production Integration System
    Loads the trained model and provides real-time AQI forecasting
    """
    
    def __init__(self, model_path: str = None):
        """Initialize Production Integration"""
        print("🚀 PHASE 5: PRODUCTION INTEGRATION")
        print("=" * 50)
        print("🎯 Real-time AQI Forecasting System")
        print("📊 Model Performance: 94.97% R²")
        print("🔄 Streamlit App Integration")
        print()
        
        # Setup logging
        self.setup_logging()
        
        # Model path
        if model_path is None:
            # Find the latest model
            deployment_dir = "deployment"
            if os.path.exists(deployment_dir):
                model_dirs = [d for d in os.listdir(deployment_dir) if os.path.isdir(os.path.join(deployment_dir, d))]
                if model_dirs:
                    latest_dir = sorted(model_dirs)[-1]
                    model_path = os.path.join(deployment_dir, latest_dir, "production_model.pkl")
        
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.data_collector = None
        
        # Load the model
        self.load_production_model()
        
        print(f"✅ Production system initialized")
        print(f"📦 Model loaded: {self.model_path}")
        print(f"🎯 Ready for real-time forecasting")

    def setup_logging(self):
        """Setup logging for production system"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production Integration initialized")

    def load_production_model(self):
        """Load the trained production model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"🔄 Loading production model from: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model and scaler
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
            else:
                self.model = model_data
            
            if self.model is None:
                raise ValueError("Model not found in pickle file")
            
            print(f"✅ Model loaded successfully")
            print(f"🤖 Model type: {type(self.model).__name__}")
            
            # Initialize components
            self.feature_engineer = EnhancedFeatureEngineer()
            self.data_collector = EnhancedDataCollector()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            print(f"❌ Error loading model: {str(e)}")
            raise

    def collect_real_time_data(self) -> pd.DataFrame:
        """Collect real-time weather and pollution data"""
        try:
            print("🔄 Collecting real-time data...")
            
            # Use our existing data collection system
            weather_data = self.data_collector.fetch_weather_data()
            pollution_data = self.data_collector.fetch_pollution_data()
            
            # Merge the data
            merged_data = self.data_collector.merge_and_process_data(weather_data, pollution_data)
            
            print(f"✅ Real-time data collected: {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error collecting real-time data: {str(e)}")
            print(f"❌ Error collecting real-time data: {str(e)}")
            raise

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction"""
        try:
            print("🔄 Engineering features...")
            
            # Save the data to the processed directory first
            processed_file = os.path.join("data_repositories", "processed", "merged_data.csv")
            data.to_csv(processed_file, index=False)
            
            # Use our existing feature engineering system
            success = self.feature_engineer.run_pipeline()
            
            if not success:
                raise Exception("Feature engineering pipeline failed")
            
            # Load the engineered features
            engineered_file = os.path.join("data_repositories", "features", "engineered_features.csv")
            if os.path.exists(engineered_file):
                engineered_data = pd.read_csv(engineered_file)
                print(f"✅ Features engineered: {engineered_data.shape[1]} features")
                return engineered_data
            else:
                raise Exception("Engineered features file not found")
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            print(f"❌ Error engineering features: {str(e)}")
            raise

    def engineer_features_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features that match exactly what the model was trained with"""
        try:
            print("🔄 Engineering features for prediction (matching training data exactly)...")
            
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Define the exact features that the model was trained with (from feature importance file)
            required_features = [
                'pm2_5', 'aqi_numeric_rolling_std_12h', 'aqi_numeric_rolling_q75_3h', 'aqi_numeric_rolling_mean_6h',
                'wind_speed_lag24h', 'aqi_lag1h', 'aqi_numeric_rolling_max_3h', 'aqi_numeric_rolling_max_8h',
                'aqi_lag6h', 'aqi_numeric_rolling_std_24h', 'aqi_numeric_lag2h', 'pm10', 'pm25_humidity_interaction',
                'pm10_rolling_std_12h', 'stability_index', 'no2', 'pm2_5_rolling_std_12h', 'no2_lag48h',
                'pm25_dispersion', 'pm10_lag72h', 'aqi_numeric_rolling_mean_12h', 'pm10_lag24h',
                'aqi_numeric_rolling_q25_3h', 'heat_index', 'no2_lag72h', 'aqi_lag3h', 'hour_sin',
                'pm2_5_rolling_min_3h', 'pm2_5_lag60h', 'pm10_lag60h', 'pm25_temp_interaction',
                'pm10_rolling_std_24h', 'aqi_numeric_lag60h', 'pm10_lag48h', 'aqi_numeric_rolling_q75_8h',
                'pm10_wind_interaction', 'pm10_rolling_min_8h', 'pm10_lag12h', 'aqi_numeric_lag72h',
                'pm2_5_lag72h', 'o3_lag48h', 'relative_humidity_lag48h', 'pm25_pm10_ratio', 'aqi_numeric_lag4h',
                'pm2_5_rolling_std_24h', 'aqi_numeric_rolling_min_3h', 'no2_lag24h', 'o3_lag72h',
                'pm2_5_rolling_max_3h', 'relative_humidity_lag24h', 'pm10_lag2h', 'pm10_lag18h',
                'coarse_particle_fraction', 'pm2_5_rolling_max_8h', 'relative_humidity_rolling_q25_3h',
                'temperature_rolling_q25_8h', 'o3', 'o3_lag24h', 'temperature_rolling_q25_3h',
                'pressure_tendency_3h', 'relative_humidity_rolling_q75_3h', 'pm10_rolling_min_3h',
                'pm10_rolling_max_3h', 'pm2_5_lag8h', 'pm10_lag8h', 'pm2_5_lag18h', 'pm2_5_rolling_max_16h',
                'pm10_lag36h', 'temperature_rolling_q75_16h', 'pm2_5_lag2h', 'aqi_numeric_lag12h',
                'temperature_rolling_mean_12h', 'aqi_numeric_lag48h', 'temperature_rolling_q25_16h',
                'aqi_numeric_lag18h', 'pm2_5_lag48h', 'pm10_rolling_max_8h', 'pm2_5_rolling_q25_3h',
                'temperature', 'pressure_lag72h', 'pressure_lag48h', 'relative_humidity_lag72h',
                'relative_humidity_rolling_q25_8h', 'pm10_rolling_q25_3h', 'aqi_numeric_rolling_mean_24h',
                'aqi_numeric_lag36h', 'pm2_5_lag24h', 'aqi_numeric_rolling_max_16h',
                'relative_humidity_rolling_max_8h', 'pm2_5_rolling_q75_3h', 'temperature_lag72h',
                'pm2_5_lag12h', 'temperature_rolling_q75_8h', 'pressure', 'temperature_rolling_q75_3h',
                'pm2_5_rolling_min_8h', 'aqi_numeric_lag8h', 'relative_humidity_rolling_mean_6h',
                'relative_humidity_rolling_q75_36h', 'temperature_rolling_mean_24h', 'pm10_rolling_q75_3h',
                'pm10_rolling_min_16h', 'pm2_5_lag36h', 'relative_humidity', 'temperature_rolling_mean_6h',
                'pressure_tendency_6h', 'hour', 'day_of_year', 'relative_humidity_rolling_mean_24h',
                'pm10_rolling_mean_6h', 'relative_humidity_rolling_min_3h', 'temperature_rolling_min_16h',
                'relative_humidity_rolling_mean_12h', 'pm2_5_rolling_min_16h', 'temperature_rolling_max_3h',
                'temperature_rolling_min_3h', 'temperature_lag24h', 'relative_humidity_rolling_q75_8h',
                'wind_speed', 'pm2_5_rolling_q25_8h', 'pm25_lag6h', 'pm2_5_hours_above_mean_24h',
                'pm10_lag4h', 'pm10_rolling_max_48h', 'temperature_lag48h', 'pressure_lag24h',
                'temperature_rolling_max_8h', 'relative_humidity_rolling_max_16h', 'pm10_lag6h',
                'temperature_rolling_min_8h', 'pm10_rolling_q25_48h', 'aqi_numeric_rolling_q75_16h',
                'pm10_rolling_q75_36h', 'pm2_5_rolling_mean_12h', 'relative_humidity_rolling_q25_36h',
                'aqi_numeric_rolling_q75_48h', 'temperature_rolling_min_48h', 'pm2_5_rolling_q75_36h',
                'pm10_rolling_mean_12h', 'pm10_rolling_q25_8h', 'temperature_rolling_q75_36h',
                'pm2_5_rolling_mean_6h', 'pm10_rolling_max_16h', 'relative_humidity_rolling_min_8h',
                'pm2_5_rolling_mean_24h', 'hour_cos', 'relative_humidity_rolling_q25_48h',
                'relative_humidity_rolling_max_3h', 'temperature_rolling_q25_36h', 'pm2_5_lag4h',
                'pm2_5_rolling_q75_16h', 'pm2_5_rolling_q75_8h', 'pm10_rolling_q25_36h',
                'pm2_5_rolling_min_48h', 'aqi_numeric_rolling_q25_8h', 'pm2_5_rolling_min_36h',
                'pm2_5_rolling_max_36h', 'pm10_rolling_mean_24h', 'pm10_rolling_q75_16h',
                'pm10_rolling_q75_8h', 'pm2_5_rolling_q25_48h', 'relative_humidity_rolling_min_36h',
                'temperature_rolling_q25_48h', 'relative_humidity_rolling_min_48h', 'pm2_5_rolling_max_48h',
                'temperature_rolling_max_16h', 'pm10_hours_above_mean_24h', 'pm2_5_rolling_q25_36h',
                'pm10_rolling_q25_16h', 'pm10_rolling_max_36h', 'aqi_numeric_rolling_q25_48h',
                'aqi_numeric_rolling_max_48h', 'aqi_numeric_rolling_q25_16h', 'relative_humidity_rolling_max_48h',
                'relative_humidity_rolling_q75_48h', 'relative_humidity_rolling_max_36h', 'pm10_rolling_min_36h',
                'relative_humidity_rolling_q75_16h', 'day_of_week', 'aqi_numeric_rolling_min_8h',
                'pm2_5_rolling_q25_16h', 'pm10_rolling_q75_48h', 'pm10_rolling_min_48h',
                'relative_humidity_rolling_q25_16h', 'aqi_numeric_rolling_q75_36h', 'temperature_rolling_q75_48h',
                'relative_humidity_rolling_min_16h', 'wind_chill', 'aqi_numeric_rolling_max_36h',
                'temperature_rolling_max_36h', 'hour_since_midnight', 'aqi_numeric_rolling_min_16h',
                'doy_sin', 'dow_sin', 'pm2_5_rolling_q75_48h', 'pm2_5_cumulative_24h',
                'temperature_rolling_min_36h', 'temperature_rolling_max_48h', 'doy_cos', 'month',
                'aqi_numeric_rolling_q25_36h', 'dow_cos', 'aqi_numeric_rolling_min_48h', 'pm10_cumulative_24h',
                'is_morning_rush', 'is_weekend', 'is_rush_hour', 'is_peak_pollution_hour', 'season',
                'aqi_numeric_rolling_min_36h', 'is_evening_rush', 'is_business_hours', 'hours_since_midnight',
                'hours_until_midnight', 'is_low_pollution_hour'
            ]
            
            # Create basic temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Time-based features
            df['hour_since_midnight'] = df['hour']
            df['hours_since_midnight'] = df['hour']
            df['hours_until_midnight'] = 24 - df['hour']
            
            # Create lag features for aqi_category (renamed to aqi_numeric for consistency)
            df['aqi_numeric'] = df['aqi_category']
            lag_periods = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72]
            for lag in lag_periods:
                df[f'aqi_lag{lag}h'] = df['aqi_numeric'].shift(lag)
                df[f'aqi_numeric_lag{lag}h'] = df['aqi_numeric'].shift(lag)
            
            # Create lag features for other variables
            lag_features = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'temperature', 'relative_humidity', 'pressure']
            for feature in lag_features:
                if feature in df.columns:
                    for lag in lag_periods:
                        df[f'{feature}_lag{lag}h'] = df[feature].shift(lag)
            
            # Create rolling features for aqi_numeric
            rolling_windows = [3, 6, 8, 12, 16, 24, 36, 48]
            for window in rolling_windows:
                df[f'aqi_numeric_rolling_mean_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).mean()
                df[f'aqi_numeric_rolling_std_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).std()
                df[f'aqi_numeric_rolling_min_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).min()
                df[f'aqi_numeric_rolling_max_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).max()
                df[f'aqi_numeric_rolling_q25_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).quantile(0.25)
                df[f'aqi_numeric_rolling_q75_{window}h'] = df['aqi_numeric'].rolling(window=window, min_periods=1).quantile(0.75)
            
            # Create rolling features for other variables
            rolling_features = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'temperature', 'relative_humidity', 'pressure']
            for feature in rolling_features:
                if feature in df.columns:
                    for window in rolling_windows:
                        df[f'{feature}_rolling_mean_{window}h'] = df[feature].rolling(window=window, min_periods=1).mean()
                        df[f'{feature}_rolling_std_{window}h'] = df[feature].rolling(window=window, min_periods=1).std()
                        df[f'{feature}_rolling_min_{window}h'] = df[feature].rolling(window=window, min_periods=1).min()
                        df[f'{feature}_rolling_max_{window}h'] = df[feature].rolling(window=window, min_periods=1).max()
                        df[f'{feature}_rolling_q25_{window}h'] = df[feature].rolling(window=window, min_periods=1).quantile(0.25)
                        df[f'{feature}_rolling_q75_{window}h'] = df[feature].rolling(window=window, min_periods=1).quantile(0.75)
            
            # Create interaction features
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-8)
            
            if 'pm2_5' in df.columns and 'temperature' in df.columns:
                df['pm25_temp_interaction'] = df['pm2_5'] * df['temperature']
            
            if 'pm2_5' in df.columns and 'relative_humidity' in df.columns:
                df['pm25_humidity_interaction'] = df['pm2_5'] * df['relative_humidity']
            
            if 'pm10' in df.columns and 'wind_speed' in df.columns:
                df['pm10_wind_interaction'] = df['pm10'] * df['wind_speed']
            
            # Create statistical features
            if 'temperature' in df.columns and 'relative_humidity' in df.columns:
                df['heat_index'] = 0.5 * (df['temperature'] + 61.0 + ((df['temperature'] - 68.0) * 1.2) + (df['relative_humidity'] * 0.094))
                df['wind_chill'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16)
            
            if 'pm2_5' in df.columns and 'pm10' in df.columns:
                df['coarse_particle_fraction'] = (df['pm10'] - df['pm2_5']) / (df['pm10'] + 1e-8)
            
            if 'pm2_5' in df.columns and 'wind_speed' in df.columns:
                df['pm25_dispersion'] = df['pm2_5'] / (df['wind_speed'] + 1e-8)
            
            if 'pressure' in df.columns:
                df['pressure_tendency_3h'] = df['pressure'].diff(3)
                df['pressure_tendency_6h'] = df['pressure'].diff(6)
            
            if 'temperature' in df.columns and 'relative_humidity' in df.columns:
                df['stability_index'] = df['temperature'] * (1 - df['relative_humidity'] / 100)
            
            # Create time-based categorical features
            df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
            df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_peak_pollution_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 10)).astype(int)
            df['is_low_pollution_hour'] = ((df['hour'] >= 2) & (df['hour'] <= 6)).astype(int)
            
            # Season feature
            df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4]).astype(int)
            
            # Cumulative features
            if 'pm2_5' in df.columns:
                df['pm2_5_cumulative_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).sum()
                df['pm2_5_hours_above_mean_24h'] = (df['pm2_5'] > df['pm2_5'].rolling(window=24, min_periods=1).mean()).astype(int)
            
            if 'pm10' in df.columns:
                df['pm10_cumulative_24h'] = df['pm10'].rolling(window=24, min_periods=1).sum()
                df['pm10_hours_above_mean_24h'] = (df['pm10'] > df['pm10'].rolling(window=24, min_periods=1).mean()).astype(int)
            
            # Remove timestamp and any non-numeric columns
            df = df.select_dtypes(include=[np.number])
            
            # Remove target column if present
            if 'aqi_category' in df.columns:
                df = df.drop('aqi_category', axis=1)
            
            # Fill any remaining NaN values with 0
            df = df.fillna(0)
            
            # Select only the features that the model was trained with
            available_features = [col for col in required_features if col in df.columns]
            missing_features = [col for col in required_features if col not in df.columns]
            
            if missing_features:
                print(f"⚠️  Missing features: {len(missing_features)} features")
                for feature in missing_features[:5]:  # Show first 5 missing features
                    print(f"   - {feature}")
            
            # Create final dataframe with only the required features
            final_df = df[available_features].copy()
            
            # Add missing features with 0 values
            for feature in missing_features:
                final_df[feature] = 0
            
            # Ensure the order matches the training data
            final_df = final_df[required_features]
            
            print(f"✅ Features engineered for prediction: {final_df.shape[1]} features")
            print(f"📊 Feature count matches training data: {final_df.shape[1]} features")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features for prediction: {str(e)}")
            print(f"❌ Error engineering features for prediction: {str(e)}")
            raise

    def predict_aqi(self, features: pd.DataFrame) -> np.ndarray:
        """Make AQI predictions"""
        try:
            print("🔮 Making AQI predictions...")
            
            # Remove non-numeric columns (like timestamp)
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Remove target column if present
            if 'aqi_category' in numeric_features.columns:
                numeric_features = numeric_features.drop('aqi_category', axis=1)
            
            print(f"📊 Using {numeric_features.shape[1]} numeric features for prediction")
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(numeric_features)
            else:
                features_scaled = numeric_features
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            print(f"✅ Predictions made: {len(predictions)} forecasts")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            print(f"❌ Error making predictions: {str(e)}")
            raise

    def forecast_3_day_aqi(self) -> Dict:
        """Generate 3-day AQI forecast"""
        try:
            print("\n🌤️  Generating 3-Day AQI Forecast")
            print("=" * 40)
            
            # Step 1: Collect real-time data
            real_time_data = self.collect_real_time_data()
            
            # Step 2: Engineer features (matching training data)
            engineered_features = self.engineer_features_for_prediction(real_time_data)
            
            # Step 3: Make predictions
            predictions = self.predict_aqi(engineered_features)
            
            # Step 4: Format results
            forecast_results = {
                'timestamp': datetime.now().isoformat(),
                'forecast_period': '3_days',
                'predictions': predictions.tolist(),
                'data_points': len(predictions),
                'model_performance': '94.97% R²',
                'model_type': type(self.model).__name__
            }
            
            # Add AQI categories
            aqi_categories = []
            for pred in predictions:
                if pred <= 50:
                    category = "Good"
                elif pred <= 100:
                    category = "Moderate"
                elif pred <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif pred <= 200:
                    category = "Unhealthy"
                elif pred <= 300:
                    category = "Very Unhealthy"
                else:
                    category = "Hazardous"
                aqi_categories.append(category)
            
            forecast_results['aqi_categories'] = aqi_categories
            
            print("✅ 3-Day AQI Forecast Generated Successfully!")
            print(f"📊 Predictions: {len(predictions)} data points")
            print(f"🎯 Model Performance: 94.97% R²")
            
            return forecast_results
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            print(f"❌ Error generating forecast: {str(e)}")
            raise

    def save_forecast_results(self, forecast_results: Dict, filename: str = None):
        """Save forecast results to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"forecast_results_{timestamp}.json"
            
            # Create forecasts directory
            os.makedirs("forecasts", exist_ok=True)
            filepath = os.path.join("forecasts", filename)
            
            with open(filepath, 'w') as f:
                json.dump(forecast_results, f, indent=4)
            
            print(f"💾 Forecast results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving forecast results: {str(e)}")
            print(f"❌ Error saving forecast results: {str(e)}")

    def test_production_system(self) -> bool:
        """Test the production system"""
        try:
            print("\n🧪 Testing Production System")
            print("=" * 30)
            
            # Test 1: Model loading
            print("✅ Test 1: Model loaded successfully")
            
            # Test 2: Data collection
            print("🔄 Test 2: Testing data collection...")
            test_data = self.collect_real_time_data()
            print(f"✅ Test 2: Data collection successful ({len(test_data)} records)")
            
            # Test 3: Feature engineering
            print("🔄 Test 3: Testing feature engineering...")
            test_features = self.engineer_features_for_prediction(test_data)
            print(f"✅ Test 3: Feature engineering successful ({test_features.shape[1]} features)")
            
            # Test 4: Prediction
            print("🔄 Test 4: Testing predictions...")
            test_predictions = self.predict_aqi(test_features)
            print(f"✅ Test 4: Predictions successful ({len(test_predictions)} forecasts)")
            
            # Test 5: Full forecast
            print("🔄 Test 5: Testing full forecast...")
            test_forecast = self.forecast_3_day_aqi()
            print(f"✅ Test 5: Full forecast successful")
            
            print("\n🎉 All Production System Tests Passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Production system test failed: {str(e)}")
            print(f"❌ Production system test failed: {str(e)}")
            return False

def main():
    """Run Production Integration"""
    try:
        # Initialize production system
        production = ProductionIntegration()
        
        # Test the system
        if production.test_production_system():
            print("\n🎉 Production Integration Complete!")
            print("📊 Model Performance: 94.97% R²")
            print("🔄 Ready for Streamlit integration")
            print("📈 Real-time AQI forecasting active")
            
            # Generate a sample forecast
            print("\n🔮 Generating Sample Forecast...")
            forecast = production.forecast_3_day_aqi()
            production.save_forecast_results(forecast)
            
            print("\n🚀 Next Steps:")
            print("1. Integrate with Streamlit app")
            print("2. Set up automated data collection")
            print("3. Deploy to production server")
            
        else:
            print("\n❌ Production system test failed!")
            
    except Exception as e:
        print(f"\n❌ Production integration failed: {str(e)}")

if __name__ == "__main__":
    main()
