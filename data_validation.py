"""
Data Validation and Quality Assurance Module
===========================================

This module provides comprehensive data validation for the AQI prediction pipeline:
- Data quality checks
- Schema validation
- Data freshness monitoring
- Error handling and reporting

Author: Data Science Team
Date: 2024-12-08
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class DataValidator:
    """Comprehensive data validation for AQI prediction pipeline"""
    
    def __init__(self, data_dir: str = "data_repositories"):
        """Initialize data validator"""
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Define expected schemas
        self.weather_schema = {
            'timestamp': 'datetime64[ns]',
            'temperature': 'float64',
            'dew_point': 'float64',
            'relative_humidity': 'float64',
            'precipitation': 'float64',
            'wind_direction': 'float64',
            'wind_speed': 'float64',
            'pressure': 'float64'
        }
        
        self.pollution_schema = {
            'timestamp': 'datetime64[ns]',
            'aqi_category': 'int64',
            'co': 'float64',
            'no': 'float64',
            'no2': 'float64',
            'o3': 'float64',
            'so2': 'float64',
            'pm2_5': 'float64',
            'pm10': 'float64',
            'nh3': 'float64'
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_records': 20,  # Minimum records per hour
            'max_missing_pct': 0.3,  # Maximum 30% missing values
            'max_age_hours': 2,  # Data should not be older than 2 hours
            'valid_aqi_range': (1, 5),  # Valid AQI categories
            'valid_temp_range': (-50, 60),  # Valid temperature range (°C)
            'valid_humidity_range': (0, 100),  # Valid humidity range (%)
        }

    def validate_weather_data(self, df: pd.DataFrame) -> Dict:
        """Validate weather data quality and schema"""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Schema validation
            for col, expected_type in self.weather_schema.items():
                if col not in df.columns:
                    validation_report['errors'].append(f"Missing column: {col}")
                    validation_report['valid'] = False
                elif str(df[col].dtype) != expected_type:
                    validation_report['warnings'].append(
                        f"Column {col} has type {df[col].dtype}, expected {expected_type}"
                    )
            
            # Data quality checks
            if len(df) < self.quality_thresholds['min_records']:
                validation_report['errors'].append(
                    f"Insufficient records: {len(df)} < {self.quality_thresholds['min_records']}"
                )
                validation_report['valid'] = False
            
            # Missing values check
            missing_pct = df.isnull().sum() / len(df)
            high_missing = missing_pct[missing_pct > self.quality_thresholds['max_missing_pct']]
            if not high_missing.empty:
                validation_report['warnings'].append(
                    f"High missing values: {high_missing.to_dict()}"
                )
            
            # Data freshness check
            if 'timestamp' in df.columns:
                latest_time = df['timestamp'].max()
                age_hours = (datetime.now() - latest_time).total_seconds() / 3600
                if age_hours > self.quality_thresholds['max_age_hours']:
                    validation_report['warnings'].append(
                        f"Data is {age_hours:.1f} hours old (max: {self.quality_thresholds['max_age_hours']})"
                    )
            
            # Value range checks
            if 'temperature' in df.columns:
                temp_range = (df['temperature'].min(), df['temperature'].max())
                if not (self.quality_thresholds['valid_temp_range'][0] <= temp_range[0] <= temp_range[1] <= self.quality_thresholds['valid_temp_range'][1]):
                    validation_report['warnings'].append(
                        f"Temperature out of expected range: {temp_range}"
                    )
            
            if 'relative_humidity' in df.columns:
                humidity_range = (df['relative_humidity'].min(), df['relative_humidity'].max())
                if not (self.quality_thresholds['valid_humidity_range'][0] <= humidity_range[0] <= humidity_range[1] <= self.quality_thresholds['valid_humidity_range'][1]):
                    validation_report['warnings'].append(
                        f"Humidity out of expected range: {humidity_range}"
                    )
            
            # Statistics
            validation_report['statistics'] = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'data_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                }
            }
            
        except Exception as e:
            validation_report['errors'].append(f"Validation error: {str(e)}")
            validation_report['valid'] = False
        
        return validation_report

    def validate_pollution_data(self, df: pd.DataFrame) -> Dict:
        """Validate pollution data quality and schema"""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Schema validation
            for col, expected_type in self.pollution_schema.items():
                if col not in df.columns:
                    validation_report['errors'].append(f"Missing column: {col}")
                    validation_report['valid'] = False
                elif str(df[col].dtype) != expected_type:
                    validation_report['warnings'].append(
                        f"Column {col} has type {df[col].dtype}, expected {expected_type}"
                    )
            
            # Data quality checks
            if len(df) < self.quality_thresholds['min_records']:
                validation_report['errors'].append(
                    f"Insufficient records: {len(df)} < {self.quality_thresholds['min_records']}"
                )
                validation_report['valid'] = False
            
            # Missing values check
            missing_pct = df.isnull().sum() / len(df)
            high_missing = missing_pct[missing_pct > self.quality_thresholds['max_missing_pct']]
            if not high_missing.empty:
                validation_report['warnings'].append(
                    f"High missing values: {high_missing.to_dict()}"
                )
            
            # AQI category validation
            if 'aqi_category' in df.columns:
                invalid_aqi = df[~df['aqi_category'].isin(range(1, 6))]
                if not invalid_aqi.empty:
                    validation_report['warnings'].append(
                        f"Invalid AQI categories found: {invalid_aqi['aqi_category'].unique()}"
                    )
            
            # Data freshness check
            if 'timestamp' in df.columns:
                latest_time = df['timestamp'].max()
                age_hours = (datetime.now() - latest_time).total_seconds() / 3600
                if age_hours > self.quality_thresholds['max_age_hours']:
                    validation_report['warnings'].append(
                        f"Data is {age_hours:.1f} hours old (max: {self.quality_thresholds['max_age_hours']})"
                    )
            
            # Statistics
            validation_report['statistics'] = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'aqi_distribution': df['aqi_category'].value_counts().to_dict() if 'aqi_category' in df.columns else {},
                'data_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                }
            }
            
        except Exception as e:
            validation_report['errors'].append(f"Validation error: {str(e)}")
            validation_report['valid'] = False
        
        return validation_report

    def validate_merged_data(self, df: pd.DataFrame) -> Dict:
        """Validate merged dataset quality"""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check for required columns
            required_columns = ['timestamp', 'aqi_category', 'temperature', 'relative_humidity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_report['errors'].append(f"Missing required columns: {missing_columns}")
                validation_report['valid'] = False
            
            # Check data completeness
            if len(df) < self.quality_thresholds['min_records']:
                validation_report['errors'].append(
                    f"Insufficient records: {len(df)} < {self.quality_thresholds['min_records']}"
                )
                validation_report['valid'] = False
            
            # Check for duplicates
            duplicates = df.duplicated(subset=['timestamp']).sum()
            if duplicates > 0:
                validation_report['warnings'].append(f"Found {duplicates} duplicate timestamps")
            
            # Check time continuity
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_diff = df_sorted['timestamp'].diff().dropna()
                if not time_diff.empty:
                    avg_interval = time_diff.mean()
                    if avg_interval > timedelta(hours=2):
                        validation_report['warnings'].append(
                            f"Large time intervals detected: {avg_interval}"
                        )
            
            # Statistics
            validation_report['statistics'] = {
                'total_records': len(df),
                'features': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                }
            }
            
        except Exception as e:
            validation_report['errors'].append(f"Validation error: {str(e)}")
            validation_report['valid'] = False
        
        return validation_report

    def save_validation_report(self, report: Dict, filepath: str):
        """Save validation report to file"""
        try:
            report['timestamp'] = datetime.now().isoformat()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            self.logger.info(f"Validation report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving validation report: {str(e)}")

    def check_data_freshness(self, data_path: str) -> Dict:
        """Check if data is fresh and up-to-date"""
        freshness_report = {
            'fresh': True,
            'last_update': None,
            'age_hours': None,
            'status': 'unknown'
        }
        
        try:
            if os.path.exists(data_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(data_path))
                age_hours = (datetime.now() - file_time).total_seconds() / 3600
                
                freshness_report['last_update'] = file_time.isoformat()
                freshness_report['age_hours'] = age_hours
                
                if age_hours <= self.quality_thresholds['max_age_hours']:
                    freshness_report['status'] = 'fresh'
                elif age_hours <= 24:
                    freshness_report['status'] = 'stale'
                    freshness_report['fresh'] = False
                else:
                    freshness_report['status'] = 'outdated'
                    freshness_report['fresh'] = False
            else:
                freshness_report['status'] = 'missing'
                freshness_report['fresh'] = False
                
        except Exception as e:
            freshness_report['status'] = 'error'
            freshness_report['fresh'] = False
            self.logger.error(f"Error checking data freshness: {str(e)}")
        
        return freshness_report

    def generate_quality_summary(self, validation_reports: Dict) -> Dict:
        """Generate overall quality summary from multiple validation reports"""
        summary = {
            'overall_status': 'pass',
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': 0,
            'details': {}
        }
        
        for name, report in validation_reports.items():
            summary['total_checks'] += 1
            summary['details'][name] = {
                'status': 'pass' if report.get('valid', False) else 'fail',
                'errors': len(report.get('errors', [])),
                'warnings': len(report.get('warnings', []))
            }
            
            if report.get('valid', False):
                summary['passed_checks'] += 1
            else:
                summary['failed_checks'] += 1
                summary['overall_status'] = 'fail'
            
            summary['warnings'] += len(report.get('warnings', []))
        
        return summary
