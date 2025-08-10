"""
Data Validation Module for AQI Prediction System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger('validation')

class DataValidator:
    def __init__(self):
        """Initialize data validator with expected schemas and value ranges"""
        # Expected columns and their types
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
        
        # Value ranges for validation
        self.value_ranges = {
            'temperature': (-30, 60),  # °C
            'relative_humidity': (0, 100),  # %
            'wind_speed': (0, 100),  # m/s
            'pressure': (800, 1200),  # hPa
            'pm2_5': (0, 1000),  # μg/m³
            'pm10': (0, 1000),  # μg/m³
            'aqi_numeric': (0, 500)  # AQI scale
        }

    def validate_api_response(self, response: dict, api_type: str) -> Tuple[bool, List[str]]:
        """
        Validate API response structure and basic content
        
        Args:
            response (dict): API response data
            api_type (str): Type of API ('weather' or 'pollution')
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        errors = []
        
        if api_type == 'weather':
            required_fields = ['time', 'temp', 'rhum', 'wspd', 'pres']
            for field in required_fields:
                if field not in response:
                    errors.append(f"Missing required field: {field}")
                    
        elif api_type == 'pollution':
            if 'list' not in response:
                errors.append("Missing 'list' in pollution data")
            else:
                for item in response['list']:
                    if 'main' not in item or 'components' not in item:
                        errors.append("Invalid pollution data structure")
                        break
        
        return len(errors) == 0, errors

    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> Tuple[bool, Dict]:
        """
        Validate DataFrame structure and content
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            data_type (str): Type of data ('weather' or 'pollution')
            
        Returns:
            Tuple[bool, Dict]: (is_valid, validation report)
        """
        report = {
            'missing_values': {},
            'invalid_types': [],
            'out_of_range': {},
            'duplicates': 0,
            'total_records': len(df)
        }
        
        # Check schema
        schema = self.weather_schema if data_type == 'weather' else self.pollution_schema
        for col, expected_type in schema.items():
            if col not in df.columns:
                report['invalid_types'].append(f"Missing column: {col}")
            elif str(df[col].dtype) != expected_type:
                report['invalid_types'].append(f"Invalid type for {col}: expected {expected_type}, got {df[col].dtype}")
        
        # Check missing values
        report['missing_values'] = df.isnull().sum().to_dict()
        
        # Check value ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid_count > 0:
                    report['out_of_range'][col] = invalid_count
        
        # Check duplicates
        report['duplicates'] = df.duplicated().sum()
        
        # Log validation results
        is_valid = (
            len(report['invalid_types']) == 0 and
            sum(report['missing_values'].values()) == 0 and
            len(report['out_of_range']) == 0 and
            report['duplicates'] == 0
        )
        
        if not is_valid:
            logger.warning(f"Validation failed for {data_type} data")
            for category, issues in report.items():
                if issues:  # If there are any issues in this category
                    logger.warning(f"{category}: {issues}")
        else:
            logger.info(f"Validation passed for {data_type} data")
        
        return is_valid, report

    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate merged dataset
        
        Args:
            df (pd.DataFrame): Merged DataFrame to validate
            
        Returns:
            Tuple[bool, Dict]: (is_valid, validation report)
        """
        report = {
            'completeness': {},
            'consistency': {},
            'quality_metrics': {}
        }
        
        # Check data completeness
        total_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        expected_records = int(total_hours) + 1
        actual_records = len(df)
        
        report['completeness'] = {
            'expected_records': expected_records,
            'actual_records': actual_records,
            'coverage_ratio': actual_records / expected_records
        }
        
        # Check data consistency
        report['consistency'] = {
            'timestamp_gaps': self._check_timestamp_gaps(df),
            'value_consistency': self._check_value_consistency(df)
        }
        
        # Calculate quality metrics
        report['quality_metrics'] = {
            'missing_rate': df.isnull().mean().to_dict(),
            'unique_values': df.nunique().to_dict()
        }
        
        # Define optional fields that can be missing
        optional_fields = {'snow', 'wpgt', 'tsun'}
        
        # Check missing rates excluding optional fields
        missing_rates = {k: v for k, v in report['quality_metrics']['missing_rate'].items() 
                        if k not in optional_fields}
        
        # Determine if data is valid
        is_valid = (
            # Data completeness checks
            report['completeness']['coverage_ratio'] >= 0.95 and  # At least 95% coverage
            report['consistency']['timestamp_gaps']['max_gap_hours'] <= 3 and  # No gaps > 3 hours
            
            # Missing value checks (excluding optional fields)
            all(v <= 0.05 for k, v in missing_rates.items()) and  # Max 5% missing values
            
            # Core field checks
            report['quality_metrics']['missing_rate']['aqi_numeric'] == 0 and  # No missing AQI
            report['quality_metrics']['missing_rate']['temperature'] <= 0.1 and  # Max 10% missing temp
            report['quality_metrics']['missing_rate']['relative_humidity'] <= 0.1 and  # Max 10% missing humidity
            
            # Data consistency checks
            report['consistency']['value_consistency']['aqi_consistency'] and  # AQI calculations consistent
            len(report['quality_metrics']['unique_values']) >= 20  # At least 20 unique values
        )
        
        # Generate validation status details
        validation_status = {
            'completeness': report['completeness']['coverage_ratio'] >= 0.95,
            'timestamp_continuity': report['consistency']['timestamp_gaps']['max_gap_hours'] <= 3,
            'missing_values': all(v <= 0.05 for k, v in missing_rates.items()),
            'aqi_complete': report['quality_metrics']['missing_rate']['aqi_numeric'] == 0,
            'temperature_complete': report['quality_metrics']['missing_rate']['temperature'] <= 0.1,
            'humidity_complete': report['quality_metrics']['missing_rate']['relative_humidity'] <= 0.1,
            'aqi_consistent': report['consistency']['value_consistency']['aqi_consistency'],
            'sufficient_variation': len(report['quality_metrics']['unique_values']) >= 20
        }
        
        if not is_valid:
            logger.warning("Merged data validation failed")
            logger.warning("Validation Status:")
            for check, passed in validation_status.items():
                logger.warning(f"  {check}: {'✅' if passed else '❌'}")
            logger.warning(f"Full report: {report}")
        else:
            logger.info("Merged data validation passed")
            logger.info("All validation checks passed:")
            for check in validation_status.keys():
                logger.info(f"  ✅ {check}")
            logger.info(f"Data quality metrics: {report['quality_metrics']}")
        
        return is_valid, report

    def _check_timestamp_gaps(self, df: pd.DataFrame) -> Dict:
        """Check for gaps in timestamp sequence"""
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp')
            time_diff = df['timestamp'].diff()
            
            return {
                'max_gap_hours': time_diff.max().total_seconds() / 3600,
                'avg_gap_hours': time_diff.mean().total_seconds() / 3600,
                'gaps_over_1h': (time_diff > pd.Timedelta(hours=1)).sum()
            }
        except Exception as e:
            print(f"Error in _check_timestamp_gaps: {str(e)}")
            print("Timestamp column type:", df['timestamp'].dtype)
            print("Sample timestamps:", df['timestamp'].head())
            raise

    def _check_value_consistency(self, df: pd.DataFrame) -> Dict:
        """Check for value consistency and relationships"""
        return {
            'aqi_consistency': (df['aqi_numeric'] >= df['aqi_pm25']).all() and 
                             (df['aqi_numeric'] >= df['aqi_pm10']).all(),
            'temp_humidity_correlation': df['temperature'].corr(df['relative_humidity']),
            'pm_correlation': df['pm2_5'].corr(df['pm10'])
        }
