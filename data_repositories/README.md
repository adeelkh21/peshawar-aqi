# AQI Data Repository Structure

## Overview
This repository contains all data for the Peshawar AQI prediction system, organized for real-time processing and CI/CD pipeline integration.

## Directory Structure

### `/raw/`
- **weather_data.csv**: Raw weather data from Meteostat API
- **pollution_data.csv**: Raw pollution data from OpenWeatherMap API
- **metadata/**: Collection metadata and validation reports

### `/processed/`
- **merged_data.csv**: Cleaned and merged weather + pollution data
- **validation_reports/**: Data quality validation reports
- **metadata/**: Processing metadata and statistics

### `/features/`
- **engineered_features.csv**: Feature-engineered dataset for modeling
- **feature_metadata.json**: Feature definitions and metadata
- **validation/**: Feature validation reports

### `/models/`
- **trained_models/**: Serialized model files
- **model_metadata.json**: Model performance and versioning info
- **evaluation_reports/**: Model evaluation metrics

### `/historical_data/`
- **150_days_baseline.csv**: 150-day historical dataset for baseline training
- **metadata/**: Historical data metadata

## Data Flow
1. **Collection**: Hourly data collection → `/raw/`
2. **Processing**: Data cleaning and merging → `/processed/`
3. **Feature Engineering**: Feature creation → `/features/`
4. **Modeling**: Model training and evaluation → `/models/`

## Version Control
- All data files are tracked with Git LFS
- Metadata files contain versioning information
- Automated validation ensures data quality

## Quality Assurance
- Data validation at each stage
- Automated quality checks
- Performance monitoring
- Error handling and recovery
