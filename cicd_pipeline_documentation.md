# 🚀 AQI Forecasting CI/CD Pipeline Documentation

## 📋 Overview

This CI/CD pipeline implements a **fully automated, real-time AQI forecasting system** using GitHub Actions. The pipeline follows your exact specifications with **hourly data collection**, **6-hour model retraining**, and **continuous forecasting**.

## 🔄 Pipeline Architecture

### **Three Main Workflows:**

1. **🕐 Hourly Data Collection** (`aqi_data_pipeline.yml`)
2. **🤖 6-Hour Model Training** (`aqi_model_training.yml`) 
3. **🔮 2-Hour Forecasting** (`aqi_forecasting.yml`)

---

## 📅 Schedule & Timing

### **Data Collection Pipeline**
- **Frequency:** Every hour (`0 * * * *`)
- **Purpose:** Collect fresh real-time data from APIs
- **Duration:** ~5-10 minutes per run
- **Output:** Updated data in `data_repositories/`

### **Model Training Pipeline**
- **Frequency:** Every 6 hours (`0 */6 * * *`)
- **Purpose:** Retrain model with incremental learning
- **Duration:** ~15-20 minutes per run
- **Output:** Updated model in `deployment/`

### **Forecasting Pipeline**
- **Frequency:** Every 2 hours (`0 */2 * * *`)
- **Purpose:** Generate real-time forecasts
- **Duration:** ~5-8 minutes per run
- **Output:** Latest forecasts in `forecasts/`

---

## 🔧 How It Works

### **1. Data Collection Process**
```
🔄 Every Hour:
├── Fetch real-time weather data (OpenWeatherMap)
├── Fetch real-time pollution data (OpenWeatherMap)
├── Validate data quality and timestamps
├── Merge weather + pollution data
├── Save to data_repositories/processed/
└── Commit changes to repository
```

### **2. Model Training Process**
```
🔄 Every 6 Hours:
├── Load 150+ days historical data
├── Load recent real-time data
├── Merge datasets (remove duplicates)
├── Feature engineering (266 features)
├── Train LightGBM model (incremental learning)
├── Evaluate performance (R² ≥ 0.85)
├── Save model to deployment/
└── Commit model and reports
```

### **3. Forecasting Process**
```
🔄 Every 2 Hours:
├── Collect latest real-time data
├── Load latest trained model
├── Calculate current AQI (EPA standards)
├── Generate 72-hour forecast
├── Apply time-based adjustments
├── Validate forecast quality
├── Save forecast to forecasts/
└── Commit forecast results
```

---

## 📊 Data Flow

### **Historical Data Integration**
- **Source:** 150+ days of historical weather and pollution data
- **Location:** `data_repositories/historical_data/real_historical_dataset.csv`
- **Purpose:** Provides baseline patterns for model training

### **Real-Time Data Collection**
- **Source:** OpenWeatherMap APIs (weather + pollution)
- **Location:** `data_repositories/processed/merged_data.csv`
- **Purpose:** Provides current conditions for forecasting

### **Model Training Data**
- **Source:** Historical + Real-time merged dataset
- **Location:** `data_repositories/processed/complete_training_dataset.csv`
- **Purpose:** Complete dataset for model retraining

---

## 🎯 Key Features

### **Real-Time Data Collection**
- ✅ **Cache-busting API calls** prevent stale data
- ✅ **Data validation** ensures quality
- ✅ **Timestamp verification** confirms freshness
- ✅ **Automatic error handling** and retry logic

### **Incremental Learning**
- ✅ **6-hour retraining cycle** balances freshness and efficiency
- ✅ **Historical data preservation** maintains baseline patterns
- ✅ **Performance monitoring** (R² ≥ 0.85 threshold)
- ✅ **Model versioning** for rollback capability

### **EPA-Standard AQI Calculation**
- ✅ **PM2.5 and PM10 calculations** using EPA breakpoints
- ✅ **Real-time AQI computation** from pollution levels
- ✅ **Category mapping** (Good, Moderate, Unhealthy, etc.)
- ✅ **Time-based adjustments** for realistic forecasts

---

## 📁 Repository Structure

```
FinalIA/
├── .github/workflows/
│   ├── aqi_data_pipeline.yml      # Hourly data collection
│   ├── aqi_model_training.yml     # 6-hour model training
│   └── aqi_forecasting.yml        # 2-hour forecasting
├── data_repositories/
│   ├── historical_data/           # 150+ days baseline
│   ├── processed/                 # Merged datasets
│   ├── features/                  # Engineered features
│   └── models/                    # Trained models
├── deployment/                    # Latest model for production
├── forecasts/                     # Generated forecasts
└── requirements.txt               # Dependencies
```

---

## 🔍 Monitoring & Maintenance

### **GitHub Actions Dashboard**
- **URL:** `https://github.com/[username]/[repo]/actions`
- **Monitor:** Workflow runs, success/failure rates
- **Artifacts:** Download reports and logs

### **Key Metrics to Monitor**
1. **Data Collection Success Rate** (should be >95%)
2. **Model Performance** (R² ≥ 0.85)
3. **Forecast Quality** (valid AQI ranges)
4. **API Response Times** (should be <30 seconds)

### **Common Issues & Solutions**

#### **Data Collection Failures**
- **Cause:** API rate limits or network issues
- **Solution:** Automatic retry logic, check API keys

#### **Model Training Failures**
- **Cause:** Insufficient data or feature engineering errors
- **Solution:** Check data quality, validate feature pipeline

#### **Forecast Generation Failures**
- **Cause:** Model loading errors or data issues
- **Solution:** Verify model files, check data availability

---

## 🚀 Deployment

### **Local Development**
```bash
# Clone repository
git clone https://github.com/[username]/[repo].git
cd [repo]

# Install dependencies
pip install -r requirements.txt

# Run individual pipelines manually
python phase1_enhanced_data_collection.py
python phase2_enhanced_feature_engineering.py
```

### **Production Deployment**
- **Automatic:** GitHub Actions handle all scheduling
- **Manual Triggers:** Available via GitHub Actions UI
- **Monitoring:** GitHub Actions dashboard and logs

---

## 📈 Performance Expectations

### **Data Quality**
- **Freshness:** 1-3 hours old (acceptable for AQI forecasting)
- **Completeness:** >95% data availability
- **Accuracy:** EPA-standard AQI calculations

### **Model Performance**
- **R² Score:** ≥0.85 (85% accuracy)
- **Training Time:** <20 minutes per cycle
- **Forecast Quality:** Realistic AQI ranges (0-500)

### **System Reliability**
- **Uptime:** >99% (GitHub Actions reliability)
- **Data Collection:** 24/7 automated
- **Model Updates:** Every 6 hours
- **Forecasts:** Every 2 hours

---

## 🔧 Configuration

### **Environment Variables**
```bash
# Required for API access
OPENWEATHERMAP_API_KEY=your_api_key_here
```

### **Schedule Customization**
Edit the cron expressions in workflow files:
- **Data Collection:** `0 * * * *` (every hour)
- **Model Training:** `0 */6 * * *` (every 6 hours)
- **Forecasting:** `0 */2 * * *` (every 2 hours)

---

## 🎉 Benefits

### **Automation**
- ✅ **Zero manual intervention** required
- ✅ **24/7 operation** with GitHub Actions
- ✅ **Automatic error recovery** and retry logic

### **Accuracy**
- ✅ **Real-time data** collection every hour
- ✅ **Incremental learning** every 6 hours
- ✅ **EPA-standard calculations** for AQI

### **Scalability**
- ✅ **GitHub Actions** handle infrastructure
- ✅ **Modular design** for easy maintenance
- ✅ **Version control** for all data and models

---

## 📞 Support

### **Troubleshooting**
1. Check GitHub Actions logs for error details
2. Verify API keys and rate limits
3. Monitor data quality reports
4. Review model performance metrics

### **Manual Operations**
- **Trigger workflows manually** via GitHub Actions UI
- **Download artifacts** for detailed analysis
- **Rollback models** if performance degrades

---

## 🎯 Success Metrics

### **Immediate Goals**
- ✅ **Automated data collection** every hour
- ✅ **Model retraining** every 6 hours
- ✅ **Real-time forecasting** every 2 hours

### **Long-term Goals**
- 📈 **Improve model accuracy** over time
- 📈 **Expand historical data** beyond 150 days
- 📈 **Add more locations** beyond Peshawar

**This CI/CD pipeline transforms your AQI forecasting system into a fully automated, production-ready solution that continuously learns and provides accurate real-time predictions!** 🚀
