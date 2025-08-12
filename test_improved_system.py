#!/usr/bin/env python3
"""
Test Improved AQI System
========================

This script tests the improved AQI prediction system with real data and calibration.
"""

from phase5_fixed_production import FixedProductionIntegration
import json

def test_improved_system():
    """Test the improved AQI prediction system"""
    print("🧪 Testing Improved AQI Prediction System")
    print("=" * 50)
    
    try:
        # Initialize the production system
        production = FixedProductionIntegration()
        
        # Load the trained model
        model, scaler = production.load_model()
        print("✅ Model loaded successfully")
        
        # Generate a 24-hour forecast
        forecast = production.generate_realistic_forecast(model, scaler, hours=24)
        
        # Display results
        print(f"\n📊 Forecast Results:")
        print(f"Current AQI: {forecast['predictions'][0]:.1f}")
        print(f"Current Category: {forecast['categories'][0]}")
        print(f"Forecast Range: {min(forecast['predictions']):.1f} - {max(forecast['predictions']):.1f}")
        print(f"Category Distribution: {dict(zip(*np.unique(forecast['categories'], return_counts=True)))}")
        
        # Compare with actual AQI
        actual_aqi = production.get_current_actual_aqi()
        predicted_aqi = forecast['predictions'][0]
        error = abs(predicted_aqi - actual_aqi)
        error_percentage = (error / actual_aqi) * 100
        
        print(f"\n🎯 Accuracy Assessment:")
        print(f"Actual AQI: {actual_aqi:.1f}")
        print(f"Predicted AQI: {predicted_aqi:.1f}")
        print(f"Absolute Error: {error:.1f}")
        print(f"Error Percentage: {error_percentage:.1f}%")
        
        if error_percentage < 20:
            print("✅ Excellent accuracy!")
        elif error_percentage < 40:
            print("✅ Good accuracy")
        else:
            print("⚠️  Needs improvement")
        
        # Save forecast for inspection
        with open('test_forecast.json', 'w') as f:
            json.dump(forecast, f, indent=2, default=str)
        
        print(f"\n💾 Forecast saved to: test_forecast.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing system: {str(e)}")
        return False

if __name__ == "__main__":
    import numpy as np
    test_improved_system()
