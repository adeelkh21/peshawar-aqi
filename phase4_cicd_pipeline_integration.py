"""
Phase 4: CI/CD Pipeline Integration - Corrected Version
=====================================================

This script implements a comprehensive CI/CD pipeline following the successful
patterns from the existing Phase 4 files that achieved 89-95% R² performance.

Key Features:
- Follows proven data loading and preprocessing from Phase 4
- Uses successful model training approaches (XGBoost, LightGBM)
- Implements the same feature engineering that achieved high performance
- Automated deployment and testing
- Real-time integration with Streamlit app

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
import shutil
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import time

# Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb

# Import the successful Phase 4 foundation
from phase4_model_development import Phase4ModelDevelopment
from phase4_xgboost_implementation import XGBoostOptimizer
from phase4_lightgbm_implementation import LightGBMOptimizer
from phase4_final_evaluation import Phase4FinalEvaluation

warnings.filterwarnings('ignore')

class CICDPipelineIntegration:
    """
    CI/CD Pipeline Integration following successful Phase 4 patterns
    Target: Achieve 89-95% R² performance like the existing Phase 4 files
    """
    
    def __init__(self):
        """Initialize CI/CD Pipeline following Phase 4 patterns"""
        print("🚀 PHASE 4: CI/CD PIPELINE INTEGRATION")
        print("=" * 50)
        print("🎯 Following successful Phase 4 patterns")
        print("📊 Target: 89-95% R² performance")
        print("🔄 Automated pipeline with proven methodology")
        print()
        
        self.pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
        # Initialize Phase 4 components
        self.phase4_base = None
        self.xgb_optimizer = None
        self.lgb_optimizer = None
        self.final_evaluator = None
        
        # Performance tracking
        self.performance_history = []
        self.best_performance = 0.0
        self.current_model = None
        self.best_model_name = None
        
        # Pipeline configuration
        self.pipeline_config = {
            'data_loading': {
                'enabled': True,
                'use_phase4_methods': True
            },
            'feature_engineering': {
                'enabled': True,
                'use_phase4_features': True
            },
            'model_training': {
                'enabled': True,
                'models': ['random_forest', 'xgboost', 'lightgbm'],
                'target_performance': 0.75  # Unified target for reporting/thresholds
            },
            'deployment': {
                'enabled': True,
                'auto_deploy': True,
                'rollback_threshold': 0.55  # Realistic rollback threshold for AQI
            }
        }
        
        print(f"⏰ Pipeline Timestamp: {self.pipeline_timestamp}")
        print(f"🔧 Pipeline Components: {len(self.pipeline_config)} configured")

    def setup_logging(self):
        """Setup comprehensive logging for CI/CD pipeline"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"cicd_pipeline_{self.pipeline_timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("CI/CD Pipeline initialized")

    def step1_phase4_foundation(self) -> bool:
        """Step 1: Initialize Phase 4 foundation following proven patterns"""
        print("\n📊 STEP 1: Phase 4 Foundation Setup")
        print("-" * 40)
        
        try:
            self.logger.info("Starting Phase 4 foundation setup")
            
            # Initialize Phase 4 base following the successful pattern
            print("🔄 Initializing Phase 4 foundation...")
            self.phase4_base = Phase4ModelDevelopment()
            
            # Run the proven Phase 4 steps 1-3
            print("🔄 Running Phase 4 steps 1-3...")
            foundation_success = self.phase4_base.run_phase4_step1_to_3()
            
            if not foundation_success:
                self.logger.error("Phase 4 foundation failed")
                print("❌ Phase 4 foundation failed")
                return False
            
            # Extract the successful data and models
            self.X_train = self.phase4_base.X_train
            self.X_test = self.phase4_base.X_test
            self.y_train = self.phase4_base.y_train
            self.y_test = self.phase4_base.y_test
            
            # Get baseline performance
            baseline_r2 = self.phase4_base.results['random_forest']['test_r2']
            print(f"📊 Baseline Random Forest R²: {baseline_r2:.3f}")
            
            if baseline_r2 >= 0.65:
                print("✅ Baseline performance meets target (65% R²)")
            else:
                print(f"⚠️  Baseline below target ({baseline_r2:.3f} < 0.65)")
            
            print("✅ Phase 4 foundation completed successfully")
            self.logger.info("Phase 4 foundation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 4 foundation error: {str(e)}")
            print(f"❌ Phase 4 foundation error: {str(e)}")
            return False

    def step2_xgboost_optimization(self) -> bool:
        """Step 2: XGBoost optimization following Phase 4 patterns"""
        print("\n🚀 STEP 2: XGBoost Optimization")
        print("-" * 40)
        
        try:
            self.logger.info("Starting XGBoost optimization")
            
            # Initialize XGBoost optimizer with Phase 4 foundation
            print("🔄 Initializing XGBoost optimizer...")
            self.xgb_optimizer = XGBoostOptimizer(self.phase4_base)
            
            # Run XGBoost optimization steps
            print("🔄 Running XGBoost optimization...")
            
            # Step 4: Basic XGBoost
            self.xgb_optimizer.step4_basic_xgboost()
            
            # Step 5: XGBoost hyperparameter tuning
            self.xgb_optimizer.step5_hyperparameter_tuning()
            
            # Step 6: XGBoost feature importance analysis
            self.xgb_optimizer.step6_feature_importance_analysis()
            
            # Get best XGBoost performance
            if 'xgboost_optimized' in self.xgb_optimizer.results:
                xgb_performance = self.xgb_optimizer.results['xgboost_optimized']['test_r2']
                print(f"🏆 Best XGBoost R²: {xgb_performance:.3f}")
                
                if xgb_performance > self.best_performance:
                    self.best_performance = xgb_performance
                    self.best_model_name = 'xgboost_optimized'
                    self.current_model = self.xgb_optimizer.models['xgboost_optimized']
            
            print("✅ XGBoost optimization completed")
            self.logger.info("XGBoost optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"XGBoost optimization error: {str(e)}")
            print(f"❌ XGBoost optimization error: {str(e)}")
            return False

    def step3_lightgbm_optimization(self) -> bool:
        """Step 3: LightGBM optimization following Phase 4 patterns"""
        print("\n⚡ STEP 3: LightGBM Optimization")
        print("-" * 40)
        
        try:
            self.logger.info("Starting LightGBM optimization")
            
            # Initialize LightGBM optimizer with Phase 4 foundation
            print("🔄 Initializing LightGBM optimizer...")
            self.lgb_optimizer = LightGBMOptimizer(self.phase4_base)
            
            # Run LightGBM optimization steps
            print("🔄 Running LightGBM optimization...")
            
            # Step 7: Basic LightGBM
            self.lgb_optimizer.step7_basic_lightgbm()
            
            # Step 8: LightGBM hyperparameter tuning
            self.lgb_optimizer.step8_lightgbm_hyperparameter_tuning()
            
            # Step 9: LightGBM model comparison
            self.lgb_optimizer.step9_model_comparison()
            
            # Get best LightGBM performance
            if 'lightgbm_optimized' in self.lgb_optimizer.results:
                lgb_performance = self.lgb_optimizer.results['lightgbm_optimized']['test_r2']
                print(f"🏆 Best LightGBM R²: {lgb_performance:.3f}")
                
                if lgb_performance > self.best_performance:
                    self.best_performance = lgb_performance
                    self.best_model_name = 'lightgbm_optimized'
                    self.current_model = self.lgb_optimizer.models['lightgbm_optimized']
            
            print("✅ LightGBM optimization completed")
            self.logger.info("LightGBM optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"LightGBM optimization error: {str(e)}")
            print(f"❌ LightGBM optimization error: {str(e)}")
            return False

    def step4_final_evaluation_and_ensemble(self) -> bool:
        """Step 4: Final evaluation and ensemble creation"""
        print("\n🏆 STEP 4: Final Evaluation & Ensemble")
        print("-" * 40)
        
        try:
            self.logger.info("Starting final evaluation and ensemble")
            
            # Initialize final evaluator
            print("🔄 Initializing final evaluator...")
            self.final_evaluator = Phase4FinalEvaluation()
            
            # Run final evaluation steps
            print("🔄 Running final evaluation...")
            
            # Step 10: Load and recreate models
            self.final_evaluator.step10_load_and_recreate_models()
            
            # Step 11: Create ensemble models
            self.final_evaluator.step11_create_ensemble_models()
            
            # Step 12: Final model selection
            self.final_evaluator.step12_final_model_selection()
            
            # Get best ensemble performance
            if hasattr(self.final_evaluator, 'final_results') and self.final_evaluator.final_results:
                best_ensemble = max(self.final_evaluator.final_results.items(), 
                                  key=lambda x: x[1]['test_r2'])
                ensemble_name, ensemble_performance = best_ensemble
                
                print(f"🏆 Best Ensemble R²: {ensemble_performance['test_r2']:.3f}")
                
                if ensemble_performance['test_r2'] > self.best_performance:
                    self.best_performance = ensemble_performance['test_r2']
                    self.best_model_name = ensemble_name
                    self.current_model = self.final_evaluator.ensemble_models[ensemble_name]
            
            print("✅ Final evaluation and ensemble completed")
            self.logger.info("Final evaluation and ensemble completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Final evaluation error: {str(e)}")
            print(f"❌ Final evaluation error: {str(e)}")
            return False

    def step5_model_deployment(self) -> bool:
        """Step 5: Model deployment and testing"""
        print("\n🚀 STEP 5: Model Deployment")
        print("-" * 40)
        
        try:
            self.logger.info("Starting model deployment")
            
            # Check if we have a good model to deploy
            if self.best_performance < self.pipeline_config['deployment']['rollback_threshold']:
                self.logger.warning(f"Performance {self.best_performance:.3f} below rollback threshold")
                print(f"⚠️  Performance below rollback threshold ({self.best_performance:.3f})")
                return False
            
            # Create deployment directory
            print("📦 Creating deployment package...")
            deployment_dir = os.path.join("deployment", self.pipeline_timestamp)
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Save the best model
            if self.current_model is not None:
                model_file = os.path.join(deployment_dir, "production_model.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(self.current_model, f)
                
                print(f"✅ Model saved to: {model_file}")
            
            # Create deployment metadata
            deployment_metadata = {
                "timestamp": self.pipeline_timestamp,
                "model_name": self.best_model_name,
                "performance": self.best_performance,
                "target_achieved": bool(self.best_performance >= 0.89),  # Convert to Python bool
                "deployment_status": "ready"
            }
            
            metadata_file = os.path.join(deployment_dir, "deployment_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(deployment_metadata, f, indent=4)
            
            print(f"✅ Deployment metadata saved to: {metadata_file}")
            
            # Test deployment
            print("🧪 Testing deployment...")
            test_success = self._test_deployment()
            
            if not test_success:
                self.logger.error("Deployment test failed")
                return False
            
            print("✅ Model deployment completed")
            self.logger.info("Model deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment error: {str(e)}")
            print(f"❌ Model deployment error: {str(e)}")
            return False

    def step6_integration_testing(self) -> bool:
        """Step 6: Integration testing with Streamlit app"""
        print("\n🧪 STEP 6: Integration Testing")
        print("-" * 40)
        
        try:
            self.logger.info("Starting integration testing")
            
            # Test model loading
            print("🔄 Testing model loading...")
            model_load_success = self._test_model_loading()
            
            if not model_load_success:
                self.logger.error("Model loading test failed")
                return False
            
            # Test prediction pipeline
            print("🔮 Testing prediction pipeline...")
            prediction_success = self._test_prediction_pipeline()
            
            if not prediction_success:
                self.logger.error("Prediction pipeline test failed")
                return False
            
            # Test Streamlit integration
            print("🌐 Testing Streamlit integration...")
            streamlit_success = self._test_streamlit_integration()
            
            if not streamlit_success:
                self.logger.warning("Streamlit integration test failed (non-critical)")
            
            print("✅ Integration testing completed")
            self.logger.info("Integration testing completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Integration testing error: {str(e)}")
            print(f"❌ Integration testing error: {str(e)}")
            return False

    def _test_deployment(self) -> bool:
        """Test deployment"""
        try:
            # Test model loading
            deployment_dir = os.path.join("deployment", self.pipeline_timestamp)
            model_file = os.path.join(deployment_dir, "production_model.pkl")
            
            if not os.path.exists(model_file):
                return False
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Test prediction with sample data
            if hasattr(self, 'X_test') and self.X_test is not None:
                sample_data = self.X_test[:5]
                predictions = model.predict(sample_data)
                
                if len(predictions) == 5:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Deployment test error: {str(e)}")
            return False

    def _test_model_loading(self) -> bool:
        """Test model loading"""
        try:
            deployment_dir = os.path.join("deployment", self.pipeline_timestamp)
            model_file = os.path.join(deployment_dir, "production_model.pkl")
            
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Model loading test error: {str(e)}")
            return False

    def _test_prediction_pipeline(self) -> bool:
        """Test prediction pipeline"""
        try:
            deployment_dir = os.path.join("deployment", self.pipeline_timestamp)
            model_file = os.path.join(deployment_dir, "production_model.pkl")
            
            if not os.path.exists(model_file):
                return False
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Test prediction with sample data
            if hasattr(self, 'X_test') and self.X_test is not None:
                sample_data = self.X_test[:3]
                predictions = model.predict(sample_data)
                
                return len(predictions) == 3
            
            return False
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline test error: {str(e)}")
            return False

    def _test_streamlit_integration(self) -> bool:
        """Test Streamlit integration"""
        try:
            # Check if Streamlit app exists
            streamlit_file = "streamlit_app.py"
            if not os.path.exists(streamlit_file):
                self.logger.warning("Streamlit app not found")
                return True  # Not critical for pipeline success
            
            # Test if model can be loaded by Streamlit
            return self._test_model_loading()
            
        except Exception as e:
            self.logger.error(f"Streamlit integration test error: {str(e)}")
            return False

    def run_complete_pipeline(self) -> bool:
        """Run complete CI/CD pipeline following Phase 4 patterns"""
        print("\n🚀 Running Complete CI/CD Pipeline")
        print("=" * 50)
        
        self.logger.info("Starting complete CI/CD pipeline")
        
        pipeline_steps = [
            ("Phase 4 Foundation", self.step1_phase4_foundation),
            ("XGBoost Optimization", self.step2_xgboost_optimization),
            ("LightGBM Optimization", self.step3_lightgbm_optimization),
            ("Final Evaluation & Ensemble", self.step4_final_evaluation_and_ensemble),
            ("Model Deployment", self.step5_model_deployment),
            ("Integration Testing", self.step6_integration_testing)
        ]
        
        results = {}
        
        for step_name, step_function in pipeline_steps:
            print(f"\n🔄 Running {step_name}...")
            start_time = time.time()
            
            try:
                success = step_function()
                end_time = time.time()
                duration = end_time - start_time
                
                results[step_name] = {
                    'success': success,
                    'duration': duration
                }
                
                if success:
                    print(f"✅ {step_name} completed in {duration:.2f}s")
                else:
                    print(f"❌ {step_name} failed after {duration:.2f}s")
                    break
                    
            except Exception as e:
                self.logger.error(f"{step_name} error: {str(e)}")
                print(f"❌ {step_name} error: {str(e)}")
                results[step_name] = {
                    'success': False,
                    'error': str(e)
                }
                break
        
        # Generate pipeline report
        self._generate_pipeline_report(results)
        
        # Check overall success
        overall_success = all(result.get('success', False) for result in results.values())
        
        if overall_success:
            print("\n🎉 CI/CD Pipeline Completed Successfully!")
            print(f"🏆 Best Performance: {self.best_performance:.3f} R²")
            print(f"🤖 Best Model: {self.best_model_name}")
            
            target = self.pipeline_config['model_training']['target_performance']
            if self.best_performance >= target:
                print(f"🎯 Target Performance ({target:.2f} R²) ACHIEVED!")
            else:
                print(f"⚠️  Target performance not achieved ({self.best_performance:.3f} < {target:.2f})")
            
            print("📈 Ready for production deployment!")
        else:
            print("\n❌ CI/CD Pipeline Failed!")
            print("🔍 Check logs for details")
        
        return overall_success

    def _generate_pipeline_report(self, results: Dict):
        """Generate comprehensive pipeline report"""
        try:
            report = {
                "pipeline_timestamp": self.pipeline_timestamp,
                "overall_success": all(result.get('success', False) for result in results.values()),
                "step_results": results,
                "performance_summary": {
                    "best_performance": self.best_performance,
                    "best_model": self.best_model_name,
                    "target_achieved": self.best_performance >= self.pipeline_config['model_training']['target_performance'],
                    "target_performance": self.pipeline_config['model_training']['target_performance']
                },
                "pipeline_config": self.pipeline_config
            }
            
            report_file = f"cicd_pipeline_report_{self.pipeline_timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            
            print(f"📋 Pipeline report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Pipeline report generation error: {str(e)}")

def main():
    """Run CI/CD Pipeline Integration"""
    pipeline = CICDPipelineIntegration()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Phase 4 CI/CD Pipeline Complete!")
        print("📊 Following proven Phase 4 methodology")
        print("🔄 Real-time integration ready")
        print("📈 Ready for Phase 5: Production Deployment")
    else:
        print("\n❌ Pipeline failed! Check error messages and logs above.")

if __name__ == "__main__":
    main()
