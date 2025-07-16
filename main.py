"""
🏠 House Price Prediction - Main Training Pipeline
Enhanced Version with Complete ML Workflow

Author: Data Science Team
Version: 2.0
Date: 2024
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import load_data, preprocess_data, explore_data
from src.train_models import train_models, get_best_model
from src.evaluate import (plot_rmse_comparison, plot_r2_comparison, 
                         plot_comprehensive_evaluation, create_interactive_comparison)
from utils.helpers import (setup_plotting_style, save_model, print_model_summary, 
                          create_directory_if_not_exists)
from config.config import *

def main():
    """Main training pipeline"""
    print("🏠" + "="*70)
    print("   HOUSE PRICE PREDICTION - ENHANCED ML PIPELINE")
    print("="*70 + "🏠")
    
    # Setup
    setup_plotting_style()
    create_directory_if_not_exists(MODEL_DIR)
    create_directory_if_not_exists(DEMO_DIR)
    
    try:
        # Step 1: Load and explore data
        print("\n📊 STEP 1: DATA LOADING & EXPLORATION")
        print("-" * 50)
        
        # Try multiple possible data paths
        possible_paths = [
            TRAIN_DATA_PATH,
            PROJECT_ROOT / "data" / "train.csv",
            "data/train.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = load_data(str(path))
                print(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find train.csv in any expected location")
        
        # Explore data
        missing_summary = explore_data(df)
        
        # Step 2: Data preprocessing
        print("\n🔧 STEP 2: DATA PREPROCESSING")
        print("-" * 50)
        
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
        
        print(f"✅ Preprocessing completed!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        
        # Step 3: Model training and evaluation
        print("\n🤖 STEP 3: MODEL TRAINING & EVALUATION")
        print("-" * 50)
        
        # Train models with hyperparameter tuning
        simple_results, detailed_results = train_models(
            X_train, X_test, y_train, y_test, tune_hyperparams=True
        )
        
        # Step 4: Visualizations
        print("\n📊 STEP 4: CREATING VISUALIZATIONS")
        print("-" * 50)
        
        # Model comparison plots
        rmse_fig = plot_rmse_comparison(simple_results, save_path=DEMO_DIR / "rmse_comparison.png")
        r2_fig = plot_r2_comparison(simple_results, save_path=DEMO_DIR / "r2_comparison.png")
        
        # Get best model for detailed analysis
        best_model_name, best_model = get_best_model(detailed_results)
        print(f"🏆 Best model selected: {best_model_name}")
        
        # Comprehensive evaluation for best model
        evaluation_plots = plot_comprehensive_evaluation(
            simple_results, best_model_name, best_model, 
            X_test, y_test, feature_names, save_dir=DEMO_DIR
        )
        
        # Interactive comparison (if plotly is available)
        try:
            interactive_fig = create_interactive_comparison(simple_results)
            interactive_fig.write_html(str(DEMO_DIR / "interactive_comparison.html"))
            print("✅ Interactive comparison chart created")
        except Exception as e:
            print(f"⚠️  Could not create interactive chart: {e}")
        
        # Step 5: Cross-validation analysis
        print("\n🔬 STEP 5: CROSS-VALIDATION ANALYSIS")
        print("-" * 50)
        
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                  cv=CV_FOLDS, scoring='r2')
        cv_rmse_scores = np.sqrt(-cross_val_score(best_model, X_train, y_train, 
                                                cv=CV_FOLDS, scoring='neg_mean_squared_error'))
        
        print(f"� {CV_FOLDS}-Fold Cross Validation Results ({best_model_name}):")
        print(f"   Mean R² Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"   Mean RMSE: {cv_rmse_scores.mean():.0f} (±{cv_rmse_scores.std()*2:.0f})")
        
        # Step 6: Save final model
        print("\n💾 STEP 6: SAVING FINAL MODEL")
        print("-" * 50)
        
        # Retrain best model on full training data
        best_model.fit(X_train, y_train)
        
        # Save model and preprocessor
        save_model(best_model, preprocessor, MODEL_PATH)
        
        # Save feature names for later use
        feature_info = {
            'feature_names': feature_names,
            'model_name': best_model_name,
            'performance_metrics': simple_results[best_model_name]
        }
        joblib.dump(feature_info, MODEL_DIR / 'feature_info.pkl')
        
        # Step 7: Final summary
        print("\n🎉 STEP 7: TRAINING SUMMARY")
        print("-" * 50)
        
        print_model_summary(simple_results)
        
        print(f"\n📁 Files saved:")
        print(f"   🤖 Model: {MODEL_PATH}")
        print(f"   📊 Visualizations: {DEMO_DIR}")
        print(f"   📈 Feature info: {MODEL_DIR / 'feature_info.pkl'}")
        
        # Performance insights
        best_metrics = simple_results[best_model_name]
        print(f"\n🔍 PERFORMANCE INSIGHTS:")
        print(f"   • Model explains {best_metrics['R2']*100:.1f}% of price variance")
        print(f"   • Average prediction error: ${best_metrics['RMSE']:,.0f}")
        print(f"   • Mean absolute error: ${best_metrics['MAE']:,.0f}")
        
        if best_metrics['R2'] > 0.85:
            print(f"   🏆 Excellent model performance!")
        elif best_metrics['R2'] > 0.75:
            print(f"   👍 Good model performance!")
        else:
            print(f"   ⚠️  Model could be improved further")
        
        print("\n✅ Training pipeline completed successfully!")
        print("🚀 Ready for deployment!")
        
        return best_model, preprocessor, feature_names, simple_results
        
    except Exception as e:
        print(f"\n❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_prediction_example():
    """Create an example prediction to test the model"""
    try:
        # Load saved model
        model, preprocessor = joblib.load(MODEL_PATH)
        feature_info = joblib.load(MODEL_DIR / 'feature_info.pkl')
        
        print("\n🧪 TESTING MODEL WITH SAMPLE PREDICTION")
        print("-" * 50)
        
        # Create sample house data (you would replace this with real input)
        sample_house = pd.DataFrame({
            # Add sample features here based on your dataset
            'GrLivArea': [2000],
            'OverallQual': [7],
            'YearBuilt': [2005],
            'TotalBsmtSF': [1200],
            'GarageCars': [2],
            # Add more features as needed based on your preprocessor
        })
        
        print("🏠 Sample house features:")
        for col, val in sample_house.iloc[0].items():
            print(f"   {col}: {val}")
        
        # Note: This is a simplified example. In practice, you'd need to ensure
        # the sample data has all required features and proper preprocessing.
        
    except Exception as e:
        print(f"⚠️  Could not run prediction example: {e}")

if __name__ == "__main__":
    # Run main training pipeline
    model, preprocessor, feature_names, results = main()
    
    # Optional: Create prediction example
    create_prediction_example()
