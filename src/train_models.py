"""
Production-Ready ML Model Training Module for House Price Prediction

This module provides a comprehensive training pipeline for multiple regression models
with automated hyperparameter tuning, cross-validation, model evaluation, and 
artifact management for production deployment.

Author: AI Engineer
Date: July 2025
Version: 2.0
"""

import os
import pickle
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_validate
)

# Optional XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE TRAINING FUNCTIONS
# ============================================================================

def get_models() -> Dict[str, Any]:
    """
    Get dictionary of initialized regression models for training.
    
    Returns:
        Dict[str, Any]: Dictionary mapping model names to initialized model objects
    """
    try:
        logger.info("🤖 Initializing regression models...")
        
        models = {
            "Linear_Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42, max_iter=10000),
            "ElasticNet": ElasticNet(random_state=42, max_iter=10000),
            "Random_Forest": RandomForestRegressor(
                random_state=42, 
                n_jobs=-1,
                n_estimators=100
            ),
            "Gradient_Boosting": GradientBoostingRegressor(
                random_state=42,
                n_estimators=100
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                n_estimators=100,
                verbosity=0
            )
        
        logger.info(f"✅ Initialized {len(models)} models")
        return models
        
    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        raise


def get_hyperparams() -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for each model for GridSearchCV/RandomizedSearchCV.
    
    Returns:
        Dict[str, Dict[str, List]]: Nested dictionary with hyperparameter grids
    """
    try:
        logger.info("⚙️ Defining hyperparameter grids...")
        
        hyperparams = {
            "Linear_Regression": {
                # Linear regression has no hyperparameters to tune
            },
            
            "Ridge": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                'max_iter': [1000, 5000]
            },
            
            "Lasso": {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-4, 1e-3, 1e-2]
            },
            
            "ElasticNet": {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-4, 1e-3]
            },
            
            "Random_Forest": {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            
            "Gradient_Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # Add XGBoost hyperparameters if available
        if XGBOOST_AVAILABLE:
            hyperparams["XGBoost"] = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
        
        logger.info(f"✅ Defined hyperparameters for {len(hyperparams)} models")
        return hyperparams
        
    except Exception as e:
        logger.error(f"❌ Error defining hyperparameters: {e}")
        raise


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    try:
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Mean Squared Log Error (if all values are positive)
        if np.all(y_true > 0) and np.all(y_pred > 0):
            msle = np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred)))
        else:
            msle = np.nan
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Median_AE': median_ae,
            'MSLE': msle
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {e}")
        raise


def train_and_evaluate_models(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    cv: int = 5,
    search_method: str = "randomized",
    n_iter: int = 50,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all models with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test target
        cv (int): Number of cross-validation folds
        search_method (str): 'grid' or 'randomized' search
        n_iter (int): Number of iterations for randomized search
        scoring (str): Scoring metric for optimization
        n_jobs (int): Number of parallel jobs
        verbose (bool): Whether to print progress
        
    Returns:
        Dict[str, Dict[str, Any]]: Results for each model including best params and metrics
    """
    try:
        logger.info("🚀 Starting model training and evaluation...")
        
        models = get_models()
        hyperparams = get_hyperparams()
        results = {}
        
        for model_name, model in models.items():
            if verbose:
                print(f"\n🔧 Training {model_name}...")
            
            start_time = time.time()
            
            # Get hyperparameters for this model
            param_grid = hyperparams.get(model_name, {})
            
            if param_grid:
                # Hyperparameter tuning
                if search_method.lower() == "grid":
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        return_train_score=True
                    )
                else:  # randomized
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        random_state=42,
                        return_train_score=True
                    )
                
                # Fit the search
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                cv_score = -search.best_score_  # Convert back to positive
                
            else:
                # No hyperparameters to tune (e.g., Linear Regression)
                search = None
                best_model = model
                best_params = {}
                
                # Manual cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring=scoring, n_jobs=n_jobs
                )
                cv_score = -cv_scores.mean()  # Convert to positive
                
                # Fit the model
                best_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            train_metrics = calculate_metrics(y_train, y_train_pred)
            
            test_metrics = None
            if X_test is not None and y_test is not None:
                y_test_pred = best_model.predict(X_test)
                test_metrics = calculate_metrics(y_test, y_test_pred)
            
            # Detailed cross-validation metrics
            detailed_cv = cross_validate(
                best_model, X_train, y_train,
                cv=cv,
                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                n_jobs=n_jobs,
                return_train_score=True
            )
            
            cv_metrics = {
                'CV_RMSE_mean': np.sqrt(-detailed_cv['test_neg_mean_squared_error'].mean()),
                'CV_RMSE_std': np.sqrt(-detailed_cv['test_neg_mean_squared_error']).std(),
                'CV_MAE_mean': -detailed_cv['test_neg_mean_absolute_error'].mean(),
                'CV_MAE_std': (-detailed_cv['test_neg_mean_absolute_error']).std(),
                'CV_R2_mean': detailed_cv['test_r2'].mean(),
                'CV_R2_std': detailed_cv['test_r2'].std()
            }
            
            # Store results
            results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'training_time': training_time,
                'search_object': search
            }
            
            if verbose:
                rmse_score = np.sqrt(cv_score)
                print(f"✅ {model_name}: CV RMSE = {rmse_score:.0f}, "
                      f"Time = {training_time:.2f}s")
                if best_params:
                    print(f"   Best params: {best_params}")
        
        logger.info(f"✅ Training completed for {len(results)} models")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in training and evaluation: {e}")
        raise


def save_best_model(
    model: Any, 
    filename: str,
    model_info: Optional[Dict] = None,
    models_dir: str = "models"
) -> str:
    """
    Save the best model using joblib with metadata.
    
    Args:
        model (Any): Trained model object
        filename (str): Base filename for the model
        model_info (Dict, optional): Additional model information to save
        models_dir (str): Directory to save models
        
    Returns:
        str: Full path to saved model file
    """
    try:
        logger.info(f"💾 Saving model: {filename}")
        
        # Create models directory if it doesn't exist
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{filename}_{timestamp}.pkl"
        full_path = models_path / model_filename
        
        # Prepare data to save
        model_data = {
            'model': model,
            'timestamp': timestamp,
            'model_info': model_info or {},
            'sklearn_version': None,  # You can add version tracking
            'training_date': datetime.now().isoformat()
        }
        
        # Save using joblib (preferred for sklearn models)
        joblib.dump(model_data, full_path)
        
        # Also save just the model for easy loading
        simple_path = models_path / f"{filename}_simple.pkl"
        joblib.dump(model, simple_path)
        
        logger.info(f"✅ Model saved to: {full_path}")
        logger.info(f"✅ Simple model saved to: {simple_path}")
        
        return str(full_path)
        
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        raise


def load_model(model_path: str) -> Tuple[Any, Optional[Dict]]:
    """
    Load a saved model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Tuple[Any, Optional[Dict]]: Loaded model and metadata
    """
    try:
        logger.info(f"📂 Loading model from: {model_path}")
        
        # Load the model data
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            # Full model data with metadata
            model = model_data['model']
            metadata = {k: v for k, v in model_data.items() if k != 'model'}
            logger.info("✅ Model and metadata loaded successfully")
            return model, metadata
        else:
            # Simple model file
            logger.info("✅ Simple model loaded successfully")
            return model_data, None
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

# ============================================================================
# REPORTING AND ANALYSIS FUNCTIONS
# ============================================================================

def get_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Get the best performing model based on cross-validation R² score.
    
    Args:
        results (Dict[str, Dict[str, Any]]): Results from train_and_evaluate_models
        
    Returns:
        Tuple[str, Any, Dict[str, Any]]: Best model name, model object, and metrics
    """
    try:
        logger.info("🏆 Identifying best model...")
        
        # Sort by CV R² score (higher is better)
        best_model_name = max(
            results.keys(), 
            key=lambda k: results[k]['cv_metrics']['CV_R2_mean']
        )
        
        best_result = results[best_model_name]
        best_model = best_result['model']
        
        logger.info(f"✅ Best model: {best_model_name}")
        return best_model_name, best_model, best_result
        
    except Exception as e:
        logger.error(f"❌ Error identifying best model: {e}")
        raise


def print_model_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a comprehensive model comparison report.
    
    Args:
        results (Dict[str, Dict[str, Any]]): Results from train_and_evaluate_models
    """
    try:
        logger.info("📊 Generating model comparison report...")
        
        print("\n" + "="*100)
        print("🏆 MODEL PERFORMANCE COMPARISON REPORT")
        print("="*100)
        
        # Sort models by CV R² score
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['cv_metrics']['CV_R2_mean'],
            reverse=True
        )
        
        # Print header
        print(f"{'Rank':<4} {'Model':<20} {'CV R²':<8} {'CV RMSE':<10} {'CV MAE':<10} "
              f"{'Test R²':<8} {'Test RMSE':<10} {'Train Time':<12}")
        print("-" * 100)
        
        # Print model results
        for rank, (model_name, result) in enumerate(sorted_models, 1):
            cv_r2 = result['cv_metrics']['CV_R2_mean']
            cv_rmse = result['cv_metrics']['CV_RMSE_mean']
            cv_mae = result['cv_metrics']['CV_MAE_mean']
            
            test_r2 = result['test_metrics']['R2'] if result['test_metrics'] else 0
            test_rmse = result['test_metrics']['RMSE'] if result['test_metrics'] else 0
            
            train_time = result['training_time']
            
            print(f"{rank:<4} {model_name:<20} {cv_r2:<8.4f} {cv_rmse:<10.0f} "
                  f"{cv_mae:<10.0f} {test_r2:<8.4f} {test_rmse:<10.0f} {train_time:<12.2f}s")
        
        # Highlight best model
        best_model_name = sorted_models[0][0]
        best_r2 = sorted_models[0][1]['cv_metrics']['CV_R2_mean']
        
        print("="*100)
        print(f"🥇 WINNER: {best_model_name} (CV R² = {best_r2:.4f})")
        
        # Best model details
        best_result = sorted_models[0][1]
        print(f"\n📋 Best Model Details:")
        print(f"   Model: {best_model_name}")
        print(f"   Cross-Validation R²: {best_result['cv_metrics']['CV_R2_mean']:.4f} ± {best_result['cv_metrics']['CV_R2_std']:.4f}")
        print(f"   Cross-Validation RMSE: {best_result['cv_metrics']['CV_RMSE_mean']:.0f} ± {best_result['cv_metrics']['CV_RMSE_std']:.0f}")
        
        if best_result['test_metrics']:
            print(f"   Test R²: {best_result['test_metrics']['R2']:.4f}")
            print(f"   Test RMSE: {best_result['test_metrics']['RMSE']:.0f}")
            print(f"   Test MAE: {best_result['test_metrics']['MAE']:.0f}")
        
        if best_result['best_params']:
            print(f"   Best Parameters: {best_result['best_params']}")
        
        print("="*100)
        
    except Exception as e:
        logger.error(f"❌ Error generating comparison report: {e}")
        raise


def create_results_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with model comparison results.
    
    Args:
        results (Dict[str, Dict[str, Any]]): Results from train_and_evaluate_models
        
    Returns:
        pd.DataFrame: DataFrame with model comparison metrics
    """
    try:
        logger.info("📊 Creating results DataFrame...")
        
        data = []
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'CV_R2_Mean': result['cv_metrics']['CV_R2_mean'],
                'CV_R2_Std': result['cv_metrics']['CV_R2_std'],
                'CV_RMSE_Mean': result['cv_metrics']['CV_RMSE_mean'],
                'CV_RMSE_Std': result['cv_metrics']['CV_RMSE_std'],
                'CV_MAE_Mean': result['cv_metrics']['CV_MAE_mean'],
                'CV_MAE_Std': result['cv_metrics']['CV_MAE_std'],
                'Training_Time': result['training_time']
            }
            
            # Add test metrics if available
            if result['test_metrics']:
                row.update({
                    'Test_R2': result['test_metrics']['R2'],
                    'Test_RMSE': result['test_metrics']['RMSE'],
                    'Test_MAE': result['test_metrics']['MAE'],
                    'Test_MAPE': result['test_metrics']['MAPE'],
                    'Test_Median_AE': result['test_metrics']['Median_AE']
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('CV_R2_Mean', ascending=False).reset_index(drop=True)
        
        logger.info("✅ Results DataFrame created")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error creating results DataFrame: {e}")
        raise


def save_results_report(
    results: Dict[str, Dict[str, Any]], 
    output_dir: str = "reports",
    include_timestamp: bool = True
) -> str:
    """
    Save comprehensive results report to files.
    
    Args:
        results (Dict[str, Dict[str, Any]]): Results from train_and_evaluate_models
        output_dir (str): Directory to save reports
        include_timestamp (bool): Whether to include timestamp in filenames
        
    Returns:
        str: Path to the main report file
    """
    try:
        logger.info("📄 Saving results report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        
        # Create results DataFrame
        df = create_results_dataframe(results)
        
        # Save CSV
        csv_filename = f"model_comparison_{timestamp}.csv" if timestamp else "model_comparison.csv"
        csv_path = output_path / csv_filename
        df.to_csv(csv_path, index=False)
        
        # Save detailed results as pickle
        pickle_filename = f"detailed_results_{timestamp}.pkl" if timestamp else "detailed_results.pkl"
        pickle_path = output_path / pickle_filename
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Create text report
        txt_filename = f"model_report_{timestamp}.txt" if timestamp else "model_report.txt"
        txt_path = output_path / txt_filename
        
        with open(txt_path, 'w') as f:
            f.write("MODEL TRAINING REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Models: {len(results)}\n\n")
            
            # Best model info
            best_name, _, best_result = get_best_model(results)
            f.write(f"Best Model: {best_name}\n")
            f.write(f"Best CV R²: {best_result['cv_metrics']['CV_R2_mean']:.4f}\n")
            f.write(f"Best CV RMSE: {best_result['cv_metrics']['CV_RMSE_mean']:.0f}\n\n")
            
            # All model results
            f.write("All Model Results:\n")
            f.write("-" * 30 + "\n")
            for model_name, result in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  CV R²: {result['cv_metrics']['CV_R2_mean']:.4f} ± {result['cv_metrics']['CV_R2_std']:.4f}\n")
                f.write(f"  CV RMSE: {result['cv_metrics']['CV_RMSE_mean']:.0f} ± {result['cv_metrics']['CV_RMSE_std']:.0f}\n")
                f.write(f"  Training Time: {result['training_time']:.2f}s\n")
                if result['best_params']:
                    f.write(f"  Best Parameters: {result['best_params']}\n")
        
        logger.info(f"✅ Reports saved to {output_path}")
        logger.info(f"   CSV: {csv_path}")
        logger.info(f"   Pickle: {pickle_path}")
        logger.info(f"   Text: {txt_path}")
        
        return str(txt_path)
        
    except Exception as e:
        logger.error(f"❌ Error saving results report: {e}")
        raise


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_models_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    cv: int = 5,
    search_method: str = "randomized",
    n_iter: int = 50,
    save_best: bool = True,
    save_reports: bool = True,
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Complete model training pipeline with evaluation, comparison, and saving.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test target
        cv (int): Cross-validation folds
        search_method (str): 'grid' or 'randomized' search
        n_iter (int): Iterations for randomized search
        save_best (bool): Whether to save the best model
        save_reports (bool): Whether to save comparison reports
        models_dir (str): Directory for saving models
        reports_dir (str): Directory for saving reports
        
    Returns:
        Dict[str, Any]: Complete pipeline results
    """
    try:
        logger.info("🚀 Starting complete model training pipeline...")
        print("\n" + "="*80)
        print("🎯 AUTOMATED ML MODEL TRAINING PIPELINE")
        print("="*80)
        
        # Step 1: Train and evaluate all models
        print("\n📈 Step 1: Training and evaluating models...")
        results = train_and_evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            search_method=search_method,
            n_iter=n_iter,
            verbose=True
        )
        
        # Step 2: Generate comparison report
        print("\n📊 Step 2: Generating model comparison...")
        print_model_comparison(results)
        
        # Step 3: Identify best model
        print("\n🏆 Step 3: Identifying best model...")
        best_name, best_model, best_result = get_best_model(results)
        
        # Step 4: Save best model
        best_model_path = None
        if save_best:
            print(f"\n💾 Step 4: Saving best model ({best_name})...")
            model_info = {
                'model_name': best_name,
                'cv_r2_score': best_result['cv_metrics']['CV_R2_mean'],
                'cv_rmse_score': best_result['cv_metrics']['CV_RMSE_mean'],
                'best_parameters': best_result['best_params'],
                'training_date': datetime.now().isoformat()
            }
            
            best_model_path = save_best_model(
                model=best_model,
                filename=f"best_{best_name.lower()}",
                model_info=model_info,
                models_dir=models_dir
            )
        
        # Step 5: Save reports
        report_path = None
        if save_reports:
            print("\n📄 Step 5: Saving comparison reports...")
            report_path = save_results_report(results, reports_dir)
        
        # Pipeline summary
        pipeline_result = {
            'results': results,
            'best_model_name': best_name,
            'best_model': best_model,
            'best_model_info': best_result,
            'best_model_path': best_model_path,
            'report_path': report_path,
            'training_summary': {
                'total_models': len(results),
                'best_cv_r2': best_result['cv_metrics']['CV_R2_mean'],
                'best_cv_rmse': best_result['cv_metrics']['CV_RMSE_mean'],
                'total_training_time': sum(r['training_time'] for r in results.values()),
                'search_method': search_method,
                'cv_folds': cv
            }
        }
        
        print("\n✅ Training pipeline completed successfully!")
        print(f"🎯 Best Model: {best_name} (CV R² = {best_result['cv_metrics']['CV_R2_mean']:.4f})")
        print("="*80)
        
        return pipeline_result
        
    except Exception as e:
        logger.error(f"❌ Error in training pipeline: {e}")
        raise


# ============================================================================
# UTILITY FUNCTIONS AND EXAMPLES
# ============================================================================

def quick_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Tuple[str, Any]:
    """
    Quick training function for fast model selection (legacy compatibility).
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test target
        
    Returns:
        Tuple[str, Any]: Best model name and model object
    """
    try:
        logger.info("⚡ Quick training mode...")
        
        # Use fast settings
        results = train_and_evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=3,  # Faster CV
            search_method="randomized",
            n_iter=10,  # Fewer iterations
            verbose=False
        )
        
        # Get best model
        best_name, best_model, _ = get_best_model(results)
        
        logger.info(f"✅ Quick training completed. Best: {best_name}")
        return best_name, best_model
        
    except Exception as e:
        logger.error(f"❌ Error in quick training: {e}")
        raise


def train_specific_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_names: List[str],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Train only specific models instead of all available models.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        model_names (List[str]): List of model names to train
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test target
        **kwargs: Additional arguments for train_and_evaluate_models
        
    Returns:
        Dict[str, Dict[str, Any]]: Results for specified models only
    """
    try:
        logger.info(f"🎯 Training specific models: {model_names}")
        
        # Get all models and filter
        all_models = get_models()
        selected_models = {name: all_models[name] for name in model_names if name in all_models}
        
        if not selected_models:
            raise ValueError(f"No valid models found in {model_names}")
        
        # Temporarily replace get_models function
        original_get_models = globals()['get_models']
        globals()['get_models'] = lambda: selected_models
        
        try:
            # Train selected models
            results = train_and_evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                **kwargs
            )
        finally:
            # Restore original function
            globals()['get_models'] = original_get_models
        
        logger.info(f"✅ Training completed for {len(results)} models")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error training specific models: {e}")
        raise


def evaluate_single_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Evaluate a single pre-trained model.
    
    Args:
        model (Any): Trained model object
        model_name (str): Name of the model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray, optional): Test features
        y_test (np.ndarray, optional): Test target
        cv (int): Cross-validation folds
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    try:
        logger.info(f"📊 Evaluating single model: {model_name}")
        
        # Train metrics
        y_train_pred = model.predict(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred)
        
        # Test metrics
        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Cross-validation
        detailed_cv = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            n_jobs=-1,
            return_train_score=True
        )
        
        cv_metrics = {
            'CV_RMSE_mean': np.sqrt(-detailed_cv['test_neg_mean_squared_error'].mean()),
            'CV_RMSE_std': np.sqrt(-detailed_cv['test_neg_mean_squared_error']).std(),
            'CV_MAE_mean': -detailed_cv['test_neg_mean_absolute_error'].mean(),
            'CV_MAE_std': (-detailed_cv['test_neg_mean_absolute_error']).std(),
            'CV_R2_mean': detailed_cv['test_r2'].mean(),
            'CV_R2_std': detailed_cv['test_r2'].std()
        }
        
        result = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'model_name': model_name
        }
        
        logger.info(f"✅ Model evaluation completed: CV R² = {cv_metrics['CV_R2_mean']:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        raise


# =========================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """
    Example: Basic model training usage
    """
    # Assuming you have preprocessed data
    # X_train, X_test, y_train, y_test = your_preprocessing_function()
    
    print("📚 Example: Basic Model Training")
    print("="*50)
    
    # Complete training pipeline
    pipeline_result = train_models_pipeline(
        X_train=None,  # Replace with your data
        y_train=None,  # Replace with your data
        X_test=None,   # Replace with your data
        y_test=None,   # Replace with your data
        cv=5,
        search_method="randomized",
        n_iter=20,
        save_best=True,
        save_reports=True
    )
    
    # Access results
    best_model = pipeline_result['best_model']
    best_name = pipeline_result['best_model_name']
    
    print(f"Best model: {best_name}")
    print(f"Model path: {pipeline_result['best_model_path']}")
    
    return pipeline_result


def example_custom_training():
    """
    Example: Custom training with specific models
    """
    print("📚 Example: Custom Model Training")
    print("="*50)
    
    # Train only specific models
    selected_models = ["Ridge", "Random_Forest", "XGBoost"]
    
    results = train_specific_models(
        X_train=None,  # Replace with your data
        y_train=None,  # Replace with your data
        model_names=selected_models,
        cv=5,
        search_method="grid",
        verbose=True
    )
    
    # Get best from selected models
    best_name, best_model, best_info = get_best_model(results)
    
    # Save manually
    save_best_model(
        model=best_model,
        filename=f"custom_{best_name}",
        model_info={'selection': 'custom', 'models': selected_models}
    )
    
    return results


def example_quick_training():
    """
    Example: Quick training for fast prototyping
    """
    print("📚 Example: Quick Training")
    print("="*50)
    
    # Quick training with minimal hyperparameter search
    best_name, best_model = quick_train(
        X_train=None,  # Replace with your data
        y_train=None,  # Replace with your data
        X_test=None,   # Replace with your data
        y_test=None    # Replace with your data
    )
    
    print(f"Quick best model: {best_name}")
    
    # Evaluate the quick model
    evaluation = evaluate_single_model(
        model=best_model,
        model_name=best_name,
        X_train=None,  # Replace with your data
        y_train=None,  # Replace with your data
        cv=3
    )
    
    return best_model, evaluation


if __name__ == "__main__":
    """
    Main execution block for testing the training module
    """
    try:
        logger.info("🧪 Testing model training module...")
        
        print("="*80)
        print("🚀 MODEL TRAINING MODULE - PRODUCTION READY")
        print("="*80)
        
        # Display available models
        models = get_models()
        print(f"\n📋 Available Models: {list(models.keys())}")
        
        # Display hyperparameter info
        hyperparams = get_hyperparams()
        print(f"⚙️ Models with Hyperparameter Tuning: {list(hyperparams.keys())}")
        
        print("\n✅ Module ready for use!")
        print("💡 Import this module and use train_models_pipeline() for complete training")
        
    except Exception as e:
        logger.error(f"❌ Error testing training module: {e}")
        raise
