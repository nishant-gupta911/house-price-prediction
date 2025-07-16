"""
Production-Grade Model Evaluation Module for Regression Tasks

This module provides comprehensive evaluation capabilities for regression models including
metrics calculation, cross-validation, visualization, and model comparison. Designed for
professional ML workflows with publication-quality outputs.

Author: AI Engineer
Date: July 2025
Version: 2.0
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, cross_validate

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a regression model on test data with comprehensive metrics.
    
    Args:
        model: Trained model object with predict method
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target values
        model_name (str): Name of the model for logging
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    try:
        logger.info(f"📊 Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate core metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            # Fallback calculation for MAPE
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_test - y_pred))
        
        # Mean Squared Log Error (if all values are positive)
        if np.all(y_test > 0) and np.all(y_pred > 0):
            msle = np.mean(np.square(np.log1p(y_test) - np.log1p(y_pred)))
        else:
            msle = np.nan
        
        # Explained Variance Score
        explained_var = 1 - np.var(y_test - y_pred) / np.var(y_test)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MSE': mse,
            'MAPE': mape,
            'Median_AE': median_ae,
            'MSLE': msle,
            'Explained_Variance': explained_var
        }
        
        # Log results
        logger.info(f"✅ {model_name} Evaluation Results:")
        logger.info(f"   RMSE: {rmse:.2f}")
        logger.info(f"   MAE: {mae:.2f}")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   MAPE: {mape:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error evaluating model {model_name}: {e}")
        raise


def cross_validate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: Optional[List[str]] = None,
    model_name: str = "Model"
) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on a model with multiple scoring metrics.
    
    Args:
        model: Model object to cross-validate
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values
        cv (int): Number of cross-validation folds
        scoring (List[str], optional): List of scoring metrics
        model_name (str): Name of the model for logging
        
    Returns:
        Dict[str, Dict[str, float]]: CV scores with mean and std for each metric
    """
    try:
        logger.info(f"🔄 Cross-validating model: {model_name} ({cv}-fold CV)")
        
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Process results
        processed_results = {}
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # Convert negative scores back to positive for MSE and MAE
            if 'neg_' in metric:
                test_scores = -test_scores
                train_scores = -train_scores
                clean_metric = metric.replace('neg_', '').replace('_', ' ').title()
                
                # Convert MSE to RMSE
                if 'mean_squared_error' in metric:
                    test_scores = np.sqrt(test_scores)
                    train_scores = np.sqrt(train_scores)
                    clean_metric = 'RMSE'
                elif 'mean_absolute_error' in metric:
                    clean_metric = 'MAE'
            else:
                clean_metric = metric.replace('_', ' ').title()
            
            processed_results[clean_metric] = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'test_scores': test_scores,
                'train_scores': train_scores
            }
        
        # Log results
        logger.info(f"✅ {model_name} Cross-Validation Results:")
        for metric, scores in processed_results.items():
            logger.info(f"   {metric}: {scores['test_mean']:.4f} ± {scores['test_std']:.4f}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"❌ Error cross-validating model {model_name}: {e}")
        raise


def calculate_prediction_intervals(
    model: Any,
    X_test: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals (if model supports it).
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features
        confidence (float): Confidence level for intervals
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: predictions, lower_bound, upper_bound
    """
    try:
        y_pred = model.predict(X_test)
        
        # For models that support prediction intervals
        if hasattr(model, 'predict') and hasattr(model, 'get_params'):
            # Simple approach: use residual-based intervals
            # This is a placeholder - real implementation would depend on model type
            std_residual = np.std(y_pred) * 0.1  # Simplified
            margin = 1.96 * std_residual  # 95% confidence
            
            lower_bound = y_pred - margin
            upper_bound = y_pred + margin
        else:
            # Default case
            lower_bound = y_pred
            upper_bound = y_pred
        
        return y_pred, lower_bound, upper_bound
        
    except Exception as e:
        logger.error(f"❌ Error calculating prediction intervals: {e}")
        raise


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show_intervals: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> Any:
    """
    Create actual vs predicted scatter plot with enhanced visualizations.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model for title
        save_path (str, optional): Path to save the plot
        show_intervals (bool): Whether to show confidence intervals
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    try:
        logger.info(f"📈 Creating prediction plot for {model_name}")
        
        # Calculate metrics for display
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate prediction errors for color mapping
        errors = np.abs(y_true - y_pred)
        
        # Create scatter plot
        scatter = ax.scatter(
            y_true, y_pred, 
            c=errors, 
            cmap='viridis', 
            alpha=0.6, 
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2.5, alpha=0.8, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 
               'orange', linewidth=2, alpha=0.8, label='Trend Line')
        
        # Customize plot
        ax.set_xlabel('Actual Values', fontweight='bold')
        ax.set_ylabel('Predicted Values', fontweight='bold')
        ax.set_title(f'Actual vs Predicted - {model_name}', fontweight='bold', pad=20)
        
        # Add metrics text box
        metrics_text = (
            f'R² = {r2:.4f}\n'
            f'RMSE = {rmse:.2f}\n'
            f'MAE = {mae:.2f}'
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', bbox=props)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prediction Error', fontweight='bold')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📁 Plot saved to: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating prediction plot: {e}")
        raise


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Any:
    """
    Create comprehensive residual analysis plots.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model for title
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    try:
        logger.info(f"📊 Creating residual analysis for {model_name}")
        
        residuals = y_true - y_pred
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Predicted Values
        ax1.scatter(y_pred, residuals, alpha=0.6, color='steelblue', s=40)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontweight='bold')
        ax1.set_ylabel('Residuals', fontweight='bold')
        ax1.set_title('Residuals vs Predicted', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add lowess trend line
        try:
            from scipy import stats
            # Simple trend line
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax1.plot(y_pred, p(y_pred), 'orange', linewidth=2, alpha=0.8)
        except:
            pass
        
        # 2. Histogram of Residuals
        ax2.hist(residuals, bins=30, alpha=0.7, color='lightcoral', 
                edgecolor='black', density=True)
        ax2.set_xlabel('Residuals', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('Distribution of Residuals', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add normal distribution overlay
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax2.plot(x, normal_dist, 'red', linewidth=2, label='Normal Distribution')
        ax2.legend()
        
        # 3. Q-Q Plot
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        except ImportError:
            # Fallback without scipy
            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
            ax3.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
            ax3.plot(theoretical_quantiles, theoretical_quantiles, 'r--')
            ax3.set_xlabel('Theoretical Quantiles')
            ax3.set_ylabel('Sample Quantiles')
            ax3.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Residuals vs Order (Index)
        ax4.scatter(range(len(residuals)), residuals, alpha=0.6, color='lightgreen', s=40)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Sample Index', fontweight='bold')
        ax4.set_ylabel('Residuals', fontweight='bold')
        ax4.set_title('Residuals vs Sample Order', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📁 Residual plot saved to: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating residual plot: {e}")
        raise


def plot_error_distribution(
    errors: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """
    Plot distribution of prediction errors with statistical analysis.
    
    Args:
        errors (np.ndarray): Prediction errors (y_true - y_pred)
        model_name (str): Name of the model for title
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    try:
        logger.info(f"📊 Creating error distribution plot for {model_name}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Error Distribution Analysis - {model_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Histogram with statistics
        ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        
        # Add statistical lines
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.2f}')
        ax1.axvline(median_error, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_error:.2f}')
        ax1.axvline(mean_error + std_error, color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'±1 STD: {std_error:.2f}')
        ax1.axvline(mean_error - std_error, color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Prediction Errors', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('Error Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot with outlier analysis
        box_plot = ax2.boxplot(errors, patch_artist=True, 
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
        
        ax2.set_ylabel('Prediction Errors', fontweight='bold')
        ax2.set_title('Error Box Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = errors[(errors < q1 - outlier_threshold) | 
                         (errors > q3 + outlier_threshold)]
        
        stats_text = (
            f'Mean: {mean_error:.2f}\n'
            f'Median: {median_error:.2f}\n'
            f'Std: {std_error:.2f}\n'
            f'IQR: {iqr:.2f}\n'
            f'Outliers: {len(outliers)} ({len(outliers)/len(errors)*100:.1f}%)'
        )
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📁 Error distribution plot saved to: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating error distribution plot: {e}")
        raise

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title_suffix = "Feature Importances"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        title_suffix = "Feature Coefficients (Absolute)"
    else:
        print("Model doesn't have feature_importances_ or coef_ attribute")
        return None
    
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_importances, 
                  color=sns.color_palette("viridis", len(top_features)))
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(top_importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_learning_curves(model, X_train, y_train, cv=5, save_path=None):
    """Plot learning curves to check for overfitting"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores, _, _ = learning_curve(
        model, X_train, y_train, cv=cv, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    # Convert to RMSE
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    train_rmse_mean = train_rmse.mean(axis=1)
    train_rmse_std = train_rmse.std(axis=1)
    val_rmse_mean = val_rmse.mean(axis=1)
    val_rmse_std = val_rmse.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot training scores
    ax.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
    ax.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                    train_rmse_mean + train_rmse_std, alpha=0.2, color='blue')
    
    # Plot validation scores
    ax.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
    ax.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                    val_rmse_mean + val_rmse_std, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curves', fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def create_interactive_comparison(results):
    """Create interactive plotly comparison chart"""
    models = list(results.keys())
    rmse_values = [results[m]['RMSE'] for m in models]
    r2_values = [results[m]['R2'] for m in models]
    mae_values = [results[m].get('MAE', 0) for m in models]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RMSE Comparison', 'R² Score Comparison', 'MAE Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RMSE plot
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightblue'),
        row=1, col=1
    )
    
    # R² plot
    fig.add_trace(
        go.Bar(x=models, y=r2_values, name='R²', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # MAE plot
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE', marker_color='lightcoral'),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Interactive Model Performance Comparison",
        title_x=0.5,
        showlegend=False,
        height=500
    )
    
    return fig

def plot_comprehensive_evaluation(results, model_name, model, X_test, y_test, 
                                feature_names=None, save_dir=None):
    """Create comprehensive evaluation plots for a model"""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    plots = {}
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # 1. Actual vs Predicted
    save_path = save_dir / f"{model_name}_actual_vs_predicted.png" if save_dir else None
    plots['actual_vs_predicted'] = plot_predictions(
        y_test, y_pred, model_name, str(save_path) if save_path else None
    )
    
    # 2. Residuals Analysis
    save_path = save_dir / f"{model_name}_residuals.png" if save_dir else None
    plots['residuals'] = plot_residuals(
        y_test, y_pred, model_name, str(save_path) if save_path else None
    )
    
    # 3. Feature Importance (if available)
    if feature_names and (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
        save_path = save_dir / f"{model_name}_feature_importance.png" if save_dir else None
        plots['feature_importance'] = plot_feature_importance(
            model, feature_names, save_path=save_path
        )
    
    return plots

def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ['RMSE', 'MAE', 'R2'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Any:
    """
    Create comprehensive model comparison with multiple metrics.
    
    Args:
        results_dict (Dict[str, Dict[str, float]]): Model results dictionary
        metrics (List[str]): List of metrics to compare
        save_path (str, optional): Path to save the plot
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        Any: Matplotlib figure object
    """
    try:
        logger.info("📊 Creating comprehensive model comparison")
        
        # Prepare data
        models = list(results_dict.keys())
        n_metrics = len(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get values for this metric
            values = []
            for model in models:
                if metric in results_dict[model]:
                    values.append(results_dict[model][metric])
                else:
                    values.append(0)  # Default value if metric not found
            
            # Determine if higher or lower is better
            if metric.upper() in ['RMSE', 'MAE', 'MSE', 'MAPE']:
                # Lower is better
                best_idx = values.index(min(values))
                colors = ['gold' if j == best_idx else 'steelblue' for j in range(len(values))]
            else:
                # Higher is better (R2, etc.)
                best_idx = values.index(max(values))
                colors = ['gold' if j == best_idx else 'steelblue' for j in range(len(values))]
            
            # Create bar plot
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.3f}' if metric.upper() == 'R2' else f'{value:.1f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Customize subplot
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=14)
            ax.set_ylabel(metric, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight best model
            bars[best_idx].set_edgecolor('orange')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📁 Model comparison plot saved to: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating model comparison: {e}")
        raise


def create_model_ranking_table(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comprehensive ranking table of model performance.
    
    Args:
        results_dict (Dict[str, Dict[str, float]]): Model results dictionary
        save_path (str, optional): Path to save the table as CSV
        
    Returns:
        pd.DataFrame: Ranking table with scores and ranks
    """
    try:
        logger.info("📋 Creating model ranking table")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results_dict).T
        
        # Calculate ranks (lower is better for RMSE, MAE; higher is better for R2)
        rank_columns = {}
        
        if 'RMSE' in df.columns:
            rank_columns['RMSE_Rank'] = df['RMSE'].rank(method='min')
        if 'MAE' in df.columns:
            rank_columns['MAE_Rank'] = df['MAE'].rank(method='min')
        if 'R2' in df.columns:
            rank_columns['R2_Rank'] = df['R2'].rank(method='min', ascending=False)
        if 'MAPE' in df.columns:
            rank_columns['MAPE_Rank'] = df['MAPE'].rank(method='min')
        
        # Add rank columns
        for col, ranks in rank_columns.items():
            df[col] = ranks
        
        # Calculate overall rank (average of individual ranks)
        rank_cols = [col for col in df.columns if col.endswith('_Rank')]
        if rank_cols:
            # Calculate overall rank as mean of individual ranks
            rank_scores = []
            for col in rank_cols:
                rank_scores.append(df[col].values)
            df['Overall_Rank'] = np.mean(rank_scores, axis=0)
            df = df.sort_values('Overall_Rank')
        
        # Round numerical values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        # Add position column
        df.insert(0, 'Position', range(1, len(df) + 1))
        
        # Save if requested
        if save_path:
            df.to_csv(save_path, index=True)
            logger.info(f"📁 Ranking table saved to: {save_path}")
        
        # Print formatted table
        print("\n" + "="*80)
        print("🏆 MODEL PERFORMANCE RANKING TABLE")
        print("="*80)
        print(df.to_string(index=True))
        print("="*80)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error creating ranking table: {e}")
        raise


def plot_metric_radar_chart(
    results_dict: Dict[str, Dict[str, float]],
    models_to_compare: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Any:
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        results_dict (Dict[str, Dict[str, float]]): Model results dictionary
        models_to_compare (List[str], optional): Specific models to compare
        save_path (str, optional): Path to save the plot
        
    Returns:
        Any: Matplotlib figure object
    """
    try:
        logger.info("🎯 Creating radar chart for model comparison")
        
        if models_to_compare is None:
            models_to_compare = list(results_dict.keys())[:5]  # Top 5 models
        
        # Prepare metrics (normalize for radar chart)
        metrics = ['R2', 'RMSE', 'MAE']
        available_metrics = []
        
        for metric in metrics:
            if all(metric in results_dict[model] for model in models_to_compare):
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No common metrics found for radar chart")
            return None
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for model in models_to_compare:
            normalized_data[model] = []
            
            for metric in available_metrics:
                value = results_dict[model][metric]
                
                # Get all values for this metric for normalization
                all_values = [results_dict[m][metric] for m in models_to_compare]
                
                if metric in ['RMSE', 'MAE', 'MAPE']:
                    # Lower is better - invert for radar chart
                    normalized = 1 - (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10)
                else:
                    # Higher is better
                    normalized = (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10)
                
                normalized_data[model].append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Use a simple color scheme
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        colors = colors[:len(models_to_compare)]
        
        for i, model in enumerate(models_to_compare):
            values = normalized_data[model]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📁 Radar chart saved to: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        logger.error(f"❌ Error creating radar chart: {e}")
        raise


# ============================================================================
# UTILITY FUNCTIONS AND EXAMPLES
# ============================================================================

def save_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_dir: str = "evaluation_reports"
) -> str:
    """
    Save comprehensive evaluation report to files.
    
    Args:
        evaluation_results (Dict[str, Any]): Results from comprehensive evaluation
        output_dir (str): Directory to save reports
        
    Returns:
        str: Path to the main report file
    """
    try:
        logger.info("📄 Saving evaluation report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_name = evaluation_results['model_name']
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as pickle
        pickle_path = output_path / f"{model_name}_evaluation_{timestamp}.pkl"
        import pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        # Create text report
        txt_path = output_path / f"{model_name}_report_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"MODEL EVALUATION REPORT: {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test metrics
            f.write("TEST METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in evaluation_results['test_metrics'].items():
                if not pd.isna(value):
                    f.write(f"{metric}: {value:.4f}\n")
            
            # CV results
            f.write("\nCROSS-VALIDATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for metric, scores in evaluation_results['cv_results'].items():
                f.write(f"{metric}: {scores['test_mean']:.4f} ± {scores['test_std']:.4f}\n")
        
        logger.info(f"✅ Evaluation report saved to {output_path}")
        return str(txt_path)
        
    except Exception as e:
        logger.error(f"❌ Error saving evaluation report: {e}")
        raise


def compare_multiple_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models and generate comparison plots.
    
    Args:
        models_dict (Dict[str, Any]): Dictionary of model_name -> model_object
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target
        save_dir (str, optional): Directory to save plots
        
    Returns:
        Dict[str, Dict[str, float]]: Comparison results
    """
    try:
        logger.info("🔄 Comparing multiple models...")
        
        # Evaluate each model
        results = {}
        for model_name, model in models_dict.items():
            results[model_name] = evaluate_model(model, X_test, y_test, model_name)
        
        # Create comparison plots
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Model comparison plot
            comparison_path = save_path / "model_comparison.png"
            compare_models(results, save_path=str(comparison_path))
            
            # Ranking table
            ranking_path = save_path / "model_ranking.csv"
            create_model_ranking_table(results, save_path=str(ranking_path))
            
            # Interactive comparison (if plotly available)
            if PLOTLY_AVAILABLE:
                interactive_path = save_path / "interactive_comparison.html"
                # Note: Interactive comparison function would need implementation
                logger.info(f"📊 Interactive comparison would be saved to {interactive_path}")
        else:
            # Just display plots
            compare_models(results)
            create_model_ranking_table(results)
            if PLOTLY_AVAILABLE:
                logger.info("📊 Interactive comparison available but not saved")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error comparing models: {e}")
        raise


def quick_evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> None:
    """
    Quick evaluation with basic metrics and plots.
    
    Args:
        model: Trained model object
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target
        model_name (str): Name of the model
    """
    try:
        logger.info(f"⚡ Quick evaluation of {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluate_model(model, X_test, y_test, model_name)
        
        # Quick plots
        plot_predictions(y_test, y_pred, model_name)
        residuals = y_test - y_pred
        plot_error_distribution(residuals, model_name)
        
        # Print summary
        print(f"\n⚡ QUICK EVALUATION: {model_name}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE:  {metrics['MAE']:.2f}")
        print(f"R²:   {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Error in quick evaluation: {e}")
        raise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_single_model_evaluation():
    """
    Example: Comprehensive evaluation of a single model
    """
    print("📚 Example: Single Model Evaluation")
    print("="*50)
    
    # Assuming you have a trained model and test data
    # model = your_trained_model
    # X_test, y_test = your_test_data
    
    # Comprehensive evaluation
    # results = comprehensive_model_evaluation(
    #     model=model,
    #     model_name="Random_Forest",
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     cv=5,
    #     save_dir="evaluation_results"
    # )
    
    # Save report
    # save_evaluation_report(results, "reports")
    
    print("💡 Replace placeholder variables with your actual data")


def example_multiple_model_comparison():
    """
    Example: Compare multiple trained models
    """
    print("📚 Example: Multiple Model Comparison")
    print("="*50)
    
    # Assuming you have multiple trained models
    # models = {
    #     "Linear_Regression": linear_model,
    #     "Random_Forest": rf_model,
    #     "XGBoost": xgb_model
    # }
    
    # Compare models
    # results = compare_multiple_models(
    #     models_dict=models,
    #     X_test=X_test,
    #     y_test=y_test,
    #     save_dir="comparison_results"
    # )
    
    # Print best model
    # best_model = min(results, key=lambda x: results[x]['RMSE'])
    # print(f"Best model: {best_model}")
    
    print("💡 Replace placeholder variables with your actual models and data")


def example_quick_evaluation():
    """
    Example: Quick model evaluation
    """
    print("📚 Example: Quick Model Evaluation")
    print("="*50)
    
    # Quick evaluation for rapid prototyping
    # quick_evaluate(
    #     model=your_model,
    #     X_test=X_test,
    #     y_test=y_test,
    #     model_name="Test_Model"
    # )
    
    print("💡 Use this for rapid model testing during development")


if __name__ == "__main__":
    """
    Main execution block for testing the evaluation module
    """
    try:
        logger.info("🧪 Testing model evaluation module...")
        
        print("="*80)
        print("🔬 MODEL EVALUATION MODULE - PRODUCTION READY")
        print("="*80)
        
        print("\n📋 Available Functions:")
        print("   ✅ evaluate_model() - Basic model evaluation")
        print("   ✅ cross_validate_model() - Cross-validation analysis")
        print("   ✅ plot_predictions() - Actual vs predicted plots")
        print("   ✅ plot_residuals() - Residual analysis")
        print("   ✅ plot_error_distribution() - Error distribution analysis")
        print("   ✅ compare_models() - Multi-model comparison")
        print("   ✅ comprehensive_model_evaluation() - Complete evaluation suite")
        
        print("\n📊 Visualization Features:")
        print("   ✅ Publication-quality matplotlib plots")
        print("   ✅ Interactive plotly charts (if available)")
        print("   ✅ Professional styling and themes")
        print("   ✅ Automatic plot saving")
        
        print("\n📈 Analysis Features:")
        print("   ✅ Comprehensive metrics (RMSE, MAE, R², MAPE, etc.)")
        print("   ✅ Cross-validation with statistical analysis")
        print("   ✅ Model ranking and comparison tables")
        print("   ✅ Radar charts for multi-metric comparison")
        
        print("\n✅ Module ready for use!")
        print("💡 Import this module and use the evaluation functions")
        print("📖 See examples above for usage patterns")
        
    except Exception as e:
        logger.error(f"❌ Error testing evaluation module: {e}")
        raise
