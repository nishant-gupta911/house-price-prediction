"""
Utility functions for House Price Prediction Project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import os
import json
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "house_price_prediction", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name (str): Logger name
        level (int): Logging level
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# DATA LOADING/SAVING UTILITIES
# ============================================================================

def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load CSV data with comprehensive error handling.
    
    Args:
        filepath (Union[str, Path]): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv()
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or corrupted
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Data file not found: {filepath}")
        
        if filepath.stat().st_size == 0:
            raise ValueError(f"❌ Data file is empty: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath, **kwargs)
        
        if df.empty:
            raise ValueError(f"❌ Loaded DataFrame is empty: {filepath}")
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Successfully loaded data from {filepath}")
        logger.info(f"📊 Data shape: {df.shape}")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"❌ CSV file is empty or corrupted: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"❌ Error parsing CSV file {filepath}: {str(e)}")
    except Exception as e:
        raise Exception(f"❌ Unexpected error loading data from {filepath}: {str(e)}")

def save_data(df: pd.DataFrame, 
              filepath: Union[str, Path], 
              create_dirs: bool = True,
              **kwargs) -> None:
    """
    Save DataFrame to CSV with error handling.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (Union[str, Path]): Output file path
        create_dirs (bool): Create directories if they don't exist
        **kwargs: Additional arguments for df.to_csv()
    """
    try:
        filepath = Path(filepath)
        
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Default arguments
        if 'index' not in kwargs:
            kwargs['index'] = False
        
        df.to_csv(filepath, **kwargs)
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Data saved to {filepath}")
        logger.info(f"📊 Saved {df.shape[0]} rows and {df.shape[1]} columns")
        
    except PermissionError:
        raise PermissionError(f"❌ Permission denied: Cannot write to {filepath}")
    except Exception as e:
        raise Exception(f"❌ Error saving data to {filepath}: {str(e)}")

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model: Any, 
               filepath: Union[str, Path],
               preprocessor: Optional[Any] = None,
               metadata: Optional[Dict[str, Any]] = None,
               create_dirs: bool = True) -> None:
    """
    Save model and optional preprocessor to disk with metadata.
    
    Args:
        model: Trained model object
        filepath (Union[str, Path]): Path to save the model
        preprocessor: Optional preprocessor object
        metadata (Dict[str, Any], optional): Additional metadata
        create_dirs (bool): Create directories if they don't exist
    """
    try:
        filepath = Path(filepath)
        
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'metadata': metadata or {},
            'saved_at': timestamp(),
            'model_type': type(model).__name__
        }
        
        # Save using joblib
        joblib.dump(model_data, filepath)
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Model saved to {filepath}")
        logger.info(f"🤖 Model type: {type(model).__name__}")
        
    except Exception as e:
        raise Exception(f"❌ Error saving model to {filepath}: {str(e)}")

def load_model(filepath: Union[str, Path]) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """
    Load model, preprocessor, and metadata from disk.
    
    Args:
        filepath (Union[str, Path]): Path to model file
        
    Returns:
        Tuple[Any, Optional[Any], Dict[str, Any]]: Model, preprocessor, metadata
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Handle legacy format (for backward compatibility)
        if isinstance(model_data, dict):
            model = model_data.get('model')
            preprocessor = model_data.get('preprocessor')
            metadata = model_data.get('metadata', {})
        else:
            # Assume it's just the model
            model = model_data
            preprocessor = None
            metadata = {}
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Model loaded from {filepath}")
        logger.info(f"🤖 Model type: {type(model).__name__}")
        
        return model, preprocessor, metadata
        
    except Exception as e:
        raise Exception(f"❌ Error loading model from {filepath}: {str(e)}")

# ============================================================================
# METRICS AND LOGGING
# ============================================================================

def log_metrics(metrics: Dict[str, Any], 
                filepath: Union[str, Path],
                format_type: str = 'json',
                create_dirs: bool = True) -> None:
    """
    Save evaluation metrics to file.
    
    Args:
        metrics (Dict[str, Any]): Metrics dictionary
        filepath (Union[str, Path]): Output file path
        format_type (str): 'json' or 'csv'
        create_dirs (bool): Create directories if they don't exist
    """
    try:
        filepath = Path(filepath)
        
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to metrics
        metrics_with_timestamp = {
            **metrics,
            'logged_at': timestamp(),
            'timestamp_iso': datetime.now().isoformat()
        }
        
        if format_type.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(metrics_with_timestamp, f, indent=2, default=str)
        elif format_type.lower() == 'csv':
            df = pd.DataFrame([metrics_with_timestamp])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("format_type must be 'json' or 'csv'")
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Metrics saved to {filepath}")
        
    except Exception as e:
        raise Exception(f"❌ Error saving metrics to {filepath}: {str(e)}")

def timestamp() -> str:
    """
    Return current timestamp string for versioning.
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_model_performance(model_name: str, 
                          metrics: Dict[str, float],
                          detailed: bool = True) -> None:
    """
    Print nicely formatted model performance metrics.
    
    Args:
        model_name (str): Name of the model
        metrics (Dict[str, float]): Performance metrics
        detailed (bool): Show detailed breakdown
    """
    print(f"\n{'='*60}")
    print(f"🏆 MODEL PERFORMANCE: {model_name.upper()}")
    print(f"{'='*60}")
    
    if detailed:
        # Main metrics
        main_metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        for metric in main_metrics:
            if metric in metrics:
                value = metrics[metric]
                if metric == 'R2':
                    print(f"📊 {metric:<15}: {value:>10.4f}")
                elif metric == 'MAPE':
                    print(f"📊 {metric:<15}: {value:>9.2f}%")
                else:
                    print(f"📊 {metric:<15}: {value:>10.0f}")
        
        # Additional metrics if available
        additional_metrics = {k: v for k, v in metrics.items() if k not in main_metrics}
        if additional_metrics:
            print(f"\n📋 Additional Metrics:")
            for metric, value in additional_metrics.items():
                print(f"   {metric:<15}: {value}")
    else:
        # Simple format
        rmse = metrics.get('RMSE', 0)
        r2 = metrics.get('R2', 0)
        print(f"RMSE: {rmse:.0f} | R²: {r2:.3f}")
    
    print(f"{'='*60}")

# ============================================================================
# DATA ANALYSIS UTILITIES
# ============================================================================

def display_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a comprehensive summary of missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Summary with missing counts and percentages
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percentage': missing_percentage.values,
        'Data_Type': df.dtypes.values
    })
    
    # Sort by missing percentage (descending)
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    # Filter to show only columns with missing values
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    
    if len(missing_summary) == 0:
        print("✅ No missing values found in the dataset!")
        return pd.DataFrame(columns=['Column', 'Missing_Count', 'Missing_Percentage', 'Data_Type'])
    
    print(f"⚠️  Found {len(missing_summary)} columns with missing values:")
    return missing_summary.reset_index(drop=True)

def create_output_dirs(base_path: Union[str, Path] = ".") -> Dict[str, Path]:
    """
    Create standard output directories for ML project.
    
    Args:
        base_path (Union[str, Path]): Base project directory
        
    Returns:
        Dict[str, Path]: Dictionary with created directory paths
    """
    base_path = Path(base_path)
    
    directories = {
        'models': base_path / 'model',
        'logs': base_path / 'logs', 
        'demo': base_path / 'demo',
        'data': base_path / 'data',
        'results': base_path / 'results',
        'plots': base_path / 'plots'
    }
    
    created_dirs = []
    for name, path in directories.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(name)
    
    logger = logging.getLogger("house_price_prediction")
    if created_dirs:
        logger.info(f"✅ Created directories: {', '.join(created_dirs)}")
    else:
        logger.info("📁 All output directories already exist")
    
    return directories

# ============================================================================
# EXISTING UTILITY FUNCTIONS (PRESERVED)
# ============================================================================

def setup_plotting_style():
    """Set up consistent plotting style across the project"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.0f}"

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """Detect outliers using IQR or Z-score method"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        try:
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column]))
            return z_scores > 3
        except ImportError:
            print("⚠️  scipy not available, falling back to IQR method")
            return detect_outliers(df, column, method='iqr')
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def create_feature_importance_plot(model, feature_names: List[str], top_n: int = 20) -> Any:
    """Create feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_features)), top_importances, color='skyblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def log_transform_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Apply log transformation to specified features"""
    df_transformed = df.copy()
    for feature in features:
        if feature in df_transformed.columns:
            # Add small constant to avoid log(0)
            df_transformed[feature] = np.log1p(df_transformed[feature])
    return df_transformed

def print_model_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary of model results"""
    print("\n" + "="*80)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort models by R2 score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R2 Score':<12} {'MAPE (%)':<12}")
    print("-"*80)
    
    for model_name, metrics in sorted_models:
        print(f"{model_name:<20} {metrics['RMSE']:<12.0f} {metrics['MAE']:<12.0f} "
              f"{metrics['R2']:<12.3f} {metrics.get('MAPE', 0):<12.2f}")
    
    best_model = sorted_models[0]
    print(f"\n🥇 Best Model: {best_model[0]} (R² = {best_model[1]['R2']:.3f})")
    print("="*80)

def create_directory_if_not_exists(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def get_model_explanation(feature_importance: np.ndarray, feature_names: List[str], 
                         prediction: float, top_n: int = 5) -> str:
    """Generate explanation for model prediction"""
    # Get top contributing features
    indices = np.argsort(np.abs(feature_importance))[::-1][:top_n]
    
    explanation = f"🏠 **Predicted Price: {format_currency(prediction)}**\n\n"
    explanation += "**Key factors influencing this prediction:**\n"
    
    for i, idx in enumerate(indices, 1):
        feature_name = feature_names[idx]
        importance = feature_importance[idx]
        impact = "positively" if importance > 0 else "negatively"
        explanation += f"{i}. **{feature_name}** - impacts price {impact}\n"
    
    return explanation

# ============================================================================
# ADVANCED UTILITY FUNCTIONS
# ============================================================================

def validate_data_integrity(df: pd.DataFrame, 
                           required_columns: Optional[List[str]] = None,
                           target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate data integrity and return detailed report.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str], optional): Required column names
        target_column (str, optional): Target column name
        
    Returns:
        Dict[str, Any]: Validation report
    """
    report = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'summary': {}
    }
    
    # Basic checks
    if df.empty:
        report['is_valid'] = False
        report['issues'].append("DataFrame is empty")
        return report
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            report['is_valid'] = False
            report['issues'].append(f"Missing required columns: {list(missing_cols)}")
    
    # Check target column
    if target_column:
        if target_column not in df.columns:
            report['is_valid'] = False
            report['issues'].append(f"Target column '{target_column}' not found")
        elif df[target_column].isnull().all():
            report['is_valid'] = False
            report['issues'].append(f"Target column '{target_column}' is entirely null")
        elif df[target_column].isnull().any():
            null_count = df[target_column].isnull().sum()
            report['warnings'].append(f"Target column has {null_count} null values")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report['warnings'].append(f"Found {duplicate_count} duplicate rows")
    
    # Data type consistency
    mixed_types = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric values are stored as strings
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                mixed_types.append(col)
            except (ValueError, TypeError):
                pass
    
    if mixed_types:
        report['warnings'].append(f"Columns with potential type issues: {mixed_types}")
    
    # Summary statistics
    report['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': duplicate_count,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return report

def create_model_comparison_report(results: Dict[str, Dict[str, float]], 
                                 output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Create a comprehensive model comparison report.
    
    Args:
        results (Dict[str, Dict[str, float]]): Model results dictionary
        output_path (Union[str, Path], optional): Path to save report
        
    Returns:
        pd.DataFrame: Comparison report
    """
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Add rankings for each metric
    for metric in df.columns:
        if metric in ['RMSE', 'MAE', 'MAPE']:
            # Lower is better
            df[f'{metric}_Rank'] = df[metric].rank(ascending=True)
        else:
            # Higher is better (R2, etc.)
            df[f'{metric}_Rank'] = df[metric].rank(ascending=False)
    
    # Calculate overall rank (average of individual ranks)
    rank_cols = [col for col in df.columns if col.endswith('_Rank')]
    if rank_cols:
        rank_values = df[rank_cols].values
        df['Overall_Rank'] = np.mean(rank_values, axis=1)
        df = df.sort_values('Overall_Rank')
    
    # Add performance categories
    if 'R2' in df.columns:
        df['Performance_Category'] = df['R2'].apply(
            lambda x: 'Excellent' if x >= 0.9 else
                     'Good' if x >= 0.8 else
                     'Fair' if x >= 0.7 else
                     'Poor'
        )
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Model comparison report saved to {output_path}")
    
    return df

def backup_file(filepath: Union[str, Path], backup_dir: str = "backups") -> Path:
    """
    Create a timestamped backup of a file.
    
    Args:
        filepath (Union[str, Path]): File to backup
        backup_dir (str): Backup directory name
        
    Returns:
        Path: Backup file path
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Create backup directory
    backup_path = filepath.parent / backup_dir
    backup_path.mkdir(exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp_str = timestamp()
    backup_filename = f"{filepath.stem}_{timestamp_str}{filepath.suffix}"
    backup_file_path = backup_path / backup_filename
    
    # Copy file
    import shutil
    shutil.copy2(filepath, backup_file_path)
    
    logger = logging.getLogger("house_price_prediction")
    logger.info(f"✅ Backup created: {backup_file_path}")
    
    return backup_file_path

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility.
    
    Returns:
        Dict[str, Any]: System information
    """
    import platform
    import sys
    
    info = {
        'timestamp': timestamp(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'hostname': platform.node(),
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }
    
    # Try to get additional package versions
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except (ImportError, AttributeError):
        pass
    
    try:
        import matplotlib
        info['matplotlib_version'] = getattr(matplotlib, '__version__', 'unknown')
    except (ImportError, AttributeError):
        pass
    
    try:
        import seaborn
        info['seaborn_version'] = getattr(seaborn, '__version__', 'unknown')
    except (ImportError, AttributeError):
        pass
    
    return info

def safe_divide(numerator: Union[int, float], 
                denominator: Union[int, float], 
                default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable format.
    
    Args:
        bytes_value (int): Number of bytes
        
    Returns:
        str: Formatted string (e.g., "1.5 MB")
    """
    value = float(bytes_value)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"

def check_memory_usage(df: pd.DataFrame, threshold_mb: float = 100.0) -> Dict[str, Any]:
    """
    Check DataFrame memory usage and provide optimization suggestions.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        threshold_mb (float): Memory threshold in MB
        
    Returns:
        Dict[str, Any]: Memory analysis results
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory_bytes = memory_usage.sum()
    total_memory_mb = total_memory_bytes / 1024**2
    
    analysis = {
        'total_memory_mb': total_memory_mb,
        'memory_per_column': {},
        'optimization_suggestions': [],
        'exceeds_threshold': total_memory_mb > threshold_mb
    }
    
    # Analyze memory per column
    for col in df.columns:
        col_memory = memory_usage[col] / 1024**2
        analysis['memory_per_column'][col] = {
            'memory_mb': col_memory,
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum()
        }
        
        # Suggest optimizations
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                analysis['optimization_suggestions'].append(
                    f"Convert '{col}' to category dtype (unique ratio: {unique_ratio:.2f})"
                )
        
        elif df[col].dtype in ['int64', 'float64']:
            if df[col].min() >= 0 and df[col].max() < 2**32:
                analysis['optimization_suggestions'].append(
                    f"Downcast '{col}' to smaller numeric type"
                )
    
    return analysis

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (Union[str, Path]): Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Configuration loaded from {config_path}")
        
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading config from {config_path}: {str(e)}")

def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path],
                create_dirs: bool = True) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (Union[str, Path]): Path to save config
        create_dirs (bool): Create directories if they don't exist
    """
    try:
        config_path = Path(config_path)
        
        if create_dirs:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger = logging.getLogger("house_price_prediction")
        logger.info(f"✅ Configuration saved to {config_path}")
        
    except Exception as e:
        raise Exception(f"Error saving config to {config_path}: {str(e)}")

# ============================================================================
# PROJECT UTILITIES
# ============================================================================

def initialize_project(project_name: str = "house_price_prediction",
                      base_path: Union[str, Path] = ".") -> Dict[str, Any]:
    """
    Initialize a complete ML project structure.
    
    Args:
        project_name (str): Name of the project
        base_path (Union[str, Path]): Base directory
        
    Returns:
        Dict[str, Any]: Project initialization summary
    """
    base_path = Path(base_path)
    
    # Create directory structure
    directories = create_output_dirs(base_path)
    
    # Setup logger
    log_file = directories['logs'] / f"{project_name}.log"
    logger = setup_logger(project_name, log_file=str(log_file))
    
    # Create default config
    config = {
        'project_name': project_name,
        'created_at': timestamp(),
        'directories': {k: str(v) for k, v in directories.items()},
        'data_source': '',
        'target_column': 'SalePrice',
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'models_to_try': ['Linear', 'RandomForest', 'XGBoost'],
        'metrics': ['RMSE', 'MAE', 'R2']
    }
    
    config_path = base_path / 'config.json'
    save_config(config, config_path)
    
    # Get system info
    system_info = get_system_info()
    system_info_path = directories['logs'] / 'system_info.json'
    
    with open(system_info_path, 'w') as f:
        json.dump(system_info, f, indent=2, default=str)
    
    summary = {
        'project_name': project_name,
        'base_path': str(base_path),
        'directories_created': directories,
        'config_path': str(config_path),
        'log_file': str(log_file),
        'system_info_path': str(system_info_path)
    }
    
    logger.info(f"🚀 Project '{project_name}' initialized successfully!")
    logger.info(f"📁 Base path: {base_path}")
    logger.info(f"📋 Config saved to: {config_path}")
    
    return summary
