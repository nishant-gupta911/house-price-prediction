"""
Production-Grade Data Preprocessing Module for House Price Prediction

This module provides a comprehensive set of functions for preprocessing house price data,
including missing value handling, feature encoding, scaling, outlier detection, and
skewness transformation.

Author: AI Engineer
Date: July 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset with comprehensive error handling and validation.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has no columns
    """
    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
            
        if df.shape[1] == 0:
            raise ValueError("Dataset has no columns")
            
        logger.info(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        logger.info(f"📊 Columns: {len(df.columns)}, Rows: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies for different data types.
    
    Strategy:
    - Numerical: Fill with median
    - Categorical: Fill with mode (most frequent)
    - Drop columns with >50% missing values
    - Drop specific irrelevant columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    try:
        logger.info("🔧 Handling missing values...")
        df_clean = df.copy()
        
        # Track original missing values
        initial_missing = df_clean.isnull().sum().sum()
        
        # Drop columns with >50% missing values
        missing_threshold = 0.5
        high_missing_cols = df_clean.columns[df_clean.isnull().mean() > missing_threshold].tolist()
        
        # Drop irrelevant columns
        irrelevant_cols = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
        cols_to_drop = list(set(high_missing_cols + irrelevant_cols))
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
        
        if existing_cols_to_drop:
            df_clean = df_clean.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
        
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.debug(f"Filled {col} with median: {median_val}")
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                logger.debug(f"Filled {col} with mode: {mode_val}")
        
        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"✅ Missing values handled: {initial_missing} → {final_missing}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Error handling missing values: {e}")
        raise


def encode_categorical(df: pd.DataFrame, max_categories: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables using appropriate encoding strategies.
    
    Strategy:
    - OneHot encoding for features with ≤ max_categories unique values
    - Label encoding for features with > max_categories unique values
    
    Args:
        df (pd.DataFrame): Input dataframe
        max_categories (int): Threshold for choosing encoding strategy
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Encoded dataframe and encoding info
    """
    try:
        logger.info("🔤 Encoding categorical features...")
        df_encoded = df.copy()
        encoding_info = {
            'onehot_columns': [],
            'label_encoded_columns': {},
            'encoders': {}
        }
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if unique_values <= max_categories:
                # OneHot encoding for low cardinality
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_cols = encoder.fit_transform(df_encoded[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df_encoded.index)
                
                # Add to main dataframe and drop original
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
                
                encoding_info['onehot_columns'].extend(feature_names)
                encoding_info['encoders'][col] = encoder
                logger.debug(f"OneHot encoded {col}: {unique_values} categories")
                
            else:
                # Label encoding for high cardinality
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                
                encoding_info['label_encoded_columns'][col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                encoding_info['encoders'][col] = encoder
                logger.debug(f"Label encoded {col}: {unique_values} categories")
        
        logger.info(f"✅ Categorical encoding completed: {len(categorical_cols)} features processed")
        return df_encoded, encoding_info
        
    except Exception as e:
        logger.error(f"❌ Error encoding categorical features: {e}")
        raise


def scale_features(df: pd.DataFrame, scaler=None, fit: bool = True) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
        scaler: Pre-fitted scaler (optional)
        fit (bool): Whether to fit the scaler
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled dataframe and fitted scaler
    """
    try:
        logger.info("⚖️ Scaling numerical features...")
        df_scaled = df.copy()
        
        # Get numerical columns
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            logger.warning("No numerical columns found for scaling")
            return df_scaled, None
        
        # Initialize scaler if not provided
        if scaler is None:
            scaler = StandardScaler()
        
        if fit:
            # Fit and transform
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            logger.info(f"✅ Fitted and scaled {len(numerical_cols)} numerical features")
        else:
            # Only transform
            df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
            logger.info(f"✅ Scaled {len(numerical_cols)} numerical features using existing scaler")
        
        return df_scaled, scaler
        
    except Exception as e:
        logger.error(f"❌ Error scaling features: {e}")
        raise


def handle_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Method for outlier detection ('iqr' or 'zscore')
        factor (float): IQR factor for outlier threshold
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    try:
        logger.info(f"🧹 Handling outliers using {method} method...")
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Get numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Remove outliers
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                outliers_removed = len(df_clean) - mask.sum()
                
                if outliers_removed > 0:
                    df_clean = df_clean[mask].copy()
                    logger.debug(f"Removed {outliers_removed} outliers from {col}")
        
        elif method == 'zscore':
            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(df_clean[col]))
                mask = z_scores < 3  # 3 standard deviations
                outliers_removed = len(df_clean) - mask.sum()
                
                if outliers_removed > 0:
                    df_clean = df_clean[mask].copy()
                    logger.debug(f"Removed {outliers_removed} outliers from {col}")
        
        final_rows = len(df_clean)
        removed_count = initial_rows - final_rows
        
        logger.info(f"✅ Outlier handling completed: {removed_count} rows removed ({removed_count/initial_rows*100:.1f}%)")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Error handling outliers: {e}")
        raise


def transform_skewed_features(df: pd.DataFrame, skewness_threshold: float = 0.75) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transform skewed numerical features using log1p transformation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        skewness_threshold (float): Threshold for skewness detection
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Transformed dataframe and list of transformed features
    """
    try:
        logger.info(f"📊 Transforming skewed features (threshold: {skewness_threshold})...")
        df_transformed = df.copy()
        transformed_features = []
        
        # Get numerical columns
        numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            # Calculate skewness
            skewness = abs(stats.skew(df_transformed[col].dropna()))
            
            if skewness > skewness_threshold:
                # Check if all values are non-negative
                if df_transformed[col].min() >= 0:
                    df_transformed[col] = np.log1p(df_transformed[col])
                    transformed_features.append(col)
                    logger.debug(f"Applied log1p transformation to {col} (skewness: {skewness:.3f})")
                else:
                    logger.warning(f"Skipped {col}: contains negative values (skewness: {skewness:.3f})")
        
        logger.info(f"✅ Skewness transformation completed: {len(transformed_features)} features transformed")
        
        return df_transformed, transformed_features
        
    except Exception as e:
        logger.error(f"❌ Error transforming skewed features: {e}")
        raise


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable
        
    Raises:
        KeyError: If target column doesn't exist
    """
    try:
        logger.info(f"🎯 Splitting features and target: {target_col}")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"✅ Split completed: Features shape {X.shape}, Target shape {y.shape}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"❌ Error splitting features and target: {e}")
        raise


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    handle_missing: bool = True,
    encode_cats: bool = True,
    scale_nums: bool = True,
    remove_outliers: bool = True,
    transform_skewed: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline with all steps.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str, optional): Target column name
        handle_missing (bool): Whether to handle missing values
        encode_cats (bool): Whether to encode categorical features
        scale_nums (bool): Whether to scale numerical features
        remove_outliers (bool): Whether to remove outliers
        transform_skewed (bool): Whether to transform skewed features
        test_size (float): Test set proportion for train-test split
        random_state (int): Random state for reproducibility
        
    Returns:
        Dict[str, Any]: Dictionary containing processed data and metadata
    """
    try:
        logger.info("🚀 Starting complete preprocessing pipeline...")
        
        result = {
            'processed_data': df.copy(),
            'metadata': {
                'original_shape': df.shape,
                'steps_applied': [],
                'encoding_info': None,
                'scaler': None,
                'transformed_features': [],
                'outliers_removed': 0
            }
        }
        
        # Step 1: Handle missing values
        if handle_missing:
            result['processed_data'] = handle_missing_values(result['processed_data'])
            result['metadata']['steps_applied'].append('missing_values_handled')
        
        # Step 2: Remove outliers (before other transformations)
        if remove_outliers:
            initial_rows = len(result['processed_data'])
            result['processed_data'] = handle_outliers(result['processed_data'])
            result['metadata']['outliers_removed'] = initial_rows - len(result['processed_data'])
            result['metadata']['steps_applied'].append('outliers_removed')
        
        # Step 3: Transform skewed features
        if transform_skewed:
            result['processed_data'], transformed_features = transform_skewed_features(result['processed_data'])
            result['metadata']['transformed_features'] = transformed_features
            result['metadata']['steps_applied'].append('skewness_transformed')
        
        # Step 4: Encode categorical features
        if encode_cats:
            result['processed_data'], encoding_info = encode_categorical(result['processed_data'])
            result['metadata']['encoding_info'] = encoding_info
            result['metadata']['steps_applied'].append('categorical_encoded')
        
        # Step 5: Scale numerical features
        if scale_nums:
            result['processed_data'], scaler = scale_features(result['processed_data'])
            result['metadata']['scaler'] = scaler
            result['metadata']['steps_applied'].append('features_scaled')
        
        # Step 6: Split features and target (if target column provided)
        if target_col:
            X, y = split_features_target(result['processed_data'], target_col)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            result['X_train'] = X_train
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_test'] = y_test
            result['metadata']['steps_applied'].append('train_test_split')
            
            logger.info(f"✅ Train-test split: Train {X_train.shape}, Test {X_test.shape}")
        
        result['metadata']['final_shape'] = result['processed_data'].shape
        
        logger.info("🎉 Preprocessing pipeline completed successfully!")
        logger.info(f"📊 Original shape: {result['metadata']['original_shape']}")
        logger.info(f"📊 Final shape: {result['metadata']['final_shape']}")
        logger.info(f"🔧 Steps applied: {', '.join(result['metadata']['steps_applied'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in preprocessing pipeline: {e}")
        raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_preprocessing_report(metadata: Dict[str, Any]) -> str:
    """
    Generate a comprehensive preprocessing report.
    
    Args:
        metadata (Dict[str, Any]): Metadata from preprocessing pipeline
        
    Returns:
        str: Formatted preprocessing report
    """
    report = []
    report.append("🔍 PREPROCESSING REPORT")
    report.append("=" * 50)
    report.append(f"📊 Original Shape: {metadata['original_shape']}")
    report.append(f"📊 Final Shape: {metadata['final_shape']}")
    report.append(f"🗑️ Outliers Removed: {metadata['outliers_removed']}")
    report.append(f"🔧 Steps Applied: {', '.join(metadata['steps_applied'])}")
    
    if metadata['transformed_features']:
        report.append(f"📈 Transformed Features: {len(metadata['transformed_features'])}")
        report.append(f"   Features: {', '.join(metadata['transformed_features'][:5])}...")
    
    if metadata['encoding_info']:
        onehot_count = len(metadata['encoding_info']['onehot_columns'])
        label_count = len(metadata['encoding_info']['label_encoded_columns'])
        report.append(f"🔤 Categorical Encoding:")
        report.append(f"   OneHot: {onehot_count} features")
        report.append(f"   Label: {label_count} features")
    
    return "\n".join(report)


def save_preprocessing_artifacts(
    metadata: Dict[str, Any], 
    output_dir: str = "./preprocessing_artifacts"
) -> None:
    """
    Save preprocessing artifacts for reproducibility.
    
    Args:
        metadata (Dict[str, Any]): Metadata from preprocessing pipeline
        output_dir (str): Directory to save artifacts
    """
    import pickle
    import os
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        if metadata['scaler']:
            with open(f"{output_dir}/scaler.pkl", 'wb') as f:
                pickle.dump(metadata['scaler'], f)
            logger.info(f"✅ Scaler saved to {output_dir}/scaler.pkl")
        
        # Save encoding info
        if metadata['encoding_info']:
            with open(f"{output_dir}/encoding_info.pkl", 'wb') as f:
                pickle.dump(metadata['encoding_info'], f)
            logger.info(f"✅ Encoding info saved to {output_dir}/encoding_info.pkl")
        
        # Save metadata
        with open(f"{output_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"✅ Metadata saved to {output_dir}/metadata.pkl")
        
    except Exception as e:
        logger.error(f"❌ Error saving preprocessing artifacts: {e}")
        raise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """
    Example: Basic preprocessing usage
    """
    # Load data
    df = load_data("../data/train.csv")
    
    # Run complete preprocessing pipeline
    result = preprocess_pipeline(
        df=df,
        target_col='SalePrice',
        handle_missing=True,
        encode_cats=True,
        scale_nums=True,
        remove_outliers=True,
        transform_skewed=True
    )
    
    # Access processed data
    X_train = result['X_train']
    X_test = result['X_test']
    y_train = result['y_train']
    y_test = result['y_test']
    
    # Generate report
    report = get_preprocessing_report(result['metadata'])
    print(report)
    
    # Save artifacts
    save_preprocessing_artifacts(result['metadata'])
    
    return result


def example_custom_preprocessing():
    """
    Example: Custom step-by-step preprocessing
    """
    # Load data
    df = load_data("../data/train.csv")
    
    # Step-by-step preprocessing
    df_clean = handle_missing_values(df)
    df_no_outliers = handle_outliers(df_clean)
    df_transformed, skewed_features = transform_skewed_features(df_no_outliers)
    df_encoded, encoding_info = encode_categorical(df_transformed)
    df_scaled, scaler = scale_features(df_encoded)
    
    # Split features and target
    X, y = split_features_target(df_scaled, 'SalePrice')
    
    return X, y, scaler, encoding_info


def example_prediction_preprocessing():
    """
    Example: Preprocessing for prediction (no target column)
    """
    # Load test data
    df_test = load_data("../data/test.csv")
    
    # Preprocess without target column
    result = preprocess_pipeline(
        df=df_test,
        target_col=None,  # No target for test data
        handle_missing=True,
        encode_cats=True,
        scale_nums=True,
        remove_outliers=False,  # Don't remove outliers from test data
        transform_skewed=True
    )
    
    return result['processed_data']


if __name__ == "__main__":
    """
    Main execution block for testing the preprocessing module
    """
    try:
        logger.info("🚀 Testing preprocessing module...")
        
        # Example usage
        result = example_basic_usage()
        
        logger.info("✅ Preprocessing module test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error testing preprocessing module: {e}")
        raise
