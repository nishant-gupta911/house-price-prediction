"""
Configuration file for House Price Prediction Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "demo"

# Data files
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Model files
MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Visualization settings
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = "viridis"
STYLE = "whitegrid"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Features to drop (high missing values or non-predictive)
FEATURES_TO_DROP = [
    'Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature',
    'FireplaceQu', 'LotFrontage'  # Added more features with high missing values
]

# Categorical features that need special handling
HIGH_CARDINALITY_FEATURES = ['Neighborhood', 'Exterior1st', 'Exterior2nd']

# Numerical features for transformation
SKEWED_FEATURES = ['LotArea', 'GrLivArea', 'SalePrice']

# Model hyperparameters
MODEL_PARAMS = {
    'LinearRegression': {},
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000, 5000]
    },
    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000, 5000, 10000]
    },
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9],
        'max_iter': [1000, 5000]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Streamlit app settings
APP_TITLE = "🏠 House Price Prediction Dashboard"
APP_DESCRIPTION = """
Welcome to the **House Price Prediction Dashboard**! 
This interactive tool uses advanced machine learning algorithms to predict house prices 
based on various features like location, size, quality, and more.
"""

# Feature descriptions for the app
FEATURE_DESCRIPTIONS = {
    'GrLivArea': 'Above ground living area (sq ft)',
    'OverallQual': 'Overall material and finish quality (1-10)',
    'OverallCond': 'Overall condition rating (1-10)',
    'YearBuilt': 'Original construction year',
    'TotalBsmtSF': 'Total basement area (sq ft)',
    'GarageCars': 'Size of garage in car capacity',
    'GarageArea': 'Size of garage (sq ft)',
    'FullBath': 'Full bathrooms above grade',
    'BedroomAbvGr': 'Number of bedrooms above basement level',
    'TotRmsAbvGrd': 'Total rooms above grade'
}
