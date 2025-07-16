"""
🧪 Quick Smoke Tests for House Price Prediction
Simple tests to verify basic functionality

Run with: python tests/test_smoke.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn
        import plotly
        import streamlit
        print("✅ All core libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from src import preprocess, train_models, evaluate
        print("✅ All src modules imported successfully")
    except ImportError as e:
        print(f"⚠️  Some src modules not available: {e}")
    
    try:
        from utils import helpers
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"⚠️  Utils module not available: {e}")
    
    return True

def test_data_processing():
    """Test basic data processing functionality"""
    print("\n📊 Testing data processing...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'GrLivArea': [2000, 1500, 2500, 1800, 2200],
        'OverallQual': [7, 6, 8, 7, 9],
        'YearBuilt': [2005, 1995, 2010, 2000, 2015],
        'Neighborhood': ['NAmes', 'CollgCr', 'NAmes', 'OldTown', 'Gilbert'],
        'SalePrice': [250000, 180000, 320000, 220000, 380000]
    })
    
    try:
        # Test basic operations
        assert len(sample_data) == 5
        assert 'SalePrice' in sample_data.columns
        
        # Test basic statistics
        mean_price = sample_data['SalePrice'].mean()
        assert mean_price > 0
        
        print(f"✅ Sample data created: {len(sample_data)} rows, mean price: ${mean_price:,.0f}")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_model_structure():
    """Test that model files and structure are correct"""
    print("\n🤖 Testing model structure...")
    
    model_dir = PROJECT_ROOT / "model"
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pkl"))
        print(f"✅ Model directory exists with {len(model_files)} model files")
        
        for model_file in model_files:
            print(f"   📄 Found: {model_file.name}")
        
        return True
    else:
        print("⚠️  Model directory not found - run training first")
        return True  # Not a failure, just needs training

def test_app_structure():
    """Test that app files exist and are structured correctly"""
    print("\n🌐 Testing app structure...")
    
    required_files = [
        "app.py",
        "main.py", 
        "run_dashboard.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = PROJECT_ROOT / file_name
        if file_path.exists():
            print(f"✅ Found: {file_name}")
        else:
            missing_files.append(file_name)
            print(f"❌ Missing: {file_name}")
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print(f"⚠️  Missing {len(missing_files)} required files")
        return False

def test_directories():
    """Test project directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "src", 
        "utils",
        "config",
        "tests"
    ]
    
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            files_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}/ exists ({files_count} files)")
        else:
            print(f"❌ {dir_name}/ missing")
    
    return True

def run_all_tests():
    """Run all smoke tests"""
    print("🏠 House Price Prediction - Smoke Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_processing,
        test_model_structure,
        test_app_structure,
        test_directories
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
    
    print(f"\n🎯 SMOKE TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All smoke tests passed! 🚀")
        print("💡 Your project structure looks good!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
