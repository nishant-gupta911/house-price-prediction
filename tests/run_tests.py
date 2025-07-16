#!/usr/bin/env python3
"""
🧪 Test Runner for House Price Prediction Project
Easy script to run all tests with different configurations

Usage: python run_tests.py [options]
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print("Error output:", e.stderr)
        print("Standard output:", e.stdout)
        return False

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run House Price Prediction tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🏠 House Price Prediction - Test Suite Runner")
    print("=" * 60)
    
    # Check if we're in the right directory (now running from tests/)
    if not Path(".").exists() or not Path("test_house_price_prediction.py").exists():
        print("❌ Error: test files not found!")
        print("💡 Make sure you're running this from the tests directory")
        return 1
    
    # Check if pytest is installed
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error: pytest not installed!")
        print("💡 Install it with: pip install pytest")
        return 1
    
    success_count = 0
    total_tests = 0
    
    if args.quick:
        # Quick tests only (unit tests)
        total_tests += 1
        if run_command(f"{sys.executable} -m pytest . -m 'not slow' --tb=short", 
                      "Quick Unit Tests"):
            success_count += 1
    
    elif args.integration:
        # Integration tests only
        total_tests += 1
        if run_command(f"{sys.executable} -m pytest . -m integration --tb=short", 
                      "Integration Tests"):
            success_count += 1
    
    elif args.coverage:
        # Tests with coverage
        total_tests += 1
        if run_command(f"{sys.executable} -m pytest . --cov=../src --cov=../utils --cov-report=html --cov-report=term", 
                      "Tests with Coverage Report"):
            success_count += 1
            print("\n📊 Coverage report generated in htmlcov/index.html")
    
    else:
        # Run all tests
        tests_to_run = [
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestDataPreprocessing -v", 
             "Data Preprocessing Tests"),
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestModelTraining -v", 
             "Model Training Tests"),
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestModelEvaluation -v", 
             "Model Evaluation Tests"),
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestUtilityFunctions -v", 
             "Utility Function Tests"),
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestStreamlitApp -v", 
             "Streamlit App Tests"),
            (f"{sys.executable} -m pytest test_house_price_prediction.py::TestDataFiles -v", 
             "Data File Tests"),
        ]
        
        for command, description in tests_to_run:
            total_tests += 1
            if run_command(command, description):
                success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"\n🎉 All tests passed! Your code is ready for production! 🚀")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
