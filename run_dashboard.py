#!/usr/bin/env python3
"""
🚀 Dashboard Launcher
Quick script to launch the House Price Prediction Dashboard

Usage: python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching House Price Prediction Dashboard...")
    print("="*50)
    
    # Check if we're in the right directory
    app_path = Path("app.py")
    if not app_path.exists():
        print("❌ Error: app.py not found in current directory")
        print("💡 Make sure you're running this from the project root directory")
        return
    
    # Check if model exists
    model_path = Path("model/best_model.pkl") 
    lasso_path = Path("model/lasso_model.pkl")
    
    if not model_path.exists() and not lasso_path.exists():
        print("⚠️  Warning: No trained model found!")
        print("💡 Run 'python main.py' first to train the model")
        print("🔄 Continuing anyway - you'll see instructions in the dashboard")
        print()
    
    try:
        # Launch Streamlit
        print("🌟 Starting Streamlit dashboard...")
        print("🔗 Dashboard will open in your browser automatically")
        print("📍 URL: http://localhost:8501")
        print()
        print("⏹️  Press Ctrl+C to stop the dashboard")
        print("="*50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for using House Price Predictor!")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found!")
        print("💡 Install it with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
