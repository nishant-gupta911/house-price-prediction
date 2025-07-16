#!/usr/bin/env python3
"""
🧹 Project Cleanup Verification Script
Verifies that the project structure is clean and properly organized

Run with: python verify_cleanup.py
"""

from pathlib import Path
import sys

def check_project_structure():
    """Verify the project has proper structure"""
    print("🏠 House Price Prediction - Project Structure Verification")
    print("=" * 60)
    
    # Required directories
    required_dirs = {
        'data': 'Dataset files',
        'src': 'Source code modules', 
        'model': 'Trained models',
        'utils': 'Utility functions',
        'config': 'Configuration files',
        'tests': 'Test suite',
        'notebooks': 'Jupyter notebooks',
        'demo': 'Demo materials',
        'logs': 'Training logs'
    }
    
    # Required files
    required_files = {
        'README.md': 'Project documentation',
        'requirements.txt': 'Python dependencies',
        'app.py': 'Streamlit dashboard',
        'main.py': 'Training pipeline',
        'run_dashboard.py': 'Dashboard launcher',
        '.gitignore': 'Git ignore rules',
        'pytest.ini': 'Test configuration'
    }
    
    print("\n📁 DIRECTORY STRUCTURE CHECK")
    print("-" * 40)
    
    all_good = True
    for dir_name, description in required_dirs.items():
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            file_count = len(list(dir_path.rglob("*")))
            print(f"✅ {dir_name:<12} {description} ({file_count} items)")
        else:
            print(f"❌ {dir_name:<12} Missing!")
            all_good = False
    
    print("\n📄 REQUIRED FILES CHECK")
    print("-" * 40)
    
    for file_name, description in required_files.items():
        file_path = Path(file_name)
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name:<20} {description} ({size:,} bytes)")
        else:
            print(f"❌ {file_name:<20} Missing!")
            all_good = False
    
    print("\n🗑️ UNWANTED FILES CHECK")
    print("-" * 40)
    
    # Check for files that shouldn't exist (excluding venv)
    unwanted_patterns = [
        '__pycache__',
        '.ipynb_checkpoints', 
        '*.pyc',
        '*.pyo',
        '*.tmp',
        '*.bak',
        'anaconda_projects',
        'test_project'
    ]
    
    found_unwanted = []
    for pattern in unwanted_patterns:
        matches = list(Path('.').rglob(pattern))
        # Filter out files/dirs in virtual environment
        filtered_matches = [
            match for match in matches 
            if not any(venv_dir in str(match) for venv_dir in ['venv', 'env', '.venv', '.env'])
        ]
        if filtered_matches:
            found_unwanted.extend(filtered_matches)
    
    if found_unwanted:
        print("⚠️  Found unwanted files/directories:")
        for item in found_unwanted:
            print(f"   - {item}")
        all_good = False
    else:
        print("✅ No unwanted files found - project is clean!")
    
    print("\n🔍 SOURCE CODE STRUCTURE")
    print("-" * 40)
    
    src_files = list(Path('src').glob('*.py')) if Path('src').exists() else []
    if src_files:
        for src_file in src_files:
            size = src_file.stat().st_size
            print(f"✅ src/{src_file.name:<15} ({size:,} bytes)")
    else:
        print("❌ No Python files in src/ directory")
        all_good = False
    
    print("\n🧪 TEST STRUCTURE")
    print("-" * 40)
    
    test_files = list(Path('tests').glob('test_*.py')) if Path('tests').exists() else []
    if test_files:
        for test_file in test_files:
            size = test_file.stat().st_size
            print(f"✅ tests/{test_file.name:<20} ({size:,} bytes)")
    else:
        print("❌ No test files found")
        all_good = False
    
    print("\n" + "=" * 60)
    print("🎯 CLEANUP VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_good:
        print("🎉 ✅ PROJECT STRUCTURE IS PERFECT!")
        print("🚀 Ready for GitHub, deployment, and recruiter presentation!")
        print("\n💡 Next steps:")
        print("   1. git add . && git commit -m 'Clean project structure'")
        print("   2. Run tests: python run_tests.py")
        print("   3. Launch dashboard: python run_dashboard.py")
        return True
    else:
        print("⚠️  ❌ PROJECT NEEDS ATTENTION")
        print("💡 Please fix the issues above and run this script again.")
        return False

def check_git_status():
    """Check git repository status"""
    print("\n📋 GIT REPOSITORY STATUS")
    print("-" * 40)
    
    git_dir = Path('.git')
    if git_dir.exists():
        print("✅ Git repository initialized")
        
        # Check if .gitignore is working
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            print("✅ .gitignore file present")
        else:
            print("⚠️  .gitignore file missing")
    else:
        print("⚠️  Not a git repository")
        print("💡 Initialize with: git init")

def main():
    """Main verification function"""
    try:
        structure_ok = check_project_structure()
        check_git_status()
        
        if structure_ok:
            print("\n🌟 CONGRATULATIONS! Your project is professionally organized! 🌟")
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
