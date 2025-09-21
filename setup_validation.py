"""
Setup validation script for the Bandwidth Prediction System.
Validates UV environment, dependencies, and basic functionality.
"""

import sys
import subprocess
import importlib
import os
from typing import List, Tuple


def check_uv_installation() -> bool:
    """Check if uv is installed and accessible."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ UV is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ UV is installed but not working properly")
            return False
    except FileNotFoundError:
        print("✗ UV is not installed or not in PATH")
        print("  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def check_virtual_environment() -> bool:
    """Check if we're in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✓ Virtual environment active: {sys.prefix}")
        return True
    else:
        print("✗ No virtual environment detected")
        print("  Run: uv sync && source .venv/bin/activate")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'optuna',
        'matplotlib',
        'seaborn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)

    return len(missing_packages) == 0, missing_packages


def check_project_structure() -> bool:
    """Check if project structure is correct."""
    required_paths = [
        'src/',
        'src/features/',
        'src/models/',
        'src/data/',
        'src/app/',
        'models/',
        'sample_data/',
        'pyproject.toml',
        'main.py'
    ]

    missing_paths = []

    for path in required_paths:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} is missing")
            missing_paths.append(path)

    return len(missing_paths) == 0


def check_data_files() -> bool:
    """Check if sample data files are present."""
    data_files = [
        'sample_data/internet_details.csv',
        'sample_data/internet_event_details.csv'
    ]

    missing_files = []

    for file_path in data_files:
        if os.path.exists(file_path):
            # Check if file has content
            if os.path.getsize(file_path) > 0:
                print(f"✓ {file_path} (has content)")
            else:
                print(f"⚠ {file_path} (empty file)")
        else:
            print(f"✗ {file_path} is missing")
            missing_files.append(file_path)

    return len(missing_files) == 0


def test_basic_imports() -> bool:
    """Test if our custom modules can be imported."""
    try:
        sys.path.insert(0, os.getcwd())

        # Test basic imports
        from src.data.loader import BandwidthDataLoader
        from src.features.temporal_features import create_temporal_features
        from src.features.event_features import EventCategorizer

        print("✓ All custom modules import successfully")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_loading() -> bool:
    """Test if data can be loaded properly."""
    try:
        sys.path.insert(0, os.getcwd())
        from src.data.loader import BandwidthDataLoader

        loader = BandwidthDataLoader(
            'sample_data/internet_details.csv',
            'sample_data/internet_event_details.csv'
        )

        # Test loading combinations
        combinations = loader.get_available_combinations()

        if combinations:
            print(f"✓ Data loader works - found {len(combinations)} combinations")

            # Test loading data for first combination
            item, service_type = combinations[0]
            train_df, test_df = loader.prepare_training_data(item, service_type)
            print(f"✓ Data preparation works - {len(train_df)} train, {len(test_df)} test samples")

            return True
        else:
            print("✗ No data combinations found")
            return False

    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False


def run_uv_sync() -> bool:
    """Run uv sync to install dependencies."""
    try:
        print("\nRunning 'uv sync' to install dependencies...")
        result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ UV sync completed successfully")
            return True
        else:
            print(f"✗ UV sync failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error running UV sync: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🔧 BANDWIDTH PREDICTION SYSTEM - SETUP VALIDATION")
    print("=" * 60)

    checks = []

    # Check UV installation
    print("\n1. Checking UV installation...")
    uv_ok = check_uv_installation()
    checks.append(("UV Installation", uv_ok))

    # Check project structure
    print("\n2. Checking project structure...")
    structure_ok = check_project_structure()
    checks.append(("Project Structure", structure_ok))

    # Check data files
    print("\n3. Checking data files...")
    data_ok = check_data_files()
    checks.append(("Data Files", data_ok))

    # If UV is available but dependencies are missing, try to install them
    print("\n4. Checking dependencies...")
    deps_ok, missing = check_dependencies()

    if not deps_ok and uv_ok:
        print(f"\nMissing packages: {missing}")
        print("Attempting to install with UV...")

        if run_uv_sync():
            print("\nRe-checking dependencies after installation...")
            deps_ok, missing = check_dependencies()

    checks.append(("Dependencies", deps_ok))

    # Check virtual environment
    print("\n5. Checking virtual environment...")
    venv_ok = check_virtual_environment()
    checks.append(("Virtual Environment", venv_ok))

    # Test imports
    print("\n6. Testing module imports...")
    imports_ok = test_basic_imports()
    checks.append(("Module Imports", imports_ok))

    # Test data loading
    if deps_ok and imports_ok and data_ok:
        print("\n7. Testing data loading...")
        loading_ok = test_data_loading()
        checks.append(("Data Loading", loading_ok))
    else:
        print("\n7. Skipping data loading test (dependencies not met)")
        checks.append(("Data Loading", False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {check_name}")
        if result:
            passed += 1

    print(f"\n📈 Results: {passed}/{len(checks)} checks passed")

    if passed == len(checks):
        print("\n🎉 Setup validation passed!")
        print("\n📋 Next steps:")
        print("   1. List available combinations:")
        print("      python main.py cli list-combinations")
        print("   2. Train your first model:")
        print("      python main.py train --item 'Google' --service-type 'cache'")
        print("   3. Make predictions:")
        print("      python main.py predict --item 'Google' --service-type 'cache' --days 14")
        print("   4. Try interactive mode:")
        print("      python main.py interactive")

    else:
        print("\n⚠️ Some validation checks failed.")
        print("\n🔧 Recommended fixes:")

        for check_name, result in checks:
            if not result:
                if check_name == "UV Installation":
                    print("   • Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
                elif check_name == "Dependencies":
                    print("   • Install dependencies: uv sync")
                elif check_name == "Virtual Environment":
                    print("   • Activate environment: source .venv/bin/activate")
                elif check_name == "Data Files":
                    print("   • Ensure sample_data/ contains the required CSV files")
                elif check_name == "Project Structure":
                    print("   • Ensure you're in the project root directory")
                    print("   • Verify all source files are present")

    return passed == len(checks)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)