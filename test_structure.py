"""
Test script to validate the project structure and imports.
This script tests the basic structure without requiring external dependencies.
"""

import os
import sys

def test_project_structure():
    """Test that all required files and directories exist."""
    print("🔍 Testing Project Structure...")

    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/features/__init__.py',
        'src/features/temporal_features.py',
        'src/features/event_features.py',
        'src/models/__init__.py',
        'src/models/trainer.py',
        'src/models/predictor.py',
        'src/data/__init__.py',
        'src/data/loader.py',
        'src/app/__init__.py',
        'src/app/cli.py',
        'src/app/interactive.py',
        'notebooks/bandwidth_prediction.ipynb',
        'notebooks/bandwidth_prediction_enhanced.ipynb'
    ]

    required_dirs = [
        'src',
        'src/features',
        'src/models',
        'src/data',
        'src/app',
        'models',
        'notebooks',
        'sample_data'
    ]

    missing_files = []
    missing_dirs = []

    # Check directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
        else:
            print(f"✅ Directory: {directory}")

    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ File: {file_path}")

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False

    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False

    print("\n✅ All required files and directories exist!")
    return True

def test_module_imports():
    """Test that our modules can be imported without external dependencies."""
    print("\n🔍 Testing Module Structure...")

    try:
        # Test basic module imports (without external deps)
        print("📦 Testing src package...")
        import src
        print("✅ src package imported")

        print("📦 Testing features package...")
        import src.features
        print("✅ src.features package imported")

        print("📦 Testing models package...")
        import src.models
        print("✅ src.models package imported")

        print("📦 Testing data package...")
        import src.data
        print("✅ src.data package imported")

        print("📦 Testing app package...")
        import src.app
        print("✅ src.app package imported")

        print("\n✅ All modules can be imported!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def test_data_files():
    """Test that data files exist."""
    print("\n🔍 Testing Data Files...")

    data_files = [
        'sample_data/internet_details.csv',
        'sample_data/internet_event_details.csv'
    ]

    missing_data = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_data.append(file_path)
        else:
            print(f"✅ Data file: {file_path}")

    if missing_data:
        print(f"\n⚠️  Missing data files: {missing_data}")
        print("   Note: Data files are needed for full functionality")
        return False

    print("\n✅ All data files exist!")
    return True

def main():
    """Run all tests."""
    print("🧪 BANDWIDTH PREDICTION SYSTEM - STRUCTURE TEST")
    print("=" * 60)

    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    tests = [
        ("Project Structure", test_project_structure),
        ("Module Imports", test_module_imports),
        ("Data Files", test_data_files)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if result:
            passed += 1

    print(f"\n📈 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! The project structure is correct.")
        print("\n🚀 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Test with real data: python main.py cli list-combinations")
        print("   3. Train a model: python main.py train --item 'Google' --service-type 'cache'")
    else:
        print("\n⚠️  Some tests failed. Please check the project structure.")

    return passed == len(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)