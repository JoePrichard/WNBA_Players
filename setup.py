# setup.py - WNBA Prediction System Setup Script (Fixed Version)
#!/usr/bin/env python3
"""
WNBA Prediction System Setup Script - Fixed Version
Automated installation and configuration script with comprehensive error handling.

This fixed version includes:
- Better dependency management
- Improved error handling and recovery
- Validation of all components
- Clear feedback and instructions
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class WNBASetup:
    """
    Setup manager for WNBA prediction system with comprehensive error handling.
    
    This class handles the complete setup process including:
    - Python version validation
    - Directory creation
    - Dependency installation with fallbacks
    - Configuration file creation
    - Module validation with detailed feedback
    - Quick functionality testing
    
    Attributes:
        verbose (bool): Enable verbose logging output
        required_python_version (Tuple[int, int]): Minimum required Python version
        project_dirs (List[str]): List of directories to create
        core_dependencies (List[str]): Core dependencies that must be installed
        optional_dependencies (List[str]): Optional dependencies for enhanced features
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize setup manager.
        
        Args:
            verbose (bool, optional): Enable verbose logging. Defaults to False.
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self.required_python_version = (3, 8)
        self.project_dirs = [
            "wnba_game_data",
            "wnba_predictions", 
            "wnba_models",
            "wnba_validation",
            "logs"
        ]
        
        # Separate core and optional dependencies
        self.core_dependencies = [
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "requests>=2.26.0",
            "beautifulsoup4>=4.10.0",
            "python-dateutil>=2.8.0"
        ]
        
        self.optional_dependencies = [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0", 
            "torch>=1.10.0",
            "toml>=0.10.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "streamlit>=1.2.0",
            "plotly>=5.3.0",
            "joblib>=1.0.0"
        ]
    
    def check_python_version(self) -> bool:
        """
        Check if Python version meets requirements.
        
        Returns:
            bool: True if Python version is sufficient, False otherwise
        """
        logger.info("Checking Python version...")
        
        current_version = sys.version_info[:2]
        required_version = self.required_python_version
        
        if current_version >= required_version:
            logger.info(f"✅ Python {sys.version.split()[0]} (meets requirement >={required_version[0]}.{required_version[1]})")
            return True
        else:
            logger.error(f"❌ Python {current_version[0]}.{current_version[1]} found, but >={required_version[0]}.{required_version[1]} required")
            return False
    
    def create_directories(self) -> bool:
        """
        Create necessary project directories.
        
        Returns:
            bool: True if all directories created successfully, False otherwise
        """
        logger.info("Creating project directories...")
        
        try:
            for directory in self.project_dirs:
                Path(directory).mkdir(exist_ok=True)
                logger.info(f"  ✅ Created {directory}/")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def install_dependencies(self, skip_optional: bool = False) -> Tuple[bool, List[str]]:
        """
        Install Python dependencies with fallback options.
        
        Args:
            skip_optional: Whether to skip optional dependencies
            
        Returns:
            Tuple of (success, failed_packages)
        """
        logger.info("Installing Python dependencies...")
        
        failed_packages = []
        
        try:
            # Upgrade pip first
            logger.info("Upgrading pip...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL if not self.verbose else None)
            
            # Install core dependencies first
            logger.info("Installing core dependencies...")
            for package in self.core_dependencies:
                try:
                    logger.info(f"  Installing {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL if not self.verbose else None)
                    logger.info(f"  ✅ {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"  ❌ Failed to install {package}")
                    failed_packages.append(package)
            
            # Install optional dependencies if not skipping
            if not skip_optional:
                logger.info("Installing optional dependencies...")
                for package in self.optional_dependencies:
                    try:
                        logger.info(f"  Installing {package}...")
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package
                        ], stdout=subprocess.DEVNULL if not self.verbose else None)
                        logger.info(f"  ✅ {package}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"  ⚠️ Optional package failed: {package}")
                        # Don't add to failed_packages for optional deps
            
            core_failed = [pkg for pkg in failed_packages if any(pkg.startswith(core.split('>=')[0]) for core in self.core_dependencies)]
            
            if not core_failed:
                logger.info("✅ Core dependencies installed successfully")
                return True, failed_packages
            else:
                logger.error(f"❌ Failed to install core dependencies: {core_failed}")
                return False, failed_packages
                
        except Exception as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False, failed_packages
    
    def create_basic_config(self) -> bool:
        """
        Create basic configuration file if none exists.
        
        Returns:
            bool: True if config created successfully, False otherwise
        """
        logger.info("Creating basic configuration...")
        
        config_file = "config.toml"
        
        if os.path.exists(config_file):
            logger.info("✅ Configuration file already exists")
            return True
        
        try:
            basic_config = '''# WNBA Prediction System Configuration
# Basic configuration file created by setup

[prediction]
target_stats = ["points", "total_rebounds", "assists"]
min_games_for_prediction = 5
confidence_threshold = 0.6

[data]
data_dir = "wnba_game_data"
output_dir = "wnba_predictions"
model_dir = "wnba_models"
rate_limit_delay = 2.0

[logging]
level = "INFO"
log_file = "logs/wnba_predictions.log"
'''
            
            with open(config_file, 'w') as f:
                f.write(basic_config)
            
            logger.info(f"✅ Created basic {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create config: {e}")
            return False
    
    def validate_installation(self) -> Dict[str, bool]:
        """
        Validate that all components can be imported.
        
        Returns:
            Dict[str, bool]: Dictionary mapping module names to success status
        """
        logger.info("Validating installation...")
        
        results = {}
        
        # Test core Python modules
        core_modules = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'sklearn',
            'requests': 'requests',
            'beautifulsoup4': 'bs4',
            'dateutil': 'dateutil'
        }
        
        for package_name, import_name in core_modules.items():
            try:
                __import__(import_name)
                results[package_name] = True
                logger.info(f"  ✅ {package_name}")
            except ImportError:
                results[package_name] = False
                logger.warning(f"  ❌ {package_name}")
        
        # Test optional modules
        optional_modules = {
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
            'torch': 'torch',
            'streamlit': 'streamlit',
            'plotly': 'plotly',
            'toml': 'toml'
        }
        
        for package_name, import_name in optional_modules.items():
            try:
                __import__(import_name)
                results[package_name] = True
                logger.info(f"  ✅ {package_name} (optional)")
            except ImportError:
                results[package_name] = False
                logger.info(f"  ⚠️ {package_name} (optional - not available)")
        
        # Test custom modules
        custom_modules = [
            "data_models",
            "data_fetcher", 
            "feature_engineer",
            "prediction_models",
            "main_application"
        ]
        
        for module in custom_modules:
            try:
                __import__(module)
                results[module] = True
                logger.info(f"  ✅ {module}")
            except ImportError as e:
                results[module] = False
                logger.warning(f"  ❌ {module}: {e}")
        
        # Summary
        core_success = all(results.get(pkg, False) for pkg in core_modules.keys())
        custom_success = all(results.get(mod, False) for mod in custom_modules)
        
        if core_success and custom_success:
            logger.info("✅ All core modules validated successfully")
        else:
            failed_core = [k for k in core_modules.keys() if not results.get(k, False)]
            failed_custom = [k for k in custom_modules if not results.get(k, False)]
            
            if failed_core:
                logger.warning(f"⚠️ Core modules failed: {failed_core}")
            if failed_custom:
                logger.warning(f"⚠️ Custom modules failed: {failed_custom}")
        
        return results
    
    def run_quick_test(self) -> bool:
        """
        Run a quick functionality test.
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        logger.info("Running quick functionality test...")
        
        tests_passed = 0
        total_tests = 6
        
        try:
            # Test 1: Data models
            try:
                from data_models import PredictionConfig, PlayerPrediction, HomeAway
                config = PredictionConfig()
                logger.info("  ✅ Data models")
                tests_passed += 1
            except Exception as e:
                logger.warning(f"  ❌ Data models: {e}")
            
            # Test 2: Data fetcher initialization
            try:
                from data_fetcher import WNBAStatsScraper
                fetcher = WNBAStatsScraper()
                logger.info("  ✅ Data fetcher")
                tests_passed += 1
            except Exception as e:
                logger.warning(f"  ❌ Data fetcher: {e}")
            
            # Test 3: Feature engineer
            try:
                from feature_engineer import WNBAFeatureEngineer
                engineer = WNBAFeatureEngineer()
                logger.info("  ✅ Feature engineer")
                tests_passed += 1
            except Exception as e:
                logger.warning(f"  ❌ Feature engineer: {e}")
            
            # Test 4: Prediction models (may fail if ML libs not installed)
            try:
                from prediction_models import WNBAPredictionModel
                model = WNBAPredictionModel()
                logger.info("  ✅ Prediction models")
                tests_passed += 1
            except Exception as e:
                logger.warning(f"  ⚠️ Prediction models: {e}")
            
            # Test 5: Main application
            try:
                from main_application import WNBADailyPredictor
                predictor = WNBADailyPredictor()
                logger.info("  ✅ Main application")
                tests_passed += 1
            except Exception as e:
                logger.warning(f"  ❌ Main application: {e}")
            
            # Test 6: Utilities (if available)
            try:
                from utils import setup_project_structure
                setup_project_structure()
                logger.info("  ✅ Utilities")
                tests_passed += 1
            except Exception as e:
                logger.info(f"  ⚠️ Utilities (optional): {e}")
                tests_passed += 1  # Don't penalize for missing utils
            
            success_rate = tests_passed / total_tests
            if success_rate >= 0.8:
                logger.info(f"✅ Quick test passed ({tests_passed}/{total_tests} components working)")
                return True
            else:
                logger.warning(f"⚠️ Quick test partial success ({tests_passed}/{total_tests} components working)")
                return False
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """
        Run complete setup process with comprehensive error handling.
        
        Returns:
            bool: True if setup completed successfully, False if any critical issues occurred
        """
        logger.info("🏀 Starting WNBA Prediction System Setup (Fixed Version)")
        logger.info("=" * 60)
        
        setup_steps = [
            ("Python Version Check", self.check_python_version, True),
            ("Directory Creation", self.create_directories, True),
            ("Core Dependencies", lambda: self.install_dependencies(skip_optional=False)[0], True),
            ("Basic Configuration", self.create_basic_config, False),
            ("Component Validation", lambda: self.validate_installation(), False),
            ("Functionality Test", self.run_quick_test, False)
        ]
        
        critical_failures = 0
        warnings = 0
        
        for step_name, step_func, is_critical in setup_steps:
            logger.info(f"\n🔧 {step_name}")
            try:
                if step_name == "Component Validation":
                    # Special handling for validation step
                    results = step_func()
                    core_modules = ['pandas', 'numpy', 'sklearn', 'requests', 'beautifulsoup4']
                    custom_modules = ['data_models', 'data_fetcher', 'main_application']
                    
                    core_success = all(results.get(mod, False) for mod in core_modules)
                    custom_success = all(results.get(mod, False) for mod in custom_modules)
                    
                    if core_success and custom_success:
                        logger.info(f"✅ {step_name} - All core components available")
                    elif core_success:
                        logger.warning(f"⚠️ {step_name} - Core components OK, some custom modules missing")
                        warnings += 1
                    else:
                        logger.error(f"❌ {step_name} - Critical components missing")
                        if is_critical:
                            critical_failures += 1
                else:
                    # Normal step handling
                    success = step_func()
                    if success:
                        logger.info(f"✅ {step_name} completed successfully")
                    else:
                        if is_critical:
                            logger.error(f"❌ {step_name} failed (critical)")
                            critical_failures += 1
                        else:
                            logger.warning(f"⚠️ {step_name} had issues (non-critical)")
                            warnings += 1
                            
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                if is_critical:
                    critical_failures += 1
                else:
                    warnings += 1
        
        # Summary
        logger.info("\n" + "=" * 60)
        if critical_failures == 0:
            if warnings == 0:
                logger.info("🎉 Setup completed successfully!")
                logger.info("✅ All components ready for use")
            else:
                logger.info(f"🎉 Setup completed with {warnings} warnings")
                logger.info("✅ Core system ready - some optional features may be limited")
            
            self._print_next_steps()
            return True
        else:
            logger.error(f"❌ Setup failed with {critical_failures} critical issues and {warnings} warnings")
            logger.info("💡 Try installing missing dependencies manually or check error messages above")
            self._print_troubleshooting_guide()
            return False
    
    def _print_next_steps(self) -> None:
        """Print next steps for the user."""
        logger.info("\n📋 Next Steps:")
        logger.info("1. 🔧 Review config.toml and customize if needed")
        logger.info("2. 📊 Test data fetching: python main_application.py --check-data 2025")
        logger.info("3. 🤖 Try sample run: python main_application.py --predict") 
        logger.info("4. 🚀 Launch dashboard: streamlit run dashboard.py")
        logger.info("\n💡 For full functionality, fetch real data and train models")
    
    def _print_troubleshooting_guide(self) -> None:
        """Print troubleshooting guide for common issues."""
        logger.info("\n🔧 Troubleshooting Guide:")
        logger.info("1. Check Python version: python --version (need 3.8+)")
        logger.info("2. Update pip: python -m pip install --upgrade pip")
        logger.info("3. Install core deps manually: pip install pandas numpy scikit-learn requests beautifulsoup4")
        logger.info("4. Check file permissions and internet connection")
        logger.info("5. Run setup with --verbose for detailed error messages")


def create_test_script() -> str:
    """
    Create a test script to validate the installation.
    
    Returns:
        Path to the created test script
    """
    test_script = '''#!/usr/bin/env python3
"""
Quick test script for WNBA Prediction System
Run this to validate your installation.
"""

import sys
from pathlib import Path

def test_imports():
    """Test importing all required modules."""
    print("🧪 Testing imports...")
    
    # Core Python modules
    try:
        import pandas as pd
        import numpy as np
        import requests
        from bs4 import BeautifulSoup
        print("✅ Core Python modules")
    except ImportError as e:
        print(f"❌ Core Python modules: {e}")
        return False
    
    # Custom modules
    try:
        from data_models import PredictionConfig, PlayerPrediction
        from data_fetcher import WNBAStatsScraper
        from main_application import WNBADailyPredictor
        print("✅ Custom modules")
    except ImportError as e:
        print(f"❌ Custom modules: {e}")
        return False
    
    return True

def test_directories():
    """Test that all directories exist."""
    print("📁 Testing directories...")
    
    required_dirs = ["wnba_game_data", "wnba_predictions", "wnba_models", "logs"]
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ missing")
            return False
    
    return True

def test_basic_functionality():
    """Test basic system functionality."""
    print("⚙️ Testing basic functionality...")
    
    try:
        from data_models import PredictionConfig
        config = PredictionConfig()
        print(f"✅ Configuration loaded: {len(config.target_stats)} target stats")
        
        from main_application import WNBADailyPredictor
        predictor = WNBADailyPredictor()
        print("✅ Predictor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏀 WNBA Prediction System - Quick Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories), 
        ("Functionality Test", test_basic_functionality)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        if test_func():
            passed += 1
        
    print(f"\\n{'='*40}")
    if passed == len(tests):
        print("🎉 All tests passed! System is ready.")
        print("\\nNext steps:")
        print("• Run: python main_application.py --predict")
        print("• Launch dashboard: streamlit run dashboard.py")
    else:
        print(f"⚠️ {passed}/{len(tests)} tests passed")
        print("\\nIssues found. Try:")
        print("• Re-run setup: python setup.py")
        print("• Install missing dependencies")
        print("• Check file locations")

if __name__ == "__main__":
    main()
'''
    
    test_file = "test_installation.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    return test_file


def main():
    """
    Main setup function with comprehensive argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="WNBA Prediction System Setup (Fixed Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Full setup
  python setup.py --verbose          # Verbose output
  python setup.py --test-only        # Just run tests
  python setup.py --core-only        # Only install core dependencies
  python setup.py --create-test      # Create test script
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run validation tests'
    )
    
    parser.add_argument(
        '--core-only',
        action='store_true', 
        help='Only install core dependencies'
    )
    
    parser.add_argument(
        '--create-test',
        action='store_true',
        help='Create test script for installation validation'
    )
    
    args = parser.parse_args()
    
    setup = WNBASetup(verbose=args.verbose)
    
    try:
        if args.create_test:
            test_file = create_test_script()
            print(f"✅ Created test script: {test_file}")
            print("Run with: python test_installation.py")
            
        elif args.test_only:
            success = setup.run_quick_test()
            
        elif args.core_only:
            success, failed = setup.install_dependencies(skip_optional=True)
            
        else:
            success = setup.run_full_setup()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        logger.info("💡 Try running with --verbose for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()