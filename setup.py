#!/usr/bin/env python3
"""
WNBA Prediction System Setup Script
Automated installation and configuration script with improved error handling.
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
    Setup manager for WNBA prediction system.
    
    This class handles the complete setup process including:
    - Python version validation
    - Directory creation
    - Dependency installation
    - Configuration file creation
    - Module validation
    - Quick functionality testing
    
    Attributes:
        verbose (bool): Enable verbose logging output
        required_python_version (Tuple[int, int]): Minimum required Python version
        project_dirs (List[str]): List of directories to create
        module_import_map (Dict[str, str]): Maps package names to import names
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
        
        # Map package names to their import names (fixes beautifulsoup4 issue)
        self.module_import_map = {
            "pandas": "pandas",
            "numpy": "numpy", 
            "sklearn": "sklearn",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "torch": "torch",
            "streamlit": "streamlit",
            "plotly": "plotly",
            "requests": "requests",
            "beautifulsoup4": "bs4",  # Fix: beautifulsoup4 package imports as bs4
            "python-dateutil": "dateutil",
            "toml": "toml"
        }
    
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
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """
        Install Python dependencies from requirements file.
        
        Args:
            requirements_file (str, optional): Path to requirements file. Defaults to "requirements.txt".
            
        Returns:
            bool: True if installation successful, False otherwise
        """
        logger.info("Installing Python dependencies...")
        
        if not os.path.exists(requirements_file):
            logger.error(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL if not self.verbose else None)
            
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
            
            if self.verbose:
                subprocess.check_call(cmd)
            else:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            logger.info("💡 Try running manually: pip install -r requirements.txt")
            return False
    
    def create_default_config(self) -> bool:
        """
        Create default configuration file.
        
        Returns:
            bool: True if config created successfully, False otherwise
        """
        logger.info("Creating default configuration...")
        
        try:
            from config_loader import ConfigLoader, WNBAConfig
            
            config = WNBAConfig()
            ConfigLoader.save_config(config, "config.toml")
            
            logger.info("✅ Created config.toml with default settings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create config: {e}")
            return False
    
    def validate_installation(self) -> Dict[str, bool]:
        """
        Validate that all components can be imported.
        
        Tests both external dependencies and custom modules to ensure
        the entire system is properly installed and configured.
        
        Returns:
            Dict[str, bool]: Dictionary mapping module names to success status
        """
        logger.info("Validating installation...")
        
        results = {}
        
        # Test external dependencies with correct import names
        for package_name, import_name in self.module_import_map.items():
            try:
                __import__(import_name)
                results[package_name] = True
                logger.info(f"  ✅ {package_name}")
            except ImportError:
                results[package_name] = False
                logger.warning(f"  ❌ {package_name}")
        
        # Test custom modules
        custom_modules = [
            "wnba_data_models",
            "wnba_data_fetcher", 
            "wnba_feature_engineer",
            "wnba_prediction_models",
            "wnba_main_application"
        ]
        
        for module in custom_modules:
            try:
                __import__(module)
                results[module] = True
                logger.info(f"  ✅ {module}")
            except ImportError as e:
                results[module] = False
                logger.warning(f"  ❌ {module}: {e}")
        
        all_good = all(results.values())
        if all_good:
            logger.info("✅ All modules validated successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"⚠️ Some modules failed validation: {failed}")
        
        return results
    
    def run_quick_test(self) -> bool:
        """
        Run a quick functionality test.
        
        Tests core system components to ensure they can be instantiated
        and basic operations work correctly.
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        logger.info("Running quick functionality test...")
        
        try:
            # Test data models
            from wnba_data_models import PredictionConfig, PlayerPrediction, HomeAway
            config = PredictionConfig()
            logger.info("  ✅ Data models")
            
            # Test configuration (simple creation, not file loading)
            try:
                from config_loader import ConfigLoader, WNBAConfig
                config = WNBAConfig()  # Create default config without file loading
                logger.info("  ✅ Configuration creation")
            except Exception as e:
                logger.warning(f"  ⚠️  Configuration loading: {e}")
                logger.info("  ✅ Configuration (basic)")
            
            # Test data fetcher initialization
            from wnba_data_fetcher import WNBADataFetcher
            fetcher = WNBADataFetcher()
            logger.info("  ✅ Data fetcher")
            
            # Test feature engineer
            from wnba_feature_engineer import WNBAFeatureEngineer
            engineer = WNBAFeatureEngineer()
            logger.info("  ✅ Feature engineer")
            
            # Test prediction model
            from wnba_prediction_models import WNBAPredictionModel
            model = WNBAPredictionModel()
            logger.info("  ✅ Prediction models")
            
            logger.info("✅ Quick functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """
        Create .env file template.
        
        Creates a template environment file with common configuration
        variables that users can customize for their setup.
        
        Returns:
            bool: True if template created successfully, False otherwise
        """
        logger.info("Creating environment file template...")
        
        env_content = """# WNBA Prediction System Environment Variables
# Copy this file to .env and customize as needed

# Data directories
WNBA_DATA_DIR=wnba_game_data
WNBA_OUTPUT_DIR=wnba_predictions  
WNBA_MODEL_DIR=wnba_models

# API rate limiting (seconds between requests)
WNBA_RATE_LIMIT=2.0

# Logging
WNBA_LOG_LEVEL=INFO
WNBA_LOG_FILE=logs/wnba_predictions.log

# Optional: Custom user agent for web scraping
# WNBA_USER_AGENT=Your-Bot-Name/1.0
"""
        
        try:
            with open(".env.template", "w") as f:
                f.write(env_content)
            
            logger.info("✅ Created .env.template")
            logger.info("💡 Copy to .env and customize as needed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create .env template: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """
        Run complete setup process.
        
        Executes all setup steps in the correct order and provides
        comprehensive feedback on success/failure of each step.
        
        Returns:
            bool: True if setup completed successfully, False if any issues occurred
        """
        logger.info("🏀 Starting WNBA Prediction System Setup")
        logger.info("=" * 50)
        
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Project Directories", self.create_directories),
            ("Dependencies", self.install_dependencies),
            ("Configuration", self.create_default_config),
            ("Environment Template", self.setup_environment_file),
            ("Validation", lambda: all(self.validate_installation().values())),
            ("Quick Test", self.run_quick_test)
        ]
        
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n🔧 {step_name}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    logger.warning(f"⚠️ {step_name} had issues")
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"❌ {step_name} failed: {e}")
        
        # Summary
        logger.info("\n" + "=" * 50)
        if not failed_steps:
            logger.info("🎉 Setup completed successfully!")
            logger.info("✅ All components ready")
            self._print_next_steps()
            return True
        else:
            logger.warning(f"⚠️ Setup completed with {len(failed_steps)} issues")
            logger.warning(f"Failed steps: {failed_steps}")
            logger.info("💡 Check error messages above for troubleshooting")
            return False
    
    def _print_next_steps(self) -> None:
        """
        Print next steps for the user.
        
        Provides clear guidance on what to do after successful setup.
        """
        logger.info("\n📋 Next Steps:")
        logger.info("1. 🔧 Review config.toml and customize if needed")
        logger.info("2. 📊 Fetch WNBA data: python wnba_main_application.py --fetch-data 2025")
        logger.info("3. 🤖 Train models: python wnba_main_application.py --train 2025") 
        logger.info("4. 🔮 Generate predictions: python wnba_main_application.py --predict")
        logger.info("5. 🚀 Launch dashboard: streamlit run wnba_dashboard.py")
        logger.info("\n💡 Or run full pipeline: python wnba_main_application.py --full-pipeline 2025")


def create_diagnostic_report() -> Dict[str, any]:
    """
    Create a diagnostic report for troubleshooting.
    
    Generates a comprehensive report of the system state to help
    diagnose issues when setup fails.
    
    Returns:
        Dict[str, any]: Diagnostic information including Python version,
                       installed packages, file structure, etc.
    """
    import platform
    import pkg_resources
    
    report = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'installed_packages': {},
        'file_structure': {},
        'permissions': {}
    }
    
    # Check installed packages
    try:
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        report['installed_packages'] = installed_packages
    except Exception as e:
        report['installed_packages'] = f"Error: {e}"
    
    # Check file structure
    try:
        current_files = list(Path('.').glob('*.py'))
        report['file_structure'] = [str(f) for f in current_files]
    except Exception as e:
        report['file_structure'] = f"Error: {e}"
    
    return report


def main():
    """
    Main setup function with comprehensive argument parsing.
    
    Provides multiple setup modes and diagnostic capabilities
    for flexible installation and troubleshooting.
    """
    parser = argparse.ArgumentParser(
        description="WNBA Prediction System Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Full setup
  python setup.py --verbose          # Verbose output
  python setup.py --test-only        # Just run tests
  python setup.py --deps-only        # Only install dependencies
  python setup.py --diagnostic       # Generate diagnostic report
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
        '--deps-only',
        action='store_true', 
        help='Only install dependencies'
    )
    
    parser.add_argument(
        '--diagnostic',
        action='store_true',
        help='Generate diagnostic report for troubleshooting'
    )
    
    args = parser.parse_args()
    
    setup = WNBASetup(verbose=args.verbose)
    
    try:
        if args.diagnostic:
            report = create_diagnostic_report()
            import json
            print("🔍 DIAGNOSTIC REPORT")
            print("=" * 30)
            print(json.dumps(report, indent=2, default=str))
            
        elif args.test_only:
            success = setup.run_quick_test()
            
        elif args.deps_only:
            success = setup.install_dependencies()
            
        else:
            success = setup.run_full_setup()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        logger.info("💡 Run with --diagnostic for troubleshooting info")
        sys.exit(1)


if __name__ == "__main__":
    main()