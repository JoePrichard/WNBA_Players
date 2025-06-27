#!/usr/bin/env python3
"""
Enhanced Configuration Management for WNBA Prediction System

This module provides comprehensive configuration management with:
- Type-safe configuration classes with validation
- TOML file loading and saving
- Environment variable overrides
- Configuration validation and error handling
- Default value management

The configuration system supports:
- Prediction model parameters
- Data fetching settings
- Validation criteria
- Dashboard preferences
- Logging configuration
"""

import os
import toml
from typing import Dict, Any, Optional, List, Union, Type, get_type_hints
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime

from data_models import WNBAConfigurationError


@dataclass
class PredictionConfig:
    """
    Configuration for prediction models and algorithms.
    
    Controls the core prediction behavior including target statistics,
    minimum data requirements, and confidence thresholds.
    
    Attributes:
        target_stats (List[str]): Statistics to predict (e.g., ['points', 'rebounds', 'assists'])
        min_games_for_prediction (int): Minimum games required for making predictions
        confidence_threshold (float): Minimum confidence score for valid predictions (0.0-1.0)
        max_uncertainty (float): Maximum allowed prediction uncertainty
        lookback_games (int): Number of recent games to use for form analysis
        min_games_for_stats (int): Minimum games required for statistical calculations
    """
    target_stats: List[str] = field(default_factory=lambda: ["points", "total_rebounds", "assists"])
    min_games_for_prediction: int = 5
    confidence_threshold: float = 0.6
    max_uncertainty: float = 10.0
    lookback_games: int = 5
    min_games_for_stats: int = 3
    
    def validate(self) -> None:
        """
        Validate prediction configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        if not self.target_stats:
            raise WNBAConfigurationError("target_stats cannot be empty")
        
        valid_stats = {"points", "total_rebounds", "assists", "steals", "blocks", "turnovers"}
        invalid_stats = set(self.target_stats) - valid_stats
        if invalid_stats:
            raise WNBAConfigurationError(f"Invalid target stats: {invalid_stats}")
        
        if not 1 <= self.min_games_for_prediction <= 50:
            raise WNBAConfigurationError(f"min_games_for_prediction must be 1-50: {self.min_games_for_prediction}")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise WNBAConfigurationError(f"confidence_threshold must be 0.0-1.0: {self.confidence_threshold}")
        
        if self.max_uncertainty <= 0:
            raise WNBAConfigurationError(f"max_uncertainty must be positive: {self.max_uncertainty}")


@dataclass 
class ModelConfig:
    """
    Configuration for machine learning models and ensemble weights.
    
    Defines the models used in the ensemble and their relative importance,
    as well as hyperparameters for each model type.
    
    Attributes:
        weights (Dict[str, float]): Ensemble weights for each model type
        xgboost (Dict[str, Any]): XGBoost hyperparameters
        lightgbm (Dict[str, Any]): LightGBM hyperparameters  
        random_forest (Dict[str, Any]): Random Forest hyperparameters
        neural_network (Dict[str, Any]): Neural Network architecture and training parameters
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.30,
        "lightgbm": 0.25, 
        "random_forest": 0.25,
        "neural_network": 0.20
    })
    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbosity": 0
    })
    lightgbm: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbosity": -1
    })
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 8,
        "random_state": 42,
        "n_jobs": -1
    })
    neural_network: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dims": [128, 64, 32, 16],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "max_epochs": 100,
        "patience": 15,
        "batch_size": 32
    })
    
    def validate(self) -> None:
        """
        Validate model configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        # Validate weights
        if not self.weights:
            raise WNBAConfigurationError("Model weights cannot be empty")
        
        weight_sum = sum(self.weights.values())
        if weight_sum <= 0:
            raise WNBAConfigurationError("Model weights must sum to positive value")
        
        for model, weight in self.weights.items():
            if weight < 0:
                raise WNBAConfigurationError(f"Model weight cannot be negative: {model}={weight}")
        
        # Validate XGBoost parameters
        if self.xgboost.get("n_estimators", 0) <= 0:
            raise WNBAConfigurationError("XGBoost n_estimators must be positive")
        
        if not 1 <= self.xgboost.get("max_depth", 0) <= 20:
            raise WNBAConfigurationError("XGBoost max_depth must be 1-20")
        
        # Validate Neural Network parameters
        if not isinstance(self.neural_network.get("hidden_dims", []), list):
            raise WNBAConfigurationError("Neural network hidden_dims must be a list")
        
        if len(self.neural_network.get("hidden_dims", [])) == 0:
            raise WNBAConfigurationError("Neural network must have at least one hidden layer")


@dataclass
class DataConfig:
    """
    Configuration for data fetching, storage, and processing.
    
    Controls how data is fetched from external sources, where it's stored,
    and how requests are rate-limited to be respectful to data providers.
    
    Attributes:
        base_url (str): Base URL for data fetching (Basketball Reference)
        rate_limit_delay (float): Seconds to wait between HTTP requests
        max_retries (int): Maximum retry attempts for failed requests
        user_agent (str): User agent string for HTTP requests
        data_dir (str): Directory for storing raw data files
        output_dir (str): Directory for prediction outputs
        model_dir (str): Directory for saved models
        validation_dir (str): Directory for validation results
        timeout_seconds (int): HTTP request timeout in seconds
    """
    base_url: str = "https://www.basketball-reference.com/wnba"
    rate_limit_delay: float = 2.0
    max_retries: int = 3
    user_agent: str = "WNBA-Analytics-Bot/1.0"
    data_dir: str = "wnba_game_data"
    output_dir: str = "wnba_predictions"
    model_dir: str = "wnba_models"
    validation_dir: str = "wnba_validation"
    timeout_seconds: int = 30
    
    def validate(self) -> None:
        """
        Validate data configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        if not self.base_url:
            raise WNBAConfigurationError("base_url cannot be empty")
        
        if not self.base_url.startswith(("http://", "https://")):
            raise WNBAConfigurationError("base_url must be a valid URL")
        
        if self.rate_limit_delay < 0:
            raise WNBAConfigurationError("rate_limit_delay cannot be negative")
        
        if self.max_retries < 0:
            raise WNBAConfigurationError("max_retries cannot be negative")
        
        if not self.user_agent:
            raise WNBAConfigurationError("user_agent cannot be empty")
        
        if self.timeout_seconds <= 0:
            raise WNBAConfigurationError("timeout_seconds must be positive")
    
    def ensure_directories_exist(self) -> None:
        """Create all configured directories if they don't exist."""
        directories = [self.data_dir, self.output_dir, self.model_dir, self.validation_dir]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@dataclass
class ValidationConfig:
    """
    Configuration for model validation and performance evaluation.
    
    Controls how models are validated using time series cross-validation
    and what performance benchmarks must be met.
    
    Attributes:
        test_weeks (int): Number of weeks to use for each test period
        min_train_weeks (int): Minimum weeks of training data required
        step_weeks (int): Step size between validation periods
        n_validation_periods (int): Maximum number of validation periods
        target_brier_score (float): Target Brier score for good calibration
        min_r2_score (float): Minimum R² score for acceptable performance
        target_coverage (float): Target 95% confidence interval coverage
        early_stopping_patience (int): Validation periods to wait before stopping
    """
    test_weeks: int = 4
    min_train_weeks: int = 8
    step_weeks: int = 2
    n_validation_periods: int = 10
    target_brier_score: float = 0.12
    min_r2_score: float = 0.5
    target_coverage: float = 0.95
    early_stopping_patience: int = 3
    
    def validate(self) -> None:
        """
        Validate validation configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        if self.test_weeks <= 0:
            raise WNBAConfigurationError("test_weeks must be positive")
        
        if self.min_train_weeks <= self.test_weeks:
            raise WNBAConfigurationError("min_train_weeks must be greater than test_weeks")
        
        if self.step_weeks <= 0:
            raise WNBAConfigurationError("step_weeks must be positive")
        
        if not 0.0 < self.target_brier_score < 1.0:
            raise WNBAConfigurationError("target_brier_score must be between 0 and 1")
        
        if not 0.0 <= self.min_r2_score <= 1.0:
            raise WNBAConfigurationError("min_r2_score must be between 0 and 1")
        
        if not 0.0 < self.target_coverage < 1.0:
            raise WNBAConfigurationError("target_coverage must be between 0 and 1")


@dataclass
class DashboardConfig:
    """
    Configuration for the Streamlit dashboard interface.
    
    Controls the appearance and behavior of the web-based dashboard
    for viewing predictions and system status.
    
    Attributes:
        page_title (str): Browser tab title
        page_icon (str): Browser tab icon (emoji or URL)
        layout (str): Streamlit layout mode ('wide' or 'centered')
        initial_sidebar_state (str): Initial sidebar state ('expanded' or 'collapsed')
        default_confidence_threshold (float): Default confidence filter in UI
        show_uncertainties (bool): Whether to display uncertainty estimates by default
        auto_refresh_minutes (int): Auto-refresh interval in minutes (0 = disabled)
        max_players_display (int): Maximum players to show per team
    """
    page_title: str = "WNBA Daily Game Predictions"
    page_icon: str = "🏀"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    default_confidence_threshold: float = 0.7
    show_uncertainties: bool = True
    auto_refresh_minutes: int = 60
    max_players_display: int = 12
    
    def validate(self) -> None:
        """
        Validate dashboard configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        if not self.page_title:
            raise WNBAConfigurationError("page_title cannot be empty")
        
        if self.layout not in ["wide", "centered"]:
            raise WNBAConfigurationError(f"layout must be 'wide' or 'centered': {self.layout}")
        
        if self.initial_sidebar_state not in ["expanded", "collapsed"]:
            raise WNBAConfigurationError(f"initial_sidebar_state must be 'expanded' or 'collapsed': {self.initial_sidebar_state}")
        
        if not 0.0 <= self.default_confidence_threshold <= 1.0:
            raise WNBAConfigurationError("default_confidence_threshold must be between 0 and 1")
        
        if self.auto_refresh_minutes < 0:
            raise WNBAConfigurationError("auto_refresh_minutes cannot be negative")
        
        if self.max_players_display <= 0:
            raise WNBAConfigurationError("max_players_display must be positive")


@dataclass
class LoggingConfig:
    """
    Configuration for logging and monitoring.
    
    Controls how the system logs events, errors, and performance metrics
    for debugging and monitoring purposes.
    
    Attributes:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        format (str): Log message format string
        log_file (str): Path to log file (empty string = console only)
        max_log_size_mb (int): Maximum log file size in MB before rotation
        backup_count (int): Number of backup log files to keep
        enable_performance_logging (bool): Whether to log performance metrics
        log_predictions (bool): Whether to log individual predictions
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "wnba_predictions.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    enable_performance_logging: bool = True
    log_predictions: bool = False
    
    def validate(self) -> None:
        """
        Validate logging configuration parameters.
        
        Raises:
            WNBAConfigurationError: If any parameter is invalid
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise WNBAConfigurationError(f"Invalid log level: {self.level}")
        
        if not self.format:
            raise WNBAConfigurationError("Log format cannot be empty")
        
        if self.max_log_size_mb <= 0:
            raise WNBAConfigurationError("max_log_size_mb must be positive")
        
        if self.backup_count < 0:
            raise WNBAConfigurationError("backup_count cannot be negative")


@dataclass
class WNBAConfig:
    """
    Main configuration class containing all sub-configurations.
    
    This is the root configuration object that contains all other
    configuration sections. It provides validation and serialization
    for the entire system configuration.
    
    Attributes:
        prediction (PredictionConfig): Prediction model configuration
        models (ModelConfig): Machine learning model configuration
        data (DataConfig): Data fetching and storage configuration
        validation (ValidationConfig): Model validation configuration
        dashboard (DashboardConfig): Dashboard interface configuration
        logging (LoggingConfig): Logging and monitoring configuration
        version (str): Configuration version for compatibility checking
        created_at (datetime): When this configuration was created
    """
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> None:
        """
        Validate entire configuration.
        
        Calls validate() on all sub-configurations to ensure
        the complete configuration is valid.
        
        Raises:
            WNBAConfigurationError: If any configuration section is invalid
        """
        try:
            self.prediction.validate()
            self.models.validate()
            self.data.validate()
            self.validation.validate()
            self.dashboard.validate()
            self.logging.validate()
        except Exception as e:
            raise WNBAConfigurationError(f"Configuration validation failed: {e}")
    
    def setup_directories(self) -> None:
        """Create all necessary directories for the configuration."""
        self.data.ensure_directories_exist()
        
        # Create logs directory if logging to file
        if self.logging.log_file:
            log_dir = Path(self.logging.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    """
    Advanced configuration loader with validation and environment support.
    
    This class handles loading configuration from TOML files, applying
    environment variable overrides, and providing comprehensive validation
    and error handling.
    
    Features:
    - TOML file loading with error handling
    - Environment variable overrides
    - Configuration validation
    - Default value management
    - Configuration migration/upgrade support
    """
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        "WNBA_DATA_DIR": ("data", "data_dir"),
        "WNBA_OUTPUT_DIR": ("data", "output_dir"),
        "WNBA_MODEL_DIR": ("data", "model_dir"),
        "WNBA_RATE_LIMIT": ("data", "rate_limit_delay"),
        "WNBA_LOG_LEVEL": ("logging", "level"),
        "WNBA_LOG_FILE": ("logging", "log_file"),
        "WNBA_USER_AGENT": ("data", "user_agent"),
        "WNBA_CONFIDENCE_THRESHOLD": ("prediction", "confidence_threshold"),
        "WNBA_MIN_GAMES": ("prediction", "min_games_for_prediction"),
    }
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> WNBAConfig:
        """
        Load configuration from file with environment overrides.
        
        Args:
            config_path (Optional[str]): Path to TOML config file. 
                                       Defaults to 'config.toml' if None.
                                       
        Returns:
            WNBAConfig: Validated configuration object
            
        Raises:
            WNBAConfigurationError: If configuration loading or validation fails
        """
        if config_path is None:
            config_path = "config.toml"
        
        # Start with defaults
        config = WNBAConfig()
        
        # Load from file if it exists
        if os.path.exists(config_path):
            try:
                toml_config = toml.load(config_path)
                config = ConfigLoader._merge_config(config, toml_config)
                logging.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                raise WNBAConfigurationError(f"Failed to load config from {config_path}: {e}")
        else:
            logging.warning(f"Config file not found: {config_path}, using defaults")
        
        # Apply environment variable overrides
        config = ConfigLoader._apply_env_overrides(config)
        
        # Validate final configuration
        try:
            config.validate()
        except Exception as e:
            raise WNBAConfigurationError(f"Configuration validation failed: {e}")
        
        # Setup directories
        config.setup_directories()
        
        return config
    
    @staticmethod
    def _merge_config(config: WNBAConfig, toml_config: Dict[str, Any]) -> WNBAConfig:
        """
        Merge TOML configuration with default config.
        
        Args:
            config (WNBAConfig): Base configuration object
            toml_config (Dict[str, Any]): TOML configuration dictionary
            
        Returns:
            WNBAConfig: Merged configuration object
        """
        # Update prediction configuration
        if "prediction" in toml_config:
            pred_config = toml_config["prediction"]
            ConfigLoader._update_dataclass_from_dict(config.prediction, pred_config)
        
        # Update model configuration
        if "models" in toml_config:
            models_config = toml_config["models"]
            ConfigLoader._update_dataclass_from_dict(config.models, models_config)
        
        # Update data configuration
        if "data" in toml_config:
            data_config = toml_config["data"]
            ConfigLoader._update_dataclass_from_dict(config.data, data_config)
        
        # Update validation configuration
        if "validation" in toml_config:
            val_config = toml_config["validation"]
            ConfigLoader._update_dataclass_from_dict(config.validation, val_config)
        
        # Update dashboard configuration
        if "dashboard" in toml_config:
            dash_config = toml_config["dashboard"]
            ConfigLoader._update_dataclass_from_dict(config.dashboard, dash_config)
        
        # Update logging configuration
        if "logging" in toml_config:
            log_config = toml_config["logging"]
            ConfigLoader._update_dataclass_from_dict(config.logging, log_config)
        
        # Update top-level fields
        if "version" in toml_config:
            config.version = toml_config["version"]
        
        return config
    
    @staticmethod
    def _update_dataclass_from_dict(obj: Any, update_dict: Dict[str, Any]) -> None:
        """
        Update dataclass object fields from dictionary.
        
        Args:
            obj (Any): Dataclass object to update
            update_dict (Dict[str, Any]): Dictionary with new values
        """
        type_hints = get_type_hints(type(obj))
        
        for key, value in update_dict.items():
            if hasattr(obj, key):
                # Type conversion if needed
                expected_type = type_hints.get(key)
                if expected_type and not isinstance(value, expected_type):
                    try:
                        if expected_type == float:
                            value = float(value)
                        elif expected_type == int:
                            value = int(value)
                        elif expected_type == bool:
                            value = bool(value)
                    except (ValueError, TypeError):
                        logging.warning(f"Could not convert {key}={value} to {expected_type}")
                        continue
                
                setattr(obj, key, value)
    
    @staticmethod
    def _apply_env_overrides(config: WNBAConfig) -> WNBAConfig:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config (WNBAConfig): Configuration object to modify
            
        Returns:
            WNBAConfig: Configuration with environment overrides applied
        """
        overrides_applied = []
        
        for env_var, (section, field) in ConfigLoader.ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    section_obj = getattr(config, section)
                    current_value = getattr(section_obj, field)
                    
                    # Convert environment string to appropriate type
                    if isinstance(current_value, bool):
                        new_value = env_value.lower() in ("true", "1", "yes", "on")
                    elif isinstance(current_value, int):
                        new_value = int(env_value)
                    elif isinstance(current_value, float):
                        new_value = float(env_value)
                    else:
                        new_value = env_value
                    
                    setattr(section_obj, field, new_value)
                    overrides_applied.append(f"{env_var}={new_value}")
                    
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to apply environment override {env_var}={env_value}: {e}")
        
        if overrides_applied:
            logging.info(f"Applied environment overrides: {overrides_applied}")
        
        return config
    
    @staticmethod
    def save_config(config: WNBAConfig, config_path: str = "config.toml") -> None:
        """
        Save configuration to TOML file.
        
        Args:
            config (WNBAConfig): Configuration object to save
            config_path (str): Path to save configuration file
            
        Raises:
            WNBAConfigurationError: If saving fails
        """
        try:
            # Validate before saving
            config.validate()
            
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Convert datetime objects to strings
            config_dict["created_at"] = config_dict["created_at"].isoformat()
            
            # Save to file
            with open(config_path, 'w') as f:
                toml.dump(config_dict, f)
            
            logging.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            raise WNBAConfigurationError(f"Failed to save config to {config_path}: {e}")
    
    @staticmethod
    def create_sample_config(config_path: str = "sample_config.toml") -> str:
        """
        Create a sample configuration file with documentation.
        
        Args:
            config_path (str): Path for the sample configuration file
            
        Returns:
            str: Path to the created sample configuration file
        """
        sample_content = '''# WNBA Prediction System Configuration
# This file contains all configurable parameters for the system

[prediction]
# Statistics to predict - can include: points, rebounds, assists, steals, blocks, turnovers
target_stats = ["points", "total_rebounds", "assists"]

# Minimum games a player must have played to generate predictions
min_games_for_prediction = 5

# Minimum confidence score (0.0-1.0) for displaying predictions
confidence_threshold = 0.6

# Maximum allowed uncertainty in predictions
max_uncertainty = 10.0

# Number of recent games to use for form analysis
lookback_games = 5

# Minimum games required for statistical calculations
min_games_for_stats = 3

[models]
# Ensemble weights for different model types (will be normalized)
[models.weights]
xgboost = 0.30
lightgbm = 0.25
random_forest = 0.25
neural_network = 0.20

# XGBoost hyperparameters
[models.xgboost]
n_estimators = 100
max_depth = 6
learning_rate = 0.1
random_state = 42
verbosity = 0

# LightGBM hyperparameters
[models.lightgbm]
n_estimators = 100
max_depth = 6
learning_rate = 0.1
random_state = 42
verbosity = -1

# Random Forest hyperparameters
[models.random_forest]
n_estimators = 100
max_depth = 8
random_state = 42
n_jobs = -1

# Neural Network architecture and training
[models.neural_network]
hidden_dims = [128, 64, 32, 16]
dropout_rate = 0.3
learning_rate = 0.001
max_epochs = 100
patience = 15
batch_size = 32

[data]
# Base URL for data fetching (Basketball Reference WNBA)
base_url = "https://www.basketball-reference.com/wnba"

# Rate limiting - seconds between requests (be respectful!)
rate_limit_delay = 2.0

# Maximum retry attempts for failed requests
max_retries = 3

# User agent for HTTP requests
user_agent = "WNBA-Analytics-Bot/1.0"

# Directory structure
data_dir = "wnba_game_data"
output_dir = "wnba_predictions"
model_dir = "wnba_models"
validation_dir = "wnba_validation"

# HTTP timeout in seconds
timeout_seconds = 30

[validation]
# Time series validation parameters
test_weeks = 4           # Weeks per test period
min_train_weeks = 8      # Minimum training weeks
step_weeks = 2           # Step between periods
n_validation_periods = 10

# Performance benchmarks
target_brier_score = 0.12  # Target Brier score (lower is better)
min_r2_score = 0.5         # Minimum R² score
target_coverage = 0.95     # Target 95% CI coverage

# Early stopping
early_stopping_patience = 3

[dashboard]
# Streamlit dashboard configuration
page_title = "WNBA Daily Game Predictions"
page_icon = "🏀"
layout = "wide"                    # "wide" or "centered"
initial_sidebar_state = "expanded" # "expanded" or "collapsed"

# Default UI settings
default_confidence_threshold = 0.7
show_uncertainties = true
auto_refresh_minutes = 60  # 0 = disabled
max_players_display = 12

[logging]
# Logging configuration
level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = "wnba_predictions.log"  # Empty string = console only
max_log_size_mb = 10
backup_count = 5
enable_performance_logging = true
log_predictions = false  # Can generate large logs

# Configuration metadata
version = "1.0"
'''
        
        with open(config_path, 'w') as f:
            f.write(sample_content)
        
        return config_path


def setup_logging(config: LoggingConfig) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config (LoggingConfig): Logging configuration object
    """
    import logging.handlers
    
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup formatter
    formatter = logging.Formatter(config.format)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Setup file handler if specified
    if config.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_log_size_mb * 1024 * 1024,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, config.level.upper()))


def main():
    """
    Demo and testing for configuration management.
    """
    print("🔧 WNBA Configuration Management - Demo")
    print("=" * 40)
    
    try:
        # Create sample configuration
        sample_path = ConfigLoader.create_sample_config()
        print(f"✅ Created sample configuration: {sample_path}")
        
        # Load configuration
        config = ConfigLoader.load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Target stats: {config.prediction.target_stats}")
        print(f"   Data directory: {config.data.data_dir}")
        print(f"   Model weights: {list(config.models.weights.keys())}")
        
        # Setup logging
        setup_logging(config.logging)
        logging.info("Logging configured successfully")
        
        # Test validation
        config.validate()
        print("✅ Configuration validation passed")
        
        # Save configuration
        ConfigLoader.save_config(config, "demo_config.toml")
        print("✅ Configuration saved to: demo_config.toml")
        
        print("\n🎉 Configuration system working correctly!")
        
    except WNBAConfigurationError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")


if __name__ == "__main__":
    main()