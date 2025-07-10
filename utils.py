# utils.py - WNBA Prediction System Utilities
"""
Utility functions for the WNBA Prediction System.

This module provides common functionality used across the application including:
- Directory management and setup
- File path utilities
- Data validation helpers
- Configuration utilities
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime, date
from team_mapping import TeamNameMapper


def ensure_directories_exist(directories: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.debug(f"Ensured directory exists: {directory}")


def get_project_directories() -> List[str]:
    """
    Get list of all project directories.
    
    Returns:
        List of directory paths used by the system
    """
    return [
        "wnba_game_data",
        "wnba_predictions", 
        "wnba_models",
        "wnba_validation",
        "logs"
    ]


def setup_project_structure() -> None:
    """Set up complete project directory structure."""
    directories = get_project_directories()
    ensure_directories_exist(directories)
    logging.info(f"Project structure initialized: {len(directories)} directories")


def add_project_to_path() -> None:
    """Add current project directory to Python path."""
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logging.debug(f"Added to Python path: {current_dir}")


def validate_team_abbreviation(team: str) -> bool:
    """
    Validate WNBA team abbreviation using centralized mapping.
    """
    return TeamNameMapper.is_valid_abbreviation(team)


def standardize_team_name(team: str) -> str:
    """
    Standardize team abbreviation or name to preferred abbreviation using centralized mapping.
    Raises ValueError if the team cannot be mapped.
    """
    abbr = TeamNameMapper.to_abbreviation(team)
    if abbr:
        return abbr
    logging.error(f"Unknown team: {team}. Only real teams from team_mapping.py are allowed.")
    raise ValueError(f"Unknown team: {team}. Only real teams from team_mapping.py are allowed.")


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if pd.isna(value) or value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    if pd.isna(value) or value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clean_player_name(name: str) -> str:
    """
    Clean and standardize player name.
    
    Args:
        name: Raw player name
        
    Returns:
        Cleaned player name
    """
    if not name or pd.isna(name):
        return "Unknown Player"
    
    # Remove extra whitespace
    name = str(name).strip()
    
    # Remove common suffixes
    suffixes = [' Jr.', ' Sr.', ' II', ' III', ' IV']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    return name


def get_latest_file(directory: str, pattern: str = "*.csv") -> Optional[Path]:
    """
    Get the most recently modified file matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        Path to latest file or None if not found
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        return None
    
    files = list(directory_path.glob(pattern))
    if not files:
        return None
    
    return max(files, key=lambda x: x.stat().st_mtime)


def create_game_id(date: Union[date, str], away_team: str, home_team: str) -> str:
    """
    Create standardized game identifier.
    
    Args:
        date: Game date
        away_team: Away team abbreviation
        home_team: Home team abbreviation
        
    Returns:
        Standardized game ID
    """
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            date = datetime.now().date()
    
    date_str = date.strftime('%Y%m%d')
    away_clean = standardize_team_name(away_team)
    home_clean = standardize_team_name(home_team)
    
    return f"{date_str}_{away_clean}_{home_clean}"


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log helpful information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name for logging identification
    """
    logging.info(f"{name} info:")
    logging.info(f"  Shape: {df.shape}")
    logging.info(f"  Columns: {list(df.columns)}")
    
    if not df.empty:
        logging.info(f"  Date range: {df.get('date', pd.Series()).min()} to {df.get('date', pd.Series()).max()}")
        logging.info(f"  Players: {df.get('player', pd.Series()).nunique()}")
        logging.info(f"  Teams: {df.get('team', pd.Series()).nunique()}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to log to
    """
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Set up formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        ensure_directories_exist([str(Path(log_file).parent)])
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))


def format_prediction_summary(predictions: List[Dict[str, Any]]) -> str:
    """
    Format prediction results for display.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Formatted summary string
    """
    if not predictions:
        return "No predictions available"
    
    total_predictions = len(predictions)
    avg_confidence = sum(p.get('confidence_score', 0) for p in predictions) / total_predictions
    
    games = set(p.get('game_id', '') for p in predictions)
    players = set(p.get('player', '') for p in predictions)
    
    summary = f"""Prediction Summary:
  • {total_predictions} player predictions
  • {len(games)} games
  • {len(players)} players
  • Average confidence: {avg_confidence:.1%}"""
    
    return summary


def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate data quality and return report.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    report = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'shape': df.shape,
        'missing_columns': [],
        'null_percentages': {},
        'duplicate_rows': 0
    }
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        report['valid'] = False
        report['missing_columns'] = missing_cols
        report['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct > 0:
            report['null_percentages'][col] = null_pct
            if null_pct > 50:
                report['issues'].append(f"Column '{col}' has {null_pct:.1f}% null values")
            elif null_pct > 10:
                report['warnings'].append(f"Column '{col}' has {null_pct:.1f}% null values")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report['duplicate_rows'] = duplicate_count
        report['warnings'].append(f"{duplicate_count} duplicate rows found")
    
    return report


def mmss_to_float(val):
    try:
        import pandas as pd
        if pd.isnull(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val)
        if ':' in val_str:
            parts = val_str.split(':')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]) + int(parts[1]) / 60.0
        return float(val_str) if val_str.replace('.', '', 1).isdigit() else 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    """Demo and testing of utility functions."""
    print("🔧 WNBA Prediction System - Utilities Demo")
    print("=" * 40)
    
    # Test directory setup
    setup_project_structure()
    print("✅ Project structure initialized")
    
    # Test team validation
    test_teams = list(TeamNameMapper.all_abbreviations()) + ['InvalidTeam', 'Los Angeles Sparks']
    for team in test_teams:
        valid = validate_team_abbreviation(team)
        try:
            standardized = standardize_team_name(team)
            print(f"Team: {team} → Valid: {valid}, Standardized: {standardized}")
        except ValueError as e:
            print(f"Team: {team} → Valid: {valid}, Standardization Error: {e}")
    
    # Test game ID creation
    game_id = create_game_id('2025-06-26', 'NY', 'LAS')
    print(f"Game ID: {game_id}")
    
    print("\n🎉 Utilities working correctly!")