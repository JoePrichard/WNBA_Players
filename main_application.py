# main_application.py - Enhanced WNBA Daily Game Prediction System
#!/usr/bin/env python3
"""
Enhanced WNBA Daily Game Prediction System

This module provides a modular, easy-to-understand prediction system with:
- Clear separation of concerns between data, models, and predictions
- Comprehensive error handling and logging
- Flexible configuration management
- Multiple data source support with fallbacks
- Clean, readable code structure

Main Classes:
    DataManager: Handles all data-related operations
    ModelManager: Manages ML model training and persistence
    PredictionManager: Generates and exports predictions
    WNBADailyPredictor: Main orchestrator class
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import argparse
import sys
import os
from pathlib import Path
import json
import random
from dataclasses import asdict
from team_mapping import TeamNameMapper

# Add project directory to path if needed
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Enhanced imports with proper error handling
try:
    from data_models import (
        GameSchedule, PlayerPrediction, PredictionConfig, HomeAway,
        WNBADataError, WNBAModelError, WNBAPredictionError,
    )
    DATA_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing data_models: {e}")
    DATA_MODELS_AVAILABLE = False

try:
    from data_fetcher import WNBAStatsScraper
    DATA_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"Error importing data_fetcher: {e}")
    DATA_FETCHER_AVAILABLE = False

try:
    from schedule_fetcher import WNBAScheduleFetcher
    SCHEDULE_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"Error importing schedule_fetcher: {e}")
    SCHEDULE_FETCHER_AVAILABLE = False

try:
    from prediction_models import WNBAPredictionModel
    PREDICTION_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing prediction_models: {e}")
    PREDICTION_MODELS_AVAILABLE = False

try:
    from utils import setup_project_structure, setup_logging
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class DataManager:
    """
    Manages all data-related operations including fetching, loading, and validation.
    
    This class handles:
    - Historical data fetching for training
    - Schedule data for predictions
    - Data validation and cleaning
    - File management and persistence
    """
    
    def __init__(self, data_dir: str = "wnba_game_data"):
        """
        Initialize data manager.
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.DataManager")
        
        # Initialize data fetchers
        self.stats_scraper = None
        self.schedule_fetcher = None
        
        if DATA_FETCHER_AVAILABLE:
            try:
                self.stats_scraper = WNBAStatsScraper()
                self.logger.info("✅ Stats scraper initialized")
            except Exception as e:
                self.logger.warning(f"❌ Stats scraper failed: {e}")
        
        if SCHEDULE_FETCHER_AVAILABLE:
            try:
                self.schedule_fetcher = WNBAScheduleFetcher()
                self.logger.info("✅ Schedule fetcher initialized")
            except Exception as e:
                self.logger.warning(f"❌ Schedule fetcher failed: {e}")
    
    def check_data_availability(self, year: int) -> Dict[str, Any]:
        """
        Check what data is available for a given year.
        
        Args:
            year: Year to check
            
        Returns:
            Dictionary with availability information
        """
        self.logger.info(f"Checking data availability for {year}")
        
        availability = {
            'year': year,
            'files_found': [],
            'total_records': 0,
            'date_range': None,
            'teams_found': [],
            'data_quality': 'unknown'
        }
        
        try:
            # Look for data files
            pattern = f"*{year}*.csv"
            data_files = list(self.data_dir.glob(pattern))
            
            for file_path in data_files:
                try:
                    df = pd.read_csv(file_path)
                    file_info = {
                        'filename': file_path.name,
                        'records': len(df),
                        'columns': len(df.columns),
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    availability['files_found'].append(file_info)
                    availability['total_records'] += len(df)
                    
                    # Extract teams if possible
                    team_columns = [col for col in df.columns if 'team' in col.lower()]
                    for col in team_columns:
                        teams = df[col].dropna().unique()
                        availability['teams_found'].extend(teams)
                    
                except Exception as e:
                    self.logger.debug(f"Error reading {file_path}: {e}")
            
            # Remove duplicates from teams
            availability['teams_found'] = list(set(availability['teams_found']))
            
            # Assess data quality
            if availability['total_records'] > 1000:
                availability['data_quality'] = 'good'
            elif availability['total_records'] > 100:
                availability['data_quality'] = 'fair'
            else:
                availability['data_quality'] = 'poor'
            
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
        
        return availability
    
    def fetch_season_data(self, year: int, force_refresh: bool = False) -> Dict[str, str]:
        """
        Fetch historical season data for training.
        
        Args:
            year: Year to fetch data for
            force_refresh: Whether to force re-fetch existing data
            
        Returns:
            Dictionary mapping data types to file paths
        """
        self.logger.info(f"Fetching season data for {year}")
        
        file_paths = {}
        
        # Check if we already have data
        if not force_refresh:
            existing_files = list(self.data_dir.glob(f"*{year}*.csv"))
            if existing_files:
                self.logger.info(f"Found {len(existing_files)} existing files for {year}")
                for file_path in existing_files:
                    file_paths[file_path.stem] = str(file_path)
                return file_paths
        
        # Fetch new data
        if not self.stats_scraper:
            raise WNBADataError("Stats scraper not available")
        
        try:
            # Define season date range
            start_date = f"{year}-05-01"  # WNBA typically starts in May
            end_date = f"{year}-10-31"    # Ends in October
            
            self.logger.info(f"Fetching data from {start_date} to {end_date}")
            
            # Fetch player stats
            stats_df = self.stats_scraper.scrape_date_range(start_date, end_date)
            
            if not stats_df.empty:
                filename = f"wnba_stats_{year}.csv"
                file_path = self.stats_scraper.save_to_csv(stats_df, filename)
                file_paths["player_stats"] = file_path
                self.logger.info(f"✅ Saved {len(stats_df)} records to {filename}")
            else:
                self.logger.warning(f"❌ No data found for {year}")
        
        except Exception as e:
            self.logger.error(f"Failed to fetch season data: {e}")
            raise WNBADataError(f"Season data fetch failed: {e}")
        
        return file_paths
    
    def load_game_logs(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load game logs from file with validation.
        
        Args:
            file_path: Specific file to load, or None for auto-detection
            
        Returns:
            Cleaned and validated game logs DataFrame
        """
        self.logger.info("Loading game logs")
        
        # Find file to load
        if file_path and Path(file_path).exists():
            target_file = Path(file_path)
        else:
            # Find most recent data file
            data_files = list(self.data_dir.glob("*.csv"))
            if not data_files:
                raise WNBADataError(f"No data files found in {self.data_dir}")
            
            # Prefer files with 'stats' in name, then by modification time
            stats_files = [f for f in data_files if 'stats' in f.name.lower()]
            if stats_files:
                target_file = max(stats_files, key=lambda x: x.stat().st_mtime)
            else:
                target_file = max(data_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"Loading data from: {target_file.name}")
        
        try:
            # Load data
            df = pd.read_csv(target_file)
            
            if df.empty:
                raise WNBADataError("Data file is empty")
            
            # Clean and validate data
            df = self._clean_game_logs(df)
            
            self.logger.info(f"✅ Loaded {len(df)} records for {df.get('player', df.iloc[:, 0]).nunique()} entities")
            
            return df
            
        except Exception as e:
            raise WNBADataError(f"Failed to load game logs: {e}")
    
    def _clean_game_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize game logs data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning game logs data")
        
        df = df.copy()
        
        # Standard column mapping
        column_map = {
            'Player': 'player',
            'Team': 'team', 
            'Date': 'date',
            'Opponent': 'opponent',
            'PTS': 'points',
            'TRB': 'total_rebounds',
            'AST': 'assists',
            'MP': 'minutes',
            'Home/Away': 'home_away'
        }
        
        # Apply column mapping
        for old_col, new_col in column_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['player', 'team', 'date']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            # Try to create from available data
            if 'player' not in df.columns:
                # Look for any name-like column
                name_columns = [col for col in df.columns if any(word in col.lower() for word in ['name', 'player'])]
                if name_columns:
                    df['player'] = df[name_columns[0]]
                else:
                    df['player'] = 'Unknown Player'
            if 'team' not in df.columns:
                raise ValueError("Missing 'team' column in game logs and cannot infer team. Only real teams from team_mapping.py are allowed.")
            if 'date' not in df.columns:
                df['date'] = datetime.now().date()
        
        # Clean data types
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            
            # Convert numeric columns
            numeric_columns = ['points', 'total_rebounds', 'assists', 'minutes']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = 0.0
            
            # Clean text columns
            if 'player' in df.columns:
                df['player'] = df['player'].astype(str).str.strip()
            
            if 'team' in df.columns:
                df['team'] = df['team'].astype(str).str.upper()
            
        except Exception as e:
            self.logger.warning(f"Data cleaning had issues: {e}")
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        self.logger.info(f"Data cleaning complete: {len(df)} records")
        return df
    
    def get_todays_schedule(self) -> Tuple[List[GameSchedule], bool]:
        """
        Get today's real-world schedule.
        Raises WNBADataError if the schedule fetcher is unavailable or
        returns an empty/placeholder payload.
        """
        today = date.today()
        self.logger.info(f"Getting schedule for {today}")

        if not self.schedule_fetcher:
            raise WNBADataError("Schedule fetcher unavailable – cannot build predictions")

        games_data = self.schedule_fetcher.get_games_for_date(today)
        if not games_data:
            raise WNBADataError(f"No schedule returned for {today}")

        schedule = []
        for g in games_data:
            if not g.get("is_real_data", True):
                raise WNBADataError("Received placeholder schedule – aborting")

            schedule.append(
                GameSchedule(
                    game_id=f"{g['date']}_{g['away_team']}_{g['home_team']}",
                    date=datetime.strptime(g['date'], "%Y-%m-%d").date(),
                    home_team=g['home_team'],
                    away_team=g['away_team'],
                    game_time=g.get("game_time", "TBD"),
                    status=g.get("status", "scheduled"),
                )
            )
        return schedule, True

    def _create_sample_schedule(self, target_date: date) -> List[GameSchedule]:
        """Create sample game schedule for testing."""
        self.logger.warning("Creating sample schedule data")
        
        teams = sorted(TeamNameMapper.all_abbreviations())
        
        games = []
        for i in range(min(3, len(teams) // 2)):
            home_team = teams[i * 2]
            away_team = teams[i * 2 + 1]
            
            game = GameSchedule(
                game_id=f"sample_{target_date}_{away_team}_{home_team}",
                date=target_date,
                home_team=home_team,
                away_team=away_team,
                game_time="7:00 PM",
                status="scheduled"
            )
            games.append(game)
        
        return games


class ModelManager:
    """
    Manages ML model training, persistence, and loading.
    
    This class handles:
    - Model training orchestration
    - Model persistence and versioning
    - Performance monitoring
    - Model loading and validation
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None, model_dir: str = "wnba_models"):
        """
        Initialize model manager.
        
        Args:
            config: Prediction configuration
            model_dir: Directory for storing models
        """
        self.config = config or PredictionConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        
        self.prediction_model = None
        if PREDICTION_MODELS_AVAILABLE:
            try:
                self.prediction_model = WNBAPredictionModel(config=self.config, model_save_dir=str(self.model_dir))
                self.logger.info("✅ Prediction model initialized")
            except Exception as e:
                self.logger.error(f"❌ Prediction model failed: {e}")
    
    def train_models(self, game_logs_df: pd.DataFrame, save_models: bool = True) -> Dict[str, Any]:
        """
        Train prediction models.
        
        Args:
            game_logs_df: Training data
            save_models: Whether to save trained models
            
        Returns:
            Training metrics and results
        """
        if not self.prediction_model:
            raise WNBAModelError("Prediction model not available")
        
        self.logger.info("Starting model training")
        
        # Validate data requirements
        self._validate_training_data(game_logs_df)
        
        try:
            # Train models
            metrics = self.prediction_model.train_all_models(game_logs_df)
            
            # Save models if requested
            if save_models:
                model_path = self.prediction_model.save_models()
                self.logger.info(f"✅ Models saved to: {model_path}")
            
            # Analyze training results
            training_summary = self._analyze_training_results(metrics)
            
            return {
                'metrics': metrics,
                'summary': training_summary,
                'model_path': model_path if save_models else None
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise WNBAModelError(f"Training failed: {e}")
    
    def _validate_training_data(self, df: pd.DataFrame) -> None:
        """Validate training data meets requirements."""
        if len(df) < 50:
            raise WNBAModelError(f"Insufficient data: {len(df)} records (need at least 50)")
        
        required_columns = ['player', 'points', 'total_rebounds', 'assists']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise WNBAModelError(f"Missing required columns: {missing_columns}")
        
        if df['player'].nunique() < 5:
            raise WNBAModelError(f"Insufficient players: {df['player'].nunique()} (need at least 5)")
    
    def _analyze_training_results(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze training results and provide insights."""
        summary = {
            'overall_performance': 'unknown',
            'best_models': {},
            'warnings': [],
            'recommendations': []
        }
        
        try:
            avg_scores = {}
            
            for stat, stat_metrics in metrics.items():
                if not stat_metrics:
                    summary['warnings'].append(f"No models trained for {stat}")
                    continue
                
                # Find best model for this statistic
                best_model = max(stat_metrics.items(), key=lambda x: x[1].r2_score)
                summary['best_models'][stat] = {
                    'model': best_model[0],
                    'r2_score': best_model[1].r2_score,
                    'mae': best_model[1].mae
                }
                
                # Calculate average R² for this stat
                avg_r2 = np.mean([m.r2_score for m in stat_metrics.values()])
                avg_scores[stat] = avg_r2
                
                # Check for suspicious scores (data leakage)
                if avg_r2 > 0.95:
                    summary['warnings'].append(f"Suspiciously high R² for {stat}: {avg_r2:.3f} - possible data leakage")
            
            # Overall performance assessment
            if avg_scores:
                overall_r2 = np.mean(list(avg_scores.values()))
                if overall_r2 > 0.8:
                    summary['overall_performance'] = 'excellent'
                elif overall_r2 > 0.6:
                    summary['overall_performance'] = 'good'
                elif overall_r2 > 0.4:
                    summary['overall_performance'] = 'fair'
                else:
                    summary['overall_performance'] = 'poor'
                
                # Generate recommendations
                if overall_r2 < 0.5:
                    summary['recommendations'].append("Consider feature engineering or more data")
                if len(summary['warnings']) > 0:
                    summary['recommendations'].append("Review feature engineering for data leakage")
                if overall_r2 > 0.7:
                    summary['recommendations'].append("Models ready for production use")
            
        except Exception as e:
            self.logger.warning(f"Error analyzing training results: {e}")
        
        return summary
    
    def load_latest_models(self) -> None:
        """Load the most recent trained models."""
        if not self.prediction_model:
            raise WNBAModelError("Prediction model not available")
        
        try:
            model_dirs = list(self.model_dir.glob("models_*"))
            
            if not model_dirs:
                raise WNBAModelError("No trained models found")
            
            # Get most recent model
            latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            
            self.prediction_model.load_models(str(latest_model_dir))
            self.logger.info(f"✅ Loaded models from: {latest_model_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise WNBAModelError(f"Model loading failed: {e}")
    
    def is_trained(self) -> bool:
        """Check if models are trained and ready."""
        return self.prediction_model and self.prediction_model.is_trained


class PredictionManager:
    """
    Manages prediction generation and export.
    
    This class handles:
    - Daily prediction generation
    - Player prediction logic
    - Result formatting and export
    - Prediction validation
    """
    
    def __init__(self, output_dir: str = "wnba_predictions"):
        """
        Initialize prediction manager.
        
        Args:
            output_dir: Directory for prediction outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.PredictionManager")
    
    def generate_game_predictions(
        self, 
        schedule: List[GameSchedule], 
        model_manager: ModelManager,
        data_manager: DataManager
    ) -> List[PlayerPrediction]:
        """
        Generate predictions for scheduled games.
        Only real player names and real data are used. No sample/demo predictions.
        """
        if not schedule:
            self.logger.warning("No games in schedule")
            return []
        if not model_manager.is_trained():
            self.logger.error("Models not trained. Cannot generate predictions.")
            return []
        self.logger.info(f"Generating predictions for {len(schedule)} games")
        predictions = []
        try:
            # Load historical data for feature creation
            game_logs_df = data_manager.load_game_logs()
            # Create features
            features_df = model_manager.prediction_model.feature_engineer.create_all_features(game_logs_df)
            # Get latest features for each player
            latest_features = (
                features_df.sort_values('date')
                .groupby('player', as_index=False)
                .tail(1)
            )
            # Generate predictions for each game
            for game in schedule:
                game_predictions = self._predict_game_players(game, latest_features, model_manager)
                predictions.extend(game_predictions)
            self.logger.info(f"✅ Generated {len(predictions)} player predictions")
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return []
        return predictions
    
    def _predict_game_players(
        self, 
        game: GameSchedule, 
        features_df: pd.DataFrame, 
        model_manager: ModelManager
    ) -> List[PlayerPrediction]:
        """Generate predictions for players in a specific game."""
        game_predictions = []
        
        for team, is_home in [(game.home_team, True), (game.away_team, False)]:
            opponent = game.away_team if is_home else game.home_team
            
            # Get players for this team
            team_players = features_df[features_df['team'] == team]
            
            if team_players.empty:
                self.logger.debug(f"No players found for team {team}")
                continue
            
            # Select top players by minutes played
            team_players = team_players.sort_values('minutes', ascending=False).head(8)
            
            for _, player_row in team_players.iterrows():
                try:
                    # Generate prediction using model
                    pred = model_manager.prediction_model.predict_player_stats(player_row)
                    
                    # Create final prediction object
                    player_prediction = PlayerPrediction(
                        game_id=game.game_id,
                        player=str(player_row['player']),
                        team=team,
                        opponent=opponent,
                        home_away=HomeAway.HOME if is_home else HomeAway.AWAY,
                        predicted_points=round(pred.predicted_points, 1),
                        predicted_rebounds=round(pred.predicted_rebounds, 1),
                        predicted_assists=round(pred.predicted_assists, 1),
                        points_uncertainty=round(pred.points_uncertainty, 1),
                        rebounds_uncertainty=round(pred.rebounds_uncertainty, 1),
                        assists_uncertainty=round(pred.assists_uncertainty, 1),
                        confidence_score=round(pred.confidence_score, 2),
                        model_version=model_manager.prediction_model.model_version
                    )
                    
                    game_predictions.append(player_prediction)
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {player_row.get('player', 'Unknown')}: {e}")
                    continue
        
        return game_predictions
    
    def export_predictions(
        self, 
        predictions: List[PlayerPrediction], 
        is_real_data: bool = True,
        target_date: Optional[date] = None
    ) -> str:
        """
        Export predictions to CSV file.
        
        Args:
            predictions: List of predictions to export
            is_real_data: Whether predictions are based on real schedule data
            target_date: Date for predictions (defaults to today)
            
        Returns:
            Path to exported file
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Exporting {len(predictions)} predictions")
        
        # Convert to DataFrame
        prediction_dicts = [pred.to_dict() for pred in predictions]
        df = pd.DataFrame(prediction_dicts)
        
        # Add metadata
        df['export_timestamp'] = datetime.now()
        df['data_source_type'] = 'Real Schedule' if is_real_data else 'Sample Data'
        df['is_real_schedule'] = is_real_data
        
        # Generate filename
        timestamp = datetime.now().strftime("%H%M%S")
        data_type = "real" if is_real_data else "sample"
        filename = f"predictions_{target_date.strftime('%Y%m%d')}_{data_type}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Save file
        df.to_csv(filepath, index=False)
        
        if is_real_data:
            self.logger.info(f"✅ Exported {len(predictions)} real predictions to: {filepath}")
        else:
            self.logger.warning(f"⚠️ Exported {len(predictions)} sample predictions to: {filepath}")
        
        return str(filepath)


class WNBADailyPredictor:
    """
    Main orchestrator for the WNBA daily prediction system.
    
    This class coordinates between data management, model training,
    and prediction generation to provide a complete prediction pipeline.
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize the daily predictor.
        
        Args:
            config: Prediction configuration
        """
        self.config = config or PredictionConfig()
        
        # Initialize managers
        self.data_manager = DataManager()
        self.model_manager = ModelManager(self.config)
        self.prediction_manager = PredictionManager()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🏀 WNBA Daily Predictor initialized")
        
        # Setup project structure if utils available
        if UTILS_AVAILABLE:
            try:
                setup_project_structure()
                setup_logging(log_level="INFO", log_file="logs/wnba_main.log")
            except Exception as e:
                self.logger.warning(f"Utils setup failed: {e}")
    
    def run_full_pipeline(self, year: int, train_models: bool = True, predict_today: bool = True) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Args:
            year: Year to fetch training data for
            train_models: Whether to train new models
            predict_today: Whether to generate today's predictions
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"🚀 Starting full prediction pipeline for {year}")
        
        results = {
            'pipeline_start_time': datetime.now(),
            'year': year,
            'steps_completed': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Step 1: Check data availability
            self.logger.info("📊 Step 1: Checking data availability")
            availability = self.data_manager.check_data_availability(year)
            results['data_availability'] = availability
            results['steps_completed'].append('data_availability_check')
            
            # Step 2: Fetch season data if needed
            if availability['total_records'] < 100:
                self.logger.info("📥 Step 2: Fetching season data")
                try:
                    file_paths = self.data_manager.fetch_season_data(year)
                    results['data_files'] = file_paths
                    results['steps_completed'].append('data_fetch')
                except Exception as e:
                    error_msg = f"Data fetching failed: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            else:
                self.logger.info("📊 Step 2: Using existing data")
                results['steps_completed'].append('data_existing')
            
            # Step 3: Train models if requested
            if train_models:
                self.logger.info("🤖 Step 3: Training models")
                try:
                    game_logs_df = self.data_manager.load_game_logs()
                    training_results = self.model_manager.train_models(game_logs_df)
                    results['training_results'] = training_results
                    results['steps_completed'].append('model_training')
                    
                    # Check for warnings
                    if training_results['summary'].get('warnings'):
                        results['warnings'].extend(training_results['summary']['warnings'])
                        
                except Exception as e:
                    error_msg = f"Model training failed: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            else:
                self.logger.info("🤖 Step 3: Loading existing models")
                try:
                    self.model_manager.load_latest_models()
                    results['steps_completed'].append('model_loading')
                except Exception as e:
                    warning_msg = f"Model loading failed: {e}"
                    results['warnings'].append(warning_msg)
                    self.logger.warning(warning_msg)
            
            # Step 4: Generate today's predictions
            if predict_today:
                self.logger.info("🔮 Step 4: Generating predictions")
                try:
                    schedule, is_real_data = self.data_manager.get_todays_schedule()
                    
                    if schedule:
                        predictions = self.prediction_manager.generate_game_predictions(
                            schedule, self.model_manager, self.data_manager
                        )
                        
                        if predictions:
                            export_path = self.prediction_manager.export_predictions(
                                predictions, is_real_data
                            )
                            
                            results['predictions'] = {
                                'file_path': export_path,
                                'count': len(predictions),
                                'is_real_schedule': is_real_data,
                                'games': len(schedule)
                            }
                            results['steps_completed'].append('prediction_generation')
                            
                            if not is_real_data:
                                results['warnings'].append("Predictions based on sample schedule data")
                        else:
                            results['warnings'].append("No predictions generated")
                    else:
                        results['warnings'].append("No games scheduled for today")
                        
                except Exception as e:
                    error_msg = f"Prediction generation failed: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Pipeline completion
            results['pipeline_end_time'] = datetime.now()
            results['pipeline_duration'] = (results['pipeline_end_time'] - results['pipeline_start_time']).total_seconds()
            
            # Success assessment
            critical_steps = ['data_availability_check']
            if train_models:
                critical_steps.append('model_training')
            if predict_today:
                critical_steps.append('prediction_generation')
            
            critical_completed = [step for step in critical_steps if step in results['steps_completed']]
            results['success_rate'] = len(critical_completed) / len(critical_steps)
            
            if results['success_rate'] == 1.0:
                self.logger.info("🎉 Pipeline completed successfully!")
            elif results['success_rate'] >= 0.7:
                self.logger.info(f"⚠️ Pipeline completed with warnings ({results['success_rate']:.1%} success)")
            else:
                self.logger.error(f"❌ Pipeline failed ({results['success_rate']:.1%} success)")
            
        except Exception as e:
            error_msg = f"Pipeline failed with unexpected error: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results


def main():
    """Main function with enhanced command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced WNBA Daily Game Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_application.py --check-data 2025
  python main_application.py --fetch-data 2025
  python main_application.py --train 2025
  python main_application.py --predict
  python main_application.py --full-pipeline 2025
        """
    )
    
    # Action arguments
    parser.add_argument('--check-data', type=int, metavar='YEAR', help='Check data availability for year')
    parser.add_argument('--fetch-data', type=int, metavar='YEAR', help='Fetch season data for year')
    parser.add_argument('--train', type=int, metavar='YEAR', help='Train models using data from year')
    parser.add_argument('--predict', action='store_true', help='Generate predictions for today')
    parser.add_argument('--full-pipeline', type=int, metavar='YEAR', help='Run complete pipeline for year')
    
    # Options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-train', action='store_true', help='Skip training in full pipeline')
    parser.add_argument('--no-predict', action='store_true', help='Skip prediction in full pipeline')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    if UTILS_AVAILABLE:
        setup_logging(log_level=log_level)
    else:
        logging.basicConfig(level=getattr(logging, log_level))
    
    # Load configuration
    config = PredictionConfig()  # Use default for now
    
    print("🏀 Enhanced WNBA Daily Game Prediction System")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = WNBADailyPredictor(config=config)
        
        # Execute requested action
        if args.check_data:
            availability = predictor.data_manager.check_data_availability(args.check_data)
            print(f"\n📊 Data Availability for {args.check_data}:")
            print(f"   Files found: {len(availability['files_found'])}")
            print(f"   Total records: {availability['total_records']}")
            print(f"   Data quality: {availability['data_quality']}")
            if availability['teams_found']:
                print(f"   Teams: {', '.join(availability['teams_found'][:5])}{'...' if len(availability['teams_found']) > 5 else ''}")
        
        elif args.fetch_data:
            file_paths = predictor.data_manager.fetch_season_data(args.fetch_data)
            print(f"\n📁 Data Files for {args.fetch_data}:")
            for data_type, path in file_paths.items():
                print(f"   {data_type}: {path}")
        
        elif args.train:
            game_logs_df = predictor.data_manager.load_game_logs()
            results = predictor.model_manager.train_models(game_logs_df)
            print(f"\n🤖 Training Results:")
            print(f"   Performance: {results['summary'].get('overall_performance', 'unknown')}")
            if results['summary'].get('warnings'):
                print(f"   Warnings: {len(results['summary']['warnings'])}")
            if results['summary'].get('best_models'):
                print(f"   Best models trained for: {list(results['summary']['best_models'].keys())}")
        
        elif args.predict:
            # Attempt to load the latest trained models
            try:
                predictor.model_manager.load_latest_models()
                print("✅ Loaded latest models")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load models: {e}. Using sample predictions.")
            # Proceed with prediction generation
            schedule, is_real_data = predictor.data_manager.get_todays_schedule()
            if not schedule:
                print(f"\n📅 No games scheduled for today")
            else:
                predictions = predictor.prediction_manager.generate_game_predictions(
                    schedule, predictor.model_manager, predictor.data_manager
                )
                if predictions:
                    export_path = predictor.prediction_manager.export_predictions(predictions, is_real_data)
                    data_type = "real" if is_real_data else "sample"
                    print(f"\n🔮 Generated {len(predictions)} predictions ({data_type} schedule)")
                    print(f"   Exported to: {export_path}")
                    print(f"   Games: {len(schedule)}")
                else:
                    print(f"\n❌ No predictions generated")
        
        elif args.full_pipeline:
            results = predictor.run_full_pipeline(
                year=args.full_pipeline,
                train_models=not args.no_train,
                predict_today=not args.no_predict
            )
            
            print(f"\n🎉 Pipeline Results:")
            print(f"   Success rate: {results['success_rate']:.1%}")
            print(f"   Steps completed: {len(results['steps_completed'])}")
            
            if results.get('warnings'):
                print(f"   Warnings: {len(results['warnings'])}")
                for warning in results['warnings'][:3]:
                    print(f"     • {warning}")
            
            if results.get('errors'):
                print(f"   Errors: {len(results['errors'])}")
                for error in results['errors'][:3]:
                    print(f"     • {error}")
            
            if results.get('predictions'):
                pred_info = results['predictions']
                print(f"   Predictions: {pred_info['count']} players, {pred_info['games']} games")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"\n💥 Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()