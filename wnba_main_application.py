#!/usr/bin/env python3
"""
WNBA Daily Game Prediction System - Improved Main Application
Enhanced version with better error handling and data validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import argparse
import sys
import os
from pathlib import Path
import json
import random

from wnba_data_models import (
    GameSchedule, PlayerPrediction, PredictionConfig, HomeAway,
    WNBADataError, WNBAModelError, WNBAPredictionError
)
from wnba_data_fetcher import WNBADataFetcher
from wnba_prediction_models import WNBAPredictionModel


class WNBADailyPredictor:
    """
    Enhanced main application class for WNBA daily game predictions.
    
    This improved version handles:
    - Data type validation and conversion
    - Graceful fallbacks when real data is unavailable
    - Better error reporting and recovery
    - Sample data generation for development
    
    Attributes:
        config: Configuration for predictions
        data_fetcher: Component for fetching game data
        prediction_model: ML model for predictions
        data_dir: Directory for storing data files
        output_dir: Directory for prediction outputs
        logger: Logger instance
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        data_dir: str = "wnba_game_data",
        output_dir: str = "wnba_predictions"
    ):
        """
        Initialize the WNBA daily predictor.
        
        Args:
            config: Configuration object, uses defaults if None
            data_dir: Directory for storing game data
            output_dir: Directory for prediction outputs
        """
        self.config = config or PredictionConfig()
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Initialize components
        self.data_fetcher = WNBADataFetcher()
        self.prediction_model = WNBAPredictionModel(config=self.config)
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def check_data_availability(self, year: int) -> Dict[str, bool]:
        """
        Check what data is available for a given year.
        
        Args:
            year: Year to check
            
        Returns:
            Dictionary indicating available data types
        """
        self.logger.info(f"Checking data availability for {year}...")
        
        # Check online availability
        online_availability = self.data_fetcher.validate_data_availability(year)
        
        # Check local files
        local_files = {
            'schedule': False,
            'player_stats': False,
            'player_game_logs': False,
            'team_stats': False
        }
        
        data_path = Path(self.data_dir)
        if data_path.exists():
            for file_path in data_path.glob(f"*{year}*.csv"):
                filename = file_path.name.lower()
                if 'schedule' in filename:
                    local_files['schedule'] = True
                elif 'player' in filename and ('game' in filename or 'log' in filename):
                    local_files['player_game_logs'] = True
                elif 'player' in filename:
                    local_files['player_stats'] = True
                elif 'team' in filename:
                    local_files['team_stats'] = True
        
        # Combine online and local availability
        combined_availability = {
            'schedule': online_availability.get('schedule', False) or local_files['schedule'],
            'player_stats': online_availability.get('player_stats', False) or local_files['player_stats'],
            'player_game_logs': local_files['player_game_logs'],
            'team_stats': local_files['team_stats']
        }
        
        self.logger.info(f"Data availability for {year}: {combined_availability}")
        return combined_availability

    def fetch_season_data(self, year: int, force_refresh: bool = False) -> Dict[str, str]:
        """
        Fetch and save season data with improved handling.
        
        Args:
            year: Season year to fetch
            force_refresh: Whether to re-fetch existing data
            
        Returns:
            Dictionary mapping data types to file paths
            
        Raises:
            WNBADataError: If data fetching fails
        """
        self.logger.info(f"Fetching season data for {year}...")
        
        file_paths = {}
        
        try:
            # Check existing files first
            if not force_refresh:
                existing_files = list(Path(self.data_dir).glob(f"*{year}*.csv"))
                
                if existing_files:
                    self.logger.info(f"Found {len(existing_files)} existing data files for {year}")
                    for file_path in existing_files:
                        filename = file_path.name.lower()
                        if 'schedule' in filename:
                            file_paths["schedule"] = str(file_path)
                        elif 'player' in filename and ('game' in filename or 'log' in filename):
                            file_paths["player_game_logs"] = str(file_path)
                        elif 'player' in filename:
                            file_paths["player_stats"] = str(file_path)
                        elif 'team' in filename:
                            file_paths["team_stats"] = str(file_path)
                    
                    if file_paths:
                        return file_paths
            
            # Fetch new data
            self.logger.info("Fetching fresh data from source...")
            
            # Season schedule
            try:
                schedule = self.data_fetcher.fetch_season_schedule(year)
                if schedule:
                    schedule_path = self.data_fetcher.export_data(
                        schedule, f"schedule_{year}", self.data_dir
                    )
                    file_paths["schedule"] = schedule_path
                    self.logger.info(f"✅ Schedule: {len(schedule)} games")
                else:
                    self.logger.warning(f"No schedule data found for {year}")
            except WNBADataError as e:
                self.logger.warning(f"Schedule fetch failed: {e}")
            
            # Player stats
            try:
                player_stats = self.data_fetcher.fetch_player_season_stats(year)
                if not player_stats.empty:
                    player_path = self.data_fetcher.export_data(
                        player_stats, f"player_stats_{year}", self.data_dir
                    )
                    file_paths["player_stats"] = player_path
                    self.logger.info(f"✅ Player stats: {len(player_stats)} players")
                else:
                    self.logger.warning(f"No player stats found for {year}")
            except WNBADataError as e:
                self.logger.warning(f"Player stats fetch failed: {e}")
            
            # If we don't have real data, create sample data for development
            if not file_paths:
                self.logger.info(f"No real data available for {year}, creating sample data for development...")
                sample_data = self.data_fetcher.create_sample_player_data(year)
                sample_path = self.data_fetcher.export_data(
                    sample_data, f"sample_player_logs_{year}", self.data_dir
                )
                file_paths["player_game_logs"] = sample_path
                self.logger.info(f"✅ Created sample player data: {len(sample_data)} game logs")
            
            return file_paths
            
        except Exception as e:
            raise WNBADataError(f"Season data fetch failed: {e}")

    def load_game_logs(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load game logs from file with improved validation.
        
        Args:
            file_path: Specific file to load, or None to find most recent
            
        Returns:
            DataFrame with game logs
            
        Raises:
            WNBADataError: If no valid game logs found
        """
        if file_path and os.path.exists(file_path):
            target_file = file_path
        else:
            # Find most appropriate data file
            data_files = list(Path(self.data_dir).glob("*.csv"))
            
            if not data_files:
                raise WNBADataError(f"No data files found in {self.data_dir}")
            
            # Prioritize player game logs, then player stats, then team data
            player_log_files = [f for f in data_files if 'player' in f.name.lower() and ('log' in f.name.lower() or 'sample' in f.name.lower())]
            player_stat_files = [f for f in data_files if 'player' in f.name.lower() and 'stat' in f.name.lower()]
            team_files = [f for f in data_files if 'team' in f.name.lower()]
            
            if player_log_files:
                target_file = max(player_log_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Using player game logs: {target_file.name}")
            elif player_stat_files:
                target_file = max(player_stat_files, key=lambda x: x.stat().st_mtime)
                self.logger.warning(f"No player game logs found, using player stats: {target_file.name}")
            elif team_files:
                target_file = max(team_files, key=lambda x: x.stat().st_mtime)
                self.logger.warning(f"No player data found, attempting to use team data: {target_file.name}")
            else:
                target_file = max(data_files, key=lambda x: x.stat().st_mtime)
                self.logger.warning(f"Using most recent data file: {target_file.name}")
        
        try:
            df = pd.read_csv(target_file)
            self.logger.info(f"Loaded data: {len(df)} records from {target_file}")
            
            if df.empty:
                raise WNBADataError("Data file is empty")
            
            # Validate and convert data format
            df = self._validate_and_convert_data(df)
            
            return df
            
        except Exception as e:
            raise WNBADataError(f"Failed to load data from {target_file}: {e}")

    def _validate_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data to expected format for player game logs.
        
        Args:
            df: Raw DataFrame from file
            
        Returns:
            DataFrame in proper format for feature engineering
            
        Raises:
            WNBADataError: If data cannot be converted to proper format
        """
        self.logger.info("Validating and converting data format...")
        
        # Required columns for player game logs
        required_columns = ['player', 'team', 'date', 'opponent', 'home_away', 'minutes', 'points', 'rebounds', 'assists']
        
        # Check if we already have the right format
        if all(col in df.columns for col in required_columns):
            self.logger.info("Data is already in correct format")
            return df
        
        # Try to map common column variations
        column_mapping = {
            'player': ['player', 'Player', 'name', 'Name', 'player_name'],
            'team': ['team', 'Team', 'tm', 'Tm', 'team_id'],
            'date': ['date', 'Date', 'game_date', 'Date_game'],
            'opponent': ['opponent', 'Opponent', 'opp', 'Opp', 'vs'],
            'home_away': ['home_away', 'Home_Away', 'location', 'venue'],
            'minutes': ['minutes', 'Minutes', 'mp', 'MP', 'min'],
            'points': ['points', 'Points', 'pts', 'PTS'],
            'rebounds': ['rebounds', 'Rebounds', 'trb', 'TRB', 'reb'],
            'assists': ['assists', 'Assists', 'ast', 'AST']
        }
        
        # Apply column mapping
        for standard_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns and standard_col not in df.columns:
                    df = df.rename(columns={col: standard_col})
                    break
        
        # Additional handling for team columns with suffixes (e.g., 'team_x', 'team_y')
        if 'team' not in df.columns:
            suffix_team_cols = [c for c in df.columns if c.lower() in ['team_x', 'team_y', 'team1', 'team2']]
            if suffix_team_cols:
                # Prefer 'team_x' over others, but use the first found
                df = df.rename(columns={suffix_team_cols[0]: 'team'})
                self.logger.info(f"Renamed column '{suffix_team_cols[0]}' to 'team'")
        
        # Check if we have the required columns now
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # If we're missing critical columns, try to create reasonable defaults
            if 'player' not in df.columns:
                if 'team' in df.columns:
                    # This might be team-level data - we can't easily convert to player data
                    raise WNBADataError(f"Data appears to be team-level, not player-level. Missing columns: {missing_columns}")
                else:
                    raise WNBADataError(f"Missing required columns: {missing_columns}")
            
            # Try to fill in missing columns with reasonable defaults
            if 'date' not in df.columns:
                df['date'] = datetime.now().date()
                self.logger.warning("Missing date column, using current date")
            
            if 'opponent' not in df.columns:
                df['opponent'] = 'UNK'
                self.logger.warning("Missing opponent column, using 'UNK'")
            
            if 'home_away' not in df.columns:
                df['home_away'] = 'H'
                self.logger.warning("Missing home_away column, assuming home games")
            
            # Fill missing stats with 0
            for stat in ['minutes', 'points', 'rebounds', 'assists']:
                if stat not in df.columns:
                    df[stat] = 0.0
                    self.logger.warning(f"Missing {stat} column, using 0")
        
        # Add game number if missing
        if 'game_num' not in df.columns:
            df['game_num'] = df.groupby('player').cumcount() + 1
        
        # Add result if missing
        if 'result' not in df.columns:
            df['result'] = 'W'  # Default to wins
        
        # Ensure proper data types
        try:
            df['date'] = pd.to_datetime(df['date']).dt.date
        except:
            self.logger.warning("Could not convert date column, using current date")
            df['date'] = datetime.now().date()
        
        # Ensure numeric columns are numeric
        numeric_columns = ['minutes', 'points', 'rebounds', 'assists']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        self.logger.info(f"Data converted successfully: {len(df)} records for {df['player'].nunique()} players")
        return df

    def train_prediction_models(
        self, 
        game_logs_df: Optional[pd.DataFrame] = None,
        save_models: bool = True
    ) -> Dict[str, Dict[str, any]]:
        """
        Train prediction models with improved error handling.
        
        Args:
            game_logs_df: Game logs DataFrame, loads from file if None
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of training metrics
            
        Raises:
            WNBAModelError: If training fails
        """
        self.logger.info("Training prediction models...")
        
        try:
            # Load data if not provided
            if game_logs_df is None:
                game_logs_df = self.load_game_logs()
            
            # Validate minimum data requirements
            if len(game_logs_df) < 50:
                raise WNBAModelError(f"Insufficient data for training: {len(game_logs_df)} records (need at least 50)")
            
            if game_logs_df['player'].nunique() < 5:
                raise WNBAModelError(f"Insufficient players for training: {game_logs_df['player'].nunique()} players (need at least 5)")
            
            # Train models
            metrics = self.prediction_model.train_all_models(game_logs_df)
            
            # Save models if requested
            if save_models:
                model_path = self.prediction_model.save_models()
                self.logger.info(f"Models saved to: {model_path}")
            
            # Log performance summary
            self.logger.info("📊 Training Results:")
            for stat, stat_metrics in metrics.items():
                if stat_metrics:
                    avg_r2 = np.mean([m.r2_score for m in stat_metrics.values()])
                    avg_mae = np.mean([m.mae for m in stat_metrics.values()])
                    avg_brier = np.mean([m.brier_score for m in stat_metrics.values()])
                    
                    self.logger.info(f"  {stat}: R²={avg_r2:.3f}, MAE={avg_mae:.3f}, Brier={avg_brier:.3f}")
                else:
                    self.logger.warning(f"  {stat}: No models trained successfully")
            
            return metrics
            
        except Exception as e:
            raise WNBAModelError(f"Model training failed: {e}")

    def get_todays_schedule(self) -> List[GameSchedule]:
        """
        Get today's game schedule.
        
        Returns:
            List of GameSchedule objects for today
            
        Raises:
            WNBADataError: If schedule cannot be retrieved
        """
        try:
            return self.data_fetcher.get_todays_games()
        except WNBADataError as e:
            self.logger.error(f"Could not get today's schedule: {e}")
            # Return empty list instead of raising error
            return []

    def predict_daily_games(
        self, 
        target_date: Optional[date] = None,
        schedule: Optional[List[GameSchedule]] = None
    ) -> List[PlayerPrediction]:
        """
        Generate predictions for games with graceful fallbacks.
        
        Args:
            target_date: Date to predict for, defaults to today
            schedule: Game schedule, fetches if None
            
        Returns:
            List of PlayerPrediction objects
            
        Raises:
            WNBAPredictionError: If predictions cannot be generated
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Generating predictions for {target_date}")
        
        try:
            # Get schedule if not provided
            if schedule is None:
                if target_date == date.today():
                    schedule = self.get_todays_schedule()
                else:
                    # For demo purposes, create a sample schedule
                    self.logger.info(f"Creating sample schedule for {target_date}")
                    schedule = self._create_sample_schedule(target_date)
            
            if not schedule:
                self.logger.warning(f"No games scheduled for {target_date}")
                return []
            
            # Check if models are trained
            if not self.prediction_model.is_trained:
                self.logger.info("Models not trained, attempting to load existing models...")
                try:
                    self._load_latest_models()
                except Exception as e:
                    self.logger.error(f"Could not load models: {e}")
                    raise WNBAPredictionError("No trained models available. Train models first.")
            
            # Generate predictions using trained models
            self.logger.info("Generating predictions using trained models")
            all_predictions = self._generate_model_predictions(schedule, target_date)
            
            self.logger.info(f"Generated {len(all_predictions)} player predictions")
            return all_predictions
            
        except Exception as e:
            raise WNBAPredictionError(f"Daily prediction failed: {e}")

    def _create_sample_schedule(self, target_date: date) -> List[GameSchedule]:
        """Create sample game schedule for development."""
        teams = ['LAS', 'NY', 'CHI', 'CONN', 'IND', 'PHX', 'SEA', 'ATL', 'DAL', 'MIN', 'WAS']
        
        # Create 3 sample games
        games = []
        for i in range(3):
            home_team = teams[i * 2]
            away_team = teams[i * 2 + 1]
            
            game = GameSchedule(
                game_id=f"{target_date}_{away_team}_{home_team}",
                date=target_date,
                home_team=home_team,
                away_team=away_team,
                game_time="7:00 PM",
                status="scheduled"
            )
            games.append(game)
        
        return games

    def _generate_model_predictions(self, schedule: List[GameSchedule], target_date: date) -> List[PlayerPrediction]:
        """Generate predictions for scheduled games using trained models."""
        predictions: List[PlayerPrediction] = []
 
        try:
            # Load historical game logs for feature creation
            game_logs_df = self.load_game_logs()
            if game_logs_df.empty:
                self.logger.warning("No historical game logs available – falling back to sample predictions")
                return self._generate_sample_predictions(schedule, target_date)
 
            # Ensure proper dtypes and engineered features
            try:
                features_df = self.prediction_model.feature_engineer.create_all_features(game_logs_df)
            except Exception as fe_err:
                self.logger.error(f"Feature engineering failed: {fe_err}")
                return self._generate_sample_predictions(schedule, target_date)
 
            # Keep each player's most recent feature row
            latest_features = (
                features_df.sort_values('date')
                .groupby('player', as_index=False)
                .tail(1)
            )
 
            # Iterate over each game and team
            for game in schedule:
                for team, is_home in [(game.home_team, True), (game.away_team, False)]:
                    opponent_raw = game.away_team if is_home else game.home_team
 
                    # Map full team names to 3-letter abbreviations used in player logs
                    name_to_abbrev = {
                        'Atlanta Dream': 'ATL',
                        'Chicago Sky': 'CHI',
                        'Connecticut Sun': 'CONN',
                        'Dallas Wings': 'DAL',
                        'Indiana Fever': 'IND',
                        'Los Angeles Sparks': 'LAS',
                        'Las Vegas Aces': 'LAS',
                        'Minnesota Lynx': 'MIN',
                        'New York Liberty': 'NY',
                        'Phoenix Mercury': 'PHX',
                        'Seattle Storm': 'SEA',
                        'Washington Mystics': 'WAS',
                        'Golden State Valkyries': 'GSV',
                    }
 
                    team_abbrev = name_to_abbrev.get(team, team)
                    opponent_abbrev = name_to_abbrev.get(opponent_raw, opponent_raw)
 
                    team_players_df = latest_features[latest_features['team'] == team_abbrev]
 
                    if team_players_df.empty:
                        self.logger.warning(f"No player data found for team {team}; skipping team predictions")
                        continue
 
                    # Select up to 8 players with most minutes in the latest game
                    team_players_df = team_players_df.sort_values('minutes', ascending=False).head(8)
 
                    for _, player_row in team_players_df.iterrows():
                        # Predict stats using ensemble model
                        try:
                            player_pred_raw = self.prediction_model.predict_player_stats(player_row)
                        except Exception as pred_err:
                            self.logger.warning(f"Prediction failed for {player_row['player']}: {pred_err}")
                            continue
 
                        # Build final PlayerPrediction with correct context
                        pred = PlayerPrediction(
                            game_id=game.game_id,
                            player=player_row['player'],
                            team=team_abbrev,
                            opponent=opponent_abbrev,
                            home_away=HomeAway.HOME if is_home else HomeAway.AWAY,
                            predicted_points=round(player_pred_raw.predicted_points, 1),
                            predicted_rebounds=round(player_pred_raw.predicted_rebounds, 1),
                            predicted_assists=round(player_pred_raw.predicted_assists, 1),
                            points_uncertainty=round(player_pred_raw.points_uncertainty, 1),
                            rebounds_uncertainty=round(player_pred_raw.rebounds_uncertainty, 1),
                            assists_uncertainty=round(player_pred_raw.assists_uncertainty, 1),
                            confidence_score=round(player_pred_raw.confidence_score, 2),
                            model_version=self.prediction_model.model_version
                        )
                        predictions.append(pred)
 
        except Exception as e:
            self.logger.error(f"Model-based prediction generation failed: {e}")
            return self._generate_sample_predictions(schedule, target_date)
 
        return predictions

    def _generate_sample_predictions(self, schedule: List[GameSchedule], target_date: date) -> List[PlayerPrediction]:
        """Generate sample predictions for development."""
        # Sample players for each team
        team_players = {
            'LAS': ['A\'ja Wilson', 'Kelsey Plum', 'Jackie Young'],
            'NY': ['Sabrina Ionescu', 'Breanna Stewart', 'Jonquel Jones'],
            'CHI': ['Chennedy Carter', 'Angel Reese', 'Dana Evans'],
            'CONN': ['Alyssa Thomas', 'DeWanna Bonner', 'DiJonai Carrington'],
            'IND': ['Caitlin Clark', 'Aliyah Boston', 'Kelsey Mitchell'],
            'PHX': ['Diana Taurasi', 'Brittney Griner', 'Kahleah Copper'],
            'SEA': ['Jewell Loyd', 'Nneka Ogwumike', 'Skylar Diggins-Smith'],
            'ATL': ['Rhyne Howard', 'Tina Charles', 'Allisha Gray'],
            'DAL': ['Arike Ogunbowale', 'Satou Sabally', 'Teaira McCowan'],
            'MIN': ['Napheesa Collier', 'Kayla McBride', 'Bridget Carleton'],
            'WAS': ['Ariel Atkins', 'Elena Delle Donne', 'Aaliyah Edwards']
        }
        
        predictions = []
        
        for game in schedule:
            # Generate predictions for players from both teams
            for team, is_home in [(game.home_team, True), (game.away_team, False)]:
                opponent = game.away_team if is_home else game.home_team
                home_away_str = 'H' if is_home else 'A'
                
                players = team_players.get(team, [f"Player 1", f"Player 2", f"Player 3"])
                
                for player in players:
                    # Generate realistic predictions based on player type
                    if any(star in player for star in ['Wilson', 'Stewart', 'Clark', 'Ionescu', 'Taurasi']):
                        # Star players
                        points = random.normalvariate(22, 3)
                        rebounds = random.normalvariate(8, 2)
                        assists = random.normalvariate(6, 2)
                        confidence = random.uniform(0.75, 0.95)
                    else:
                        # Role players
                        points = random.normalvariate(12, 3)
                        rebounds = random.normalvariate(5, 2)
                        assists = random.normalvariate(3, 1.5)
                        confidence = random.uniform(0.6, 0.8)
                    
                    # Ensure non-negative values
                    points = max(0, points)
                    rebounds = max(0, rebounds)
                    assists = max(0, assists)
                    
                    prediction = PlayerPrediction(
                        game_id=game.game_id,
                        player=player,
                        team=team,
                        opponent=opponent,
                        home_away=HomeAway.HOME if is_home else HomeAway.AWAY,
                        predicted_points=round(points, 1),
                        predicted_rebounds=round(rebounds, 1),
                        predicted_assists=round(assists, 1),
                        points_uncertainty=round(points * 0.2, 1),
                        rebounds_uncertainty=round(rebounds * 0.25, 1),
                        assists_uncertainty=round(assists * 0.3, 1),
                        confidence_score=round(confidence, 2),
                        model_version="sample_v1.0"
                    )
                    
                    predictions.append(prediction)
        
        return predictions

    def _load_latest_models(self) -> None:
        """Load the most recent trained models."""
        model_dirs = list(Path("wnba_models").glob("models_*"))
        
        if not model_dirs:
            raise WNBAModelError("No trained models found. Run training first.")
        
        # Get most recent model
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        self.prediction_model.load_models(str(latest_model_dir))
        self.logger.info(f"Loaded models from: {latest_model_dir}")

    def export_predictions(
        self, 
        predictions: List[PlayerPrediction],
        target_date: Optional[date] = None
    ) -> str:
        """
        Export predictions to CSV file.
        
        Args:
            predictions: List of PlayerPrediction objects
            target_date: Date for filename, uses today if None
            
        Returns:
            Path to exported file
        """
        if target_date is None:
            target_date = date.today()
        
        # Convert to DataFrame
        prediction_dicts = [pred.to_dict() for pred in predictions]
        df = pd.DataFrame(prediction_dicts)
        
        # Export file
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"predictions_{target_date.strftime('%Y%m%d')}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported {len(predictions)} predictions to: {filepath}")
        
        return filepath

    def run_full_pipeline(
        self, 
        year: int,
        train_models: bool = True,
        predict_today: bool = True
    ) -> Dict[str, any]:
        """
        Run the complete prediction pipeline with improved error handling.
        
        Args:
            year: Season year for training data
            train_models: Whether to train new models
            predict_today: Whether to generate today's predictions
            
        Returns:
            Dictionary with pipeline results
        """
        results = {}
        
        try:
            self.logger.info("🏀 Starting WNBA Prediction Pipeline")
            self.logger.info("=" * 50)
            
            # 1. Check data availability
            availability = self.check_data_availability(year)
            results['data_availability'] = availability
            
            # 2. Fetch season data
            try:
                file_paths = self.fetch_season_data(year)
                results['data_files'] = file_paths
            except WNBADataError as e:
                self.logger.error(f"Data fetching failed: {e}")
                results['data_error'] = str(e)
            
            # 3. Train models if requested and data available
            if train_models and 'data_files' in results:
                try:
                    metrics = self.train_prediction_models()
                    results['training_metrics'] = metrics
                except WNBAModelError as e:
                    self.logger.error(f"Model training failed: {e}")
                    results['training_error'] = str(e)
            
            # 4. Generate today's predictions if requested
            if predict_today:
                try:
                    predictions = self.predict_daily_games()
                    if predictions:
                        export_path = self.export_predictions(predictions)
                        results['predictions_file'] = export_path
                        results['num_predictions'] = len(predictions)
                    else:
                        results['predictions_note'] = "No games today"
                except WNBAPredictionError as e:
                    self.logger.error(f"Daily prediction failed: {e}")
                    results['prediction_error'] = str(e)
            
            # Summary
            if not any(key.endswith('_error') for key in results.keys()):
                self.logger.info("🎉 Pipeline completed successfully!")
            else:
                self.logger.info("⚠️ Pipeline completed with some issues")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['pipeline_error'] = str(e)
            return results


def main():
    """Main function with improved command line interface."""
    parser = argparse.ArgumentParser(
        description="WNBA Daily Game Prediction System - Improved Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wnba_main_application.py --check-data 2025
  python wnba_main_application.py --fetch-data 2025
  python wnba_main_application.py --train 2025
  python wnba_main_application.py --predict
  python wnba_main_application.py --full-pipeline 2025
        """
    )
    
    parser.add_argument(
        '--check-data',
        type=int,
        metavar='YEAR',
        help='Check data availability for year'
    )
    
    parser.add_argument(
        '--fetch-data',
        type=int,
        metavar='YEAR',
        help='Fetch season data for year'
    )
    
    parser.add_argument(
        '--train',
        type=int,
        metavar='YEAR',
        help='Train models using data from year'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate predictions for today'
    )
    
    parser.add_argument(
        '--full-pipeline',
        type=int,
        metavar='YEAR',
        help='Run complete pipeline for year'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = PredictionConfig(**config_dict)
    
    # Initialize predictor
    predictor = WNBADailyPredictor(config=config)
    
    print("🏀 WNBA Daily Game Prediction System")
    print("=" * 50)
    
    try:
        if args.check_data:
            availability = predictor.check_data_availability(args.check_data)
            print(f"\n📊 Data Availability for {args.check_data}:")
            for data_type, available in availability.items():
                status = "✅ Available" if available else "❌ Not Available"
                print(f"  {data_type}: {status}")
        
        elif args.fetch_data:
            file_paths = predictor.fetch_season_data(args.fetch_data)
            print(f"\n📁 Data Files for {args.fetch_data}:")
            for data_type, path in file_paths.items():
                print(f"  {data_type}: {path}")
        
        elif args.train:
            try:
                metrics = predictor.train_prediction_models()
                print(f"\n🤖 Training Complete for {args.train}")
                print("Performance Summary:")
                for stat, stat_metrics in metrics.items():
                    if stat_metrics:
                        print(f"  {stat}:")
                        for model, metric in stat_metrics.items():
                            print(f"    {model}: R²={metric.r2_score:.3f}, MAE={metric.mae:.3f}")
                    else:
                        print(f"  {stat}: No models trained successfully")
            except Exception as e:
                print(f"\n❌ Training failed: {e}")
        
        elif args.predict:
            predictions = predictor.predict_daily_games()
            if predictions:
                export_path = predictor.export_predictions(predictions)
                print(f"\n🔮 Generated {len(predictions)} predictions")
                print(f"   Exported to: {export_path}")
            else:
                print("\n📅 No games scheduled for today")
        
        elif args.full_pipeline:
            results = predictor.run_full_pipeline(args.full_pipeline)
            print(f"\n🎉 Pipeline Results:")
            print(json.dumps(results, indent=2, default=str))
        
        else:
            parser.print_help()
    
    except (WNBADataError, WNBAModelError, WNBAPredictionError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()