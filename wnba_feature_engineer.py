#!/usr/bin/env python3
"""
WNBA Feature Engineering
Transforms raw game data into features for machine learning models.
Based on research from successful sports prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import asdict

from wnba_data_models import (
    PlayerGameLog, TeamGameLog, HomeAway, GameResult,
    WNBADataError, PredictionConfig
)


class WNBAFeatureEngineer:
    """
    Creates features for WNBA prediction models based on research insights.
    
    Implements features from:
    - DARKO: Recent form, opponent strength, context
    - XGBoost Synergy: Team composition, player interactions
    - Neural Networks: Usage patterns, efficiency metrics
    
    Attributes:
        config: Configuration for feature engineering
        target_stats: Statistics to predict
        feature_columns: List of feature column names
        logger: Logger instance
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        lookback_games: int = 5,
        min_games_for_stats: int = 3
    ):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object, uses defaults if None
            lookback_games: Number of recent games for form calculation
            min_games_for_stats: Minimum games required for statistical features
        """
        self.config = config or PredictionConfig()
        self.lookback_games = lookback_games
        self.min_games_for_stats = min_games_for_stats
        self.target_stats = self.config.target_stats
        self.feature_columns: List[str] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input game log data.
        
        Args:
            df: Raw game log DataFrame
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            WNBADataError: If required columns are missing or data is invalid
        """
        required_columns = [
            'player', 'team', 'date', 'opponent', 'home_away',
            'minutes', 'points', 'rebounds', 'assists'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise WNBADataError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise WNBADataError("Input DataFrame is empty")
        
        # Convert data types
        df = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except (ValueError, TypeError) as e:
                raise WNBADataError(f"Could not convert date column: {e}")
        
        # Ensure numeric columns are numeric
        numeric_columns = ['minutes', 'points', 'rebounds', 'assists']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing target stats
        initial_rows = len(df)
        df = df.dropna(subset=['points', 'rebounds', 'assists'])
        final_rows = len(df)
        
        if final_rows < initial_rows:
            self.logger.warning(f"Removed {initial_rows - final_rows} rows with missing target stats")
        
        if df.empty:
            raise WNBADataError("No valid rows remaining after cleaning")
        
        # Sort by player and date
        df = df.sort_values(['player', 'date']).reset_index(drop=True)
        
        self.logger.info(f"Validated data: {len(df)} game logs for {df['player'].nunique()} players")
        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic statistical features.
        
        Args:
            df: Game log DataFrame
            
        Returns:
            DataFrame with basic features added
        """
        df = df.copy()
        
        # Ensure we have required columns with defaults
        default_columns = {
            'fg_made': 'points',  # Approximate from points
            'fg_attempted': 'points',
            'ft_made': 'points',
            'ft_attempted': 'points',
            'turnovers': 'assists',
            'steals': None,
            'blocks': None,
            'fouls': None
        }
        
        for col, source_col in default_columns.items():
            if col not in df.columns:
                if source_col and source_col in df.columns:
                    # Create reasonable estimates
                    if col == 'fg_made':
                        df[col] = df[source_col] / 2.2  # Average points per FG
                    elif col == 'fg_attempted':
                        df[col] = df[source_col] / 1.5  # Shooting percentage estimate
                    elif col == 'ft_made':
                        df[col] = df[source_col] * 0.15  # FT proportion of scoring
                    elif col == 'ft_attempted':
                        df[col] = df[source_col] * 0.2
                    elif col == 'turnovers':
                        df[col] = df[source_col] * 0.6  # TO to assist ratio
                else:
                    # Use position-based defaults
                    defaults_map = {
                        'steals': 1.2,
                        'blocks': 0.5,
                        'fouls': 2.5
                    }
                    df[col] = defaults_map.get(col, 0.0)
        
        # Fill any remaining missing values
        numeric_columns = ['fg_made', 'fg_attempted', 'ft_made', 'ft_attempted', 
                          'turnovers', 'steals', 'blocks', 'fouls']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Create efficiency metrics
        df['usage_rate'] = np.where(
            df['minutes'] > 0,
            (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) / df['minutes'],
            0.2
        )
        
        df['scoring_efficiency'] = np.where(
            df['fg_attempted'] > 0,
            df['points'] / df['fg_attempted'],
            1.0
        )
        
        df['assist_rate'] = np.where(
            df['minutes'] > 0,
            df['assists'] / df['minutes'],
            0.1
        )
        
        df['rebound_rate'] = np.where(
            df['minutes'] > 0,
            df['rebounds'] / df['minutes'],
            0.2
        )
        
        # True shooting percentage
        df['ts_pct'] = np.where(
            (df['fg_attempted'] + 0.44 * df['ft_attempted']) > 0,
            df['points'] / (2 * (df['fg_attempted'] + 0.44 * df['ft_attempted'])),
            0.5
        ).clip(0, 1)
        
        return df

    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recent form and momentum features (DARKO-inspired).
        
        Args:
            df: Game log DataFrame with basic features
            
        Returns:
            DataFrame with form features added
        """
        df = df.copy()
        
        # Calculate rolling averages for recent form
        for stat in self.target_stats:
            # Rolling mean for last N games
            df[f'{stat}_l{self.lookback_games}'] = (
                df.groupby('player')[stat]
                .rolling(window=self.lookback_games, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Season averages (expanding window)
            df[f'season_avg_{stat}'] = (
                df.groupby('player')[stat]
                .expanding(min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Momentum (recent form vs season average)
            df[f'{stat}_momentum'] = (
                df[f'{stat}_l{self.lookback_games}'] - df[f'season_avg_{stat}']
            )
        
        # Game number in season
        df['game_number_season'] = df.groupby('player').cumcount() + 1
        
        # Season progress (0-1 scale)
        max_games = df.groupby('player')['game_number_season'].transform('max')
        df['season_progress'] = df['game_number_season'] / max_games
        
        # Early/late season indicators
        df['early_season'] = (df['game_number_season'] <= 10).astype(int)
        df['late_season'] = (df['game_number_season'] >= 25).astype(int)
        
        return df

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game context features (rest, opponent, pace).
        
        Args:
            df: Game log DataFrame
            
        Returns:
            DataFrame with context features added
        """
        df = df.copy()
        
        # Rest days calculation
        df['prev_game_date'] = df.groupby('player')['date'].shift(1)
        df['rest_days'] = (df['date'] - df['prev_game_date']).dt.days
        df['rest_days'] = df['rest_days'].fillna(3).clip(0, 10)  # Cap at 10 days
        
        # Rest/fatigue indicators
        df['rest_advantage'] = (df['rest_days'] >= 2).astype(int)
        df['fatigue_factor'] = (df['rest_days'] == 0).astype(int)  # Back-to-back
        
        # Home court advantage
        df['home_boost'] = (df['home_away'] == 'H').astype(int)
        
        # Opponent strength (requires team stats)
        # For now, create placeholder - would need team defensive ratings
        if 'opp_def_rating' not in df.columns:
            # Create estimated opponent strength based on team performance
            team_strength = self._estimate_team_strength(df)
            df = df.merge(team_strength, left_on='opponent', right_index=True, how='left')
            df['opp_def_rating'] = df['opp_def_rating'].fillna(105.0)  # League average
        
        df['opp_strength'] = (df['opp_def_rating'] - 100) / 10  # Normalize around 0
        df['matchup_advantage'] = (df['opp_strength'] < 0).astype(int)
        
        # Pace factors (if available)
        if 'team_pace' not in df.columns:
            df['team_pace'] = 80.0  # WNBA average pace
        
        df['pace_factor'] = df['team_pace'] / 80.0  # Normalize to league average
        
        # Blowout risk (games likely to have different rotation patterns)
        df['blowout_risk'] = (np.abs(df['opp_strength']) > 1.5).astype(int)
        
        return df

    def create_synergy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team synergy and composition features (XGBoost Synergy-inspired).
        
        Args:
            df: Game log DataFrame
            
        Returns:
            DataFrame with synergy features added
        """
        df = df.copy()
        
        # Position-based features (infer from stats if not available)
        if 'position' not in df.columns:
            df['position'] = self._infer_positions(df)
        
        # Position dummy variables and expectations
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        position_weights = {
            'PG': {'ast_weight': 1.5, 'reb_weight': 0.7, 'pts_weight': 1.0},
            'SG': {'ast_weight': 0.8, 'reb_weight': 0.8, 'pts_weight': 1.3},
            'SF': {'ast_weight': 1.0, 'reb_weight': 1.0, 'pts_weight': 1.1},
            'PF': {'ast_weight': 0.7, 'reb_weight': 1.4, 'pts_weight': 1.0},
            'C': {'ast_weight': 0.5, 'reb_weight': 1.6, 'pts_weight': 0.9}
        }
        
        for pos in positions:
            df[f'pos_is_{pos}'] = (df['position'] == pos).astype(int)
            
            if pos in position_weights:
                weights = position_weights[pos]
                for stat, weight in weights.items():
                    stat_name = stat.replace('_weight', '')
                    df[f'{pos}_{stat_name}_expectation'] = df[f'pos_is_{pos}'] * weight
        
        # Team context features
        team_stats = self._calculate_team_context(df)
        df = df.merge(team_stats, on=['team', 'date'], how='left')
        
        # Player role within team
        df['usage_above_team'] = df['usage_rate'] - df['team_usage_mean'].fillna(0.2)
        
        df['primary_scorer'] = (
            df['season_avg_points'] / df['team_total_scoring'].fillna(80.0) > 0.25
        ).astype(int)
        
        df['primary_facilitator'] = (
            df['season_avg_assists'] / df['team_total_assists'].fillna(20.0) > 0.3
        ).astype(int)
        
        # Player share of team production
        df['player_share_pts'] = df['season_avg_points'] / (df['team_avg_pts'].fillna(15.0) + 0.1)
        df['player_share_reb'] = df['season_avg_rebounds'] / (df['team_avg_reb'].fillna(8.0) + 0.1)
        df['player_share_ast'] = df['season_avg_assists'] / (df['team_avg_ast'].fillna(4.0) + 0.1)
        
        return df

    def _estimate_team_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate team defensive strength from available data.
        
        Args:
            df: Game log DataFrame
            
        Returns:
            DataFrame with team strength estimates
        """
        # Calculate team stats by averaging opponent performance against them
        team_defense = df.groupby('opponent').agg({
            'points': 'mean',
            'rebounds': 'mean',
            'assists': 'mean'
        })
        
        # Convert to defensive ratings (lower is better)
        # Teams that allow fewer points have better defense
        league_avg_pts = df['points'].mean()
        team_defense['opp_def_rating'] = 100 + (team_defense['points'] - league_avg_pts) * 2
        
        return team_defense[['opp_def_rating']]

    def _infer_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Infer player positions from their statistical profiles.
        
        Args:
            df: Game log DataFrame
            
        Returns:
            Series with inferred positions
        """
        # Simple position inference based on season averages
        player_profiles = df.groupby('player').agg({
            'points': 'mean',
            'rebounds': 'mean',
            'assists': 'mean',
            'blocks': 'mean'
        })
        
        positions = []
        for _, profile in player_profiles.iterrows():
            pts, reb, ast, blk = profile['points'], profile['rebounds'], profile['assists'], profile['blocks']
            
            if ast > 5:  # High assist rate
                pos = 'PG'
            elif reb > 8 and blk > 1:  # High rebounds and blocks
                pos = 'C'
            elif reb > 7:  # Good rebounds but fewer blocks
                pos = 'PF'
            elif pts > 15 and reb < 6:  # High scoring, low rebounds
                pos = 'SG'
            else:
                pos = 'SF'  # Default/versatile
            
            positions.append(pos)
        
        # Map back to original DataFrame
        position_map = dict(zip(player_profiles.index, positions))
        return df['player'].map(position_map).fillna('SF')

    def _calculate_team_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-level context for each game.
        
        Args:
            df: Game log DataFrame
            
        Returns:
            DataFrame with team context features
        """
        # Calculate team stats for each game
        team_games = df.groupby(['team', 'date']).agg({
            'usage_rate': ['mean', 'std'],
            'season_avg_points': 'sum',
            'season_avg_assists': 'sum',
            'season_avg_rebounds': 'sum',
            'minutes': 'sum'
        })
        
        # Flatten column names
        team_games.columns = [
            'team_usage_mean', 'team_usage_std',
            'team_total_scoring', 'team_total_assists', 'team_total_rebounds',
            'team_total_minutes'
        ]
        
        # Fill NaN values
        team_games['team_usage_std'] = team_games['team_usage_std'].fillna(0.05)
        
        # Calculate averages
        team_games['team_avg_pts'] = team_games['team_total_scoring'] / 5  # Assume 5 main players
        team_games['team_avg_reb'] = team_games['team_total_rebounds'] / 5
        team_games['team_avg_ast'] = team_games['team_total_assists'] / 5
        
        return team_games.reset_index()

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in the correct order.
        
        Args:
            df: Raw game log DataFrame
            
        Returns:
            DataFrame with all features
            
        Raises:
            WNBADataError: If feature creation fails
        """
        try:
            self.logger.info("Starting feature engineering pipeline...")
            
            # Validate input
            df = self.validate_input_data(df)
            
            # Create features in order (dependencies matter)
            df = self.create_basic_features(df)
            self.logger.info("✅ Created basic features")
            
            df = self.create_form_features(df)
            self.logger.info("✅ Created form features")
            
            df = self.create_context_features(df)
            self.logger.info("✅ Created context features")
            
            df = self.create_synergy_features(df)
            self.logger.info("✅ Created synergy features")
            
            # Store feature columns (exclude targets and identifiers)
            exclude_columns = [
                'player', 'team', 'date', 'opponent', 'home_away', 'result',
                'points', 'rebounds', 'assists',  # Target variables
                'prev_game_date',  # Temporary columns
                'position',  # Categorical - excluded from numeric features
                'team_y'  # Duplicate string column from merge operations
            ]
            
            self.feature_columns = [
                col for col in df.columns 
                if col not in exclude_columns and not col.startswith('season_avg_')
            ]
            
            self.logger.info(f"✅ Feature engineering complete: {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            raise WNBADataError(f"Feature engineering failed: {e}")

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by category for analysis.
        
        Returns:
            Dictionary mapping category names to feature lists
        """
        feature_groups = {
            'basic_stats': [],
            'efficiency': [],
            'form_momentum': [],
            'context': [],
            'synergy': [],
            'positional': []
        }
        
        for feature in self.feature_columns:
            if any(x in feature for x in ['usage_rate', 'scoring_efficiency', 'assist_rate', 'rebound_rate', 'ts_pct']):
                feature_groups['efficiency'].append(feature)
            elif any(x in feature for x in ['momentum', '_l5', 'season_progress']):
                feature_groups['form_momentum'].append(feature)
            elif any(x in feature for x in ['rest', 'home', 'pace', 'opp_', 'matchup']):
                feature_groups['context'].append(feature)
            elif any(x in feature for x in ['team_', 'primary_', 'player_share']):
                feature_groups['synergy'].append(feature)
            elif any(x in feature for x in ['pos_', '_expectation']):
                feature_groups['positional'].append(feature)
            else:
                feature_groups['basic_stats'].append(feature)
        
        return feature_groups


def main():
    """
    Example usage of the feature engineer.
    """
    print("🏀 WNBA Feature Engineer - Testing")
    print("=" * 40)
    
    # This would normally load real data
    print("❌ No sample data provided - requires real game logs")
    print("💡 Use with actual data:")
    print("   engineer = WNBAFeatureEngineer()")
    print("   features_df = engineer.create_all_features(game_logs_df)")


if __name__ == "__main__":
    main()