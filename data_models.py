#!/usr/bin/env python3
"""
WNBA Prediction System - Enhanced Data Models and Configuration
Defines comprehensive data structures, types, and configuration with validation.

This module provides:
- Type-safe data classes for all WNBA entities
- Configuration management with validation
- Custom exceptions for error handling
- Utility functions for data validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any, ClassVar
from datetime import datetime, date
from enum import Enum
import pandas as pd
from pathlib import Path
import logging
from team_mapping import TeamNameMapper


class HomeAway(Enum):
    """
    Enum for home/away designation.
    
    Attributes:
        HOME: Indicates home game (venue advantage)
        AWAY: Indicates away game (travel considerations)
    """
    HOME = "H"
    AWAY = "A"
    
    @classmethod
    def from_string(cls, value: str) -> 'HomeAway':
        """
        Create HomeAway from string representation.
        
        Args:
            value (str): String value ('H', 'HOME', 'A', 'AWAY')
            
        Returns:
            HomeAway: Corresponding enum value
            
        Raises:
            ValueError: If value is not recognized
        """
        value_upper = value.upper()
        if value_upper in ['H', 'HOME']:
            return cls.HOME
        elif value_upper in ['A', 'AWAY']:
            return cls.AWAY
        else:
            raise ValueError(f"Invalid HomeAway value: {value}")


class GameResult(Enum):
    """
    Enum for game results.
    
    Attributes:
        WIN: Team/player won the game
        LOSS: Team/player lost the game  
        PENDING: Game has not been completed
    """
    WIN = "W"
    LOSS = "L"
    PENDING = "PENDING"
    
    @classmethod
    def from_string(cls, value: str) -> 'GameResult':
        """
        Create GameResult from string representation.
        
        Args:
            value (str): String value ('W', 'WIN', 'L', 'LOSS', 'PENDING')
            
        Returns:
            GameResult: Corresponding enum value
            
        Raises:
            ValueError: If value is not recognized
        """
        value_upper = value.upper()
        if value_upper in ['W', 'WIN']:
            return cls.WIN
        elif value_upper in ['L', 'LOSS']:
            return cls.LOSS
        elif value_upper in ['PENDING', 'TBD', 'SCHEDULED']:
            return cls.PENDING
        else:
            raise ValueError(f"Invalid GameResult value: {value}")


class ModelType(Enum):
    """
    Enum for supported machine learning model types.
    
    Each model type represents a different approach to prediction:
    - XGBOOST: Gradient boosting with trees (primary model)
    - LIGHTGBM: Fast gradient boosting variant
    - RANDOM_FOREST: Ensemble of decision trees
    - NEURAL_NETWORK: Deep learning with uncertainty quantification
    - BAYESIAN_RIDGE: Probabilistic linear regression
    """
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm" 
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN_RIDGE = "bayesian_ridge"


@dataclass
class PlayerGameLog:
    """
    Represents a single player's performance in a game.
    
    This is the core data structure for individual player statistics
    in a specific game. Used for both historical data and predictions.
    
    Attributes:
        player (str): Player full name
        team (str): Team abbreviation (e.g., 'LAS', 'NY', 'CHI')
        game_num (int): Sequential game number in season (1-based)
        date (date): Date the game was played
        opponent (str): Opponent team abbreviation
        home_away (HomeAway): Whether playing at home or away
        result (GameResult): Game outcome (W/L/PENDING)
        minutes (float): Minutes played in the game
        points (float): Points scored
        fg_made (float): Field goals made
        fg_attempted (float): Field goals attempted
        fg_pct (float): Field goal percentage
        fg3_made (float): 3-pointers made
        fg3_attempted (float): 3-pointers attempted
        fg3_pct (float): 3-point percentage
        ft_made (float): Free throws made  
        ft_attempted (float): Free throws attempted
        ft_pct (float): Free throw percentage
        off_rebounds (float): Offensive rebounds
        def_rebounds (float): Defensive rebounds
        total_rebounds (float): Total rebounds
        assists (float): Assists recorded
        steals (float): Steals recorded
        blocks (float): Blocks recorded
        turnovers (float): Turnovers committed
        fouls (float): Personal fouls committed
        plus_minus (float): Plus/Minus
        rest_days (int): Days of rest before this game
    """
    player: str
    team: str
    game_num: int
    date: date
    opponent: str
    home_away: HomeAway
    result: GameResult
    minutes: float
    points: float
    fg_made: float = 0.0
    fg_attempted: float = 0.0
    fg_pct: float = 0.0
    fg3_made: float = 0.0
    fg3_attempted: float = 0.0
    fg3_pct: float = 0.0
    ft_made: float = 0.0
    ft_attempted: float = 0.0
    ft_pct: float = 0.0
    off_rebounds: float = 0.0
    def_rebounds: float = 0.0
    total_rebounds: float = 0.0
    assists: float = 0.0
    steals: float = 0.0
    blocks: float = 0.0
    turnovers: float = 0.0
    fouls: float = 0.0
    plus_minus: float = 0.0
    rest_days: int = 1
    
    def __post_init__(self) -> None:
        """Validate data after initialization."""
        self._validate_data()
        # Calculate derived fields if not provided
        if self.total_rebounds == 0.0:
            self.total_rebounds = self.off_rebounds + self.def_rebounds
        if self.fg_pct == 0.0 and self.fg_attempted > 0:
            self.fg_pct = self.fg_made / self.fg_attempted
        if self.fg3_pct == 0.0 and self.fg3_attempted > 0:
            self.fg3_pct = self.fg3_made / self.fg3_attempted
        if self.ft_pct == 0.0 and self.ft_attempted > 0:
            self.ft_pct = self.ft_made / self.ft_attempted
    
    def _validate_data(self) -> None:
        """
        Validate player game log data for consistency.
        
        Raises:
            ValueError: If data is invalid or inconsistent
        """
        if self.minutes < 0:
            raise ValueError(f"Minutes cannot be negative: {self.minutes}")
        
        if self.minutes > 60:  # WNBA games are 40 minutes + overtime
            logging.warning(f"Unusually high minutes for {self.player}: {self.minutes}")
        
        if any(stat < 0 for stat in [self.points, self.total_rebounds]):
            raise ValueError("Basic stats cannot be negative")
        
        if self.fg_attempted > 0 and self.fg_made > self.fg_attempted:
            raise ValueError("Field goals made cannot exceed attempted")
        
        if self.fg3_attempted > 0 and self.fg3_made > self.fg3_attempted:
            raise ValueError("3-pointers made cannot exceed attempted")
        
        if self.ft_attempted > 0 and self.ft_made > self.ft_attempted:
            raise ValueError("Free throws made cannot exceed attempted")
        
        if self.total_rebounds < (self.off_rebounds + self.def_rebounds):
            raise ValueError("Total rebounds cannot be less than sum of offensive and defensive rebounds")
    
    @property
    def field_goal_percentage(self) -> float:
        return self.fg_pct if self.fg_pct > 0 else (self.fg_made / self.fg_attempted if self.fg_attempted > 0 else 0.0)
    
    @property
    def three_point_percentage(self) -> float:
        return self.fg3_pct if self.fg3_pct > 0 else (self.fg3_made / self.fg3_attempted if self.fg3_attempted > 0 else 0.0)
    
    @property
    def free_throw_percentage(self) -> float:
        return self.ft_pct if self.ft_pct > 0 else (self.ft_made / self.ft_attempted if self.ft_attempted > 0 else 0.0)
    
    @property
    def total_rebounds_calc(self) -> float:
        return self.off_rebounds + self.def_rebounds
    
    @property
    def plus_minus_stat(self) -> float:
        return self.plus_minus
    
    @property
    def usage_rate(self) -> float:
        """Estimate usage rate (simplified calculation)."""
        if self.minutes == 0:
            return 0.0
        return (self.fg_attempted + 0.44 * self.ft_attempted + self.turnovers) / self.minutes


@dataclass
class TeamGameLog:
    """
    Represents a team's performance in a game.
    
    Used for team-level analysis and context for player predictions.
    Includes advanced metrics like pace and efficiency ratings.
    
    Attributes:
        team (str): Team abbreviation
        game_num (int): Game number in season (1-based)
        date (date): Game date
        opponent (str): Opponent team abbreviation
        home_away (HomeAway): Home or away designation
        result (GameResult): Game result
        pace (float): Possessions per game (team + opponent)
        offensive_rating (float): Points scored per 100 possessions
        defensive_rating (float): Opponent points per 100 possessions
        team_points (float): Points scored by the team
        opponent_points (float): Points allowed to opponent
    """
    team: str
    game_num: int
    date: date
    opponent: str
    home_away: HomeAway
    result: GameResult
    pace: float
    offensive_rating: float
    defensive_rating: float
    team_points: float
    opponent_points: float
    
    def __post_init__(self) -> None:
        """Validate team game log data."""
        if self.pace < 50 or self.pace > 120:
            logging.warning(f"Unusual pace for {self.team}: {self.pace}")
        
        if self.team_points < 0 or self.opponent_points < 0:
            raise ValueError("Points cannot be negative")
    
    @property
    def point_differential(self) -> float:
        """Calculate point differential (positive = win margin)."""
        return self.team_points - self.opponent_points
    
    @property
    def win_probability(self) -> float:
        """Estimate win probability based on efficiency ratings."""
        rating_diff = self.offensive_rating - self.defensive_rating
        # Simplified logistic function
        import math
        return 1 / (1 + math.exp(-rating_diff / 10))


@dataclass
class GameSchedule:
    """
    Represents a scheduled game.
    
    Used for fetching upcoming games and generating predictions
    for future matchups.
    
    Attributes:
        game_id (str): Unique identifier for the game
        date (date): Scheduled game date
        home_team (str): Home team abbreviation
        away_team (str): Away team abbreviation
        game_time (str): Scheduled start time (format varies)
        status (str): Game status ('scheduled', 'in_progress', 'final', etc.)
    """
    game_id: str
    date: date
    home_team: str
    away_team: str
    game_time: str
    status: str = "scheduled"
    
    def __post_init__(self) -> None:
        """Validate game schedule data."""
        if not self.game_id:
            raise ValueError("Game ID cannot be empty")
        
        if self.home_team == self.away_team:
            raise ValueError("Home and away teams cannot be the same")
    
    @property
    def is_completed(self) -> bool:
        """Check if game has been completed."""
        return self.status.lower() in ['final', 'completed', 'finished']
    
    @property
    def is_today(self) -> bool:
        """Check if game is scheduled for today."""
        return self.date == date.today()


@dataclass
class PlayerPrediction:
    """
    Represents predictions for a player in a specific game.
    
    This is the primary output of the prediction system, containing
    both point estimates and uncertainty quantification for all
    target statistics.
    
    Attributes:
        game_id (str): Unique game identifier
        player (str): Player full name
        team (str): Player's team abbreviation
        opponent (str): Opponent team abbreviation
        home_away (HomeAway): Home or away designation
        predicted_points (float): Predicted points scored
        predicted_rebounds (float): Predicted rebounds
        predicted_assists (float): Predicted assists
        points_uncertainty (float): Standard deviation of points prediction
        rebounds_uncertainty (float): Standard deviation of rebounds prediction
        assists_uncertainty (float): Standard deviation of assists prediction
        confidence_score (float): Overall prediction confidence (0-1)
        model_version (str): Version identifier of the model used
        prediction_timestamp (datetime): When prediction was generated
    """
    game_id: str
    player: str
    team: str
    opponent: str
    home_away: HomeAway
    predicted_points: float
    predicted_rebounds: float
    predicted_assists: float
    points_uncertainty: float
    rebounds_uncertainty: float
    assists_uncertainty: float
    confidence_score: float
    model_version: str = "1.0"
    prediction_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate prediction data."""
        if not 0 <= self.confidence_score <= 1:
            raise ValueError(f"Confidence score must be 0-1: {self.confidence_score}")
        
        if any(pred < 0 for pred in [self.predicted_points, self.predicted_rebounds, self.predicted_assists]):
            raise ValueError("Predictions cannot be negative")
        
        if any(unc < 0 for unc in [self.points_uncertainty, self.rebounds_uncertainty, self.assists_uncertainty]):
            raise ValueError("Uncertainties cannot be negative")

    @property
    def points_ci_lower(self) -> float:
        """Calculate 95% confidence interval lower bound for points."""
        return max(0.0, self.predicted_points - 1.96 * self.points_uncertainty)
    
    @property
    def points_ci_upper(self) -> float:
        """Calculate 95% confidence interval upper bound for points."""
        return self.predicted_points + 1.96 * self.points_uncertainty
    
    @property
    def rebounds_ci_lower(self) -> float:
        """Calculate 95% confidence interval lower bound for rebounds."""
        return max(0.0, self.predicted_rebounds - 1.96 * self.rebounds_uncertainty)
    
    @property
    def rebounds_ci_upper(self) -> float:
        """Calculate 95% confidence interval upper bound for rebounds."""
        return self.predicted_rebounds + 1.96 * self.rebounds_uncertainty
    
    @property
    def assists_ci_lower(self) -> float:
        """Calculate 95% confidence interval lower bound for assists."""
        return max(0.0, self.predicted_assists - 1.96 * self.assists_uncertainty)
    
    @property
    def assists_ci_upper(self) -> float:
        """Calculate 95% confidence interval upper bound for assists."""
        return self.predicted_assists + 1.96 * self.assists_uncertainty
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prediction to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation including confidence intervals
        """
        return {
            'game_id': self.game_id,
            'player': self.player,
            'team': self.team,
            'opponent': self.opponent,
            'home_away': self.home_away.value,
            'predicted_points': self.predicted_points,
            'predicted_rebounds': self.predicted_rebounds,
            'predicted_assists': self.predicted_assists,
            'points_uncertainty': self.points_uncertainty,
            'rebounds_uncertainty': self.rebounds_uncertainty,
            'assists_uncertainty': self.assists_uncertainty,
            'points_ci_lower': self.points_ci_lower,
            'points_ci_upper': self.points_ci_upper,
            'rebounds_ci_lower': self.rebounds_ci_lower,
            'rebounds_ci_upper': self.rebounds_ci_upper,
            'assists_ci_lower': self.assists_ci_lower,
            'assists_ci_upper': self.assists_ci_upper,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version,
            'prediction_timestamp': self.prediction_timestamp
        }


@dataclass
class ModelMetrics:
    """
    Represents performance metrics for a machine learning model.
    
    Used for model evaluation, comparison, and monitoring during
    both validation and production deployment.
    
    Attributes:
        model_name (str): Name/type of the model (e.g., 'xgboost')
        stat_type (str): Statistic being predicted ('points', 'rebounds', 'assists')
        mae (float): Mean Absolute Error
        mse (float): Mean Squared Error  
        r2_score (float): R-squared coefficient of determination
        brier_score (float): Brier score for probabilistic calibration
        coverage_95 (float): 95% confidence interval coverage rate
        n_predictions (int): Number of predictions in evaluation
        training_time (float): Time taken to train model (seconds)
        prediction_time (float): Average time per prediction (milliseconds)
    """
    model_name: str
    stat_type: str
    mae: float
    mse: float
    r2_score: float
    brier_score: float
    coverage_95: float
    n_predictions: int
    training_time: float = 0.0
    prediction_time: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate model metrics."""
        if self.n_predictions <= 0:
            raise ValueError("Number of predictions must be positive")
        
        if not 0 <= self.coverage_95 <= 1:
            raise ValueError(f"Coverage must be 0-1: {self.coverage_95}")
    
    @property
    def rmse(self) -> float:
        """Calculate Root Mean Squared Error."""
        return self.mse ** 0.5
    
    @property
    def is_well_calibrated(self) -> bool:
        """Check if model is well calibrated (95% CI coverage within 90-98%)."""
        return 0.90 <= self.coverage_95 <= 0.98
    
    @property
    def meets_research_benchmark(self) -> bool:
        """Check if Brier score meets research benchmark (<0.12)."""
        return self.brier_score < 0.12

@dataclass
class OptimizedPredictionConfig:
    """
    Optimized prediction configuration based on training results analysis.
    
    This configuration focuses on the most predictable statistics and
    adjusts model weights based on observed performance.
    """
    
    # OPTIMIZED: Focus on highly predictable core stats only
    target_stats: List[str] = field(default_factory=lambda: [
        "points",           # R²=0.798 - Core fantasy stat
        "assists",          # R²=0.861 - Highly predictable  
        "total_rebounds",   # R²=0.792 - Very good performance
        "minutes",          # R²=0.679 - Decent for context
        "fg_made",          # R²=0.747 - Good shooting stat
        "fg_attempted",     # R²=0.779 - Good volume stat
        "fg_pct"            # R²=0.874 - Excellent percentage stat
        # REMOVED: blocks, steals, turnovers, fouls, plus_minus (poor R²)
        # REMOVED: 3-point stats (moderate R² but less reliable)
        # REMOVED: free throw stats (can add back if needed)
    ])
    
    # OPTIMIZED: Adjust weights based on observed performance
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "lightgbm": 0.35,      # Best overall performer
        "xgboost": 0.35,       # Strong on shooting stats
        "random_forest": 0.20, # Solid backup performer
        "neural_network": 0.10 # Reduced until technical issues fixed
    })
    
    # Standard parameters (keep existing)
    min_games_for_prediction: int = 5
    confidence_threshold: float = 0.7  # Slightly higher for quality
    max_uncertainty: float = 8.0      # Slightly lower for reliability
    feature_importance_threshold: float = 0.01
    validation_split: float = 0.2
    random_state: int = 42
    n_cross_validation_folds: int = 5
    max_training_time_minutes: int = 60
    
    # OPTIMIZED: Enhanced model parameters based on good performance
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,    # Increased from 100
        "max_depth": 6,         # Keep current
        "learning_rate": 0.08,  # Slightly lower for stability
        "random_state": 42,
        "verbosity": 0,
        "reg_alpha": 0.1,       # Add regularization
        "reg_lambda": 0.1
    })
    
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,    # Increased from 100
        "max_depth": 6,         # Keep current
        "learning_rate": 0.08,  # Slightly lower for stability
        "random_state": 42,
        "verbosity": -1,
        "reg_alpha": 0.1,       # Add regularization
        "reg_lambda": 0.1
    })
    
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 120,    # Increased from 100
        "max_depth": 8,         # Keep current
        "random_state": 42,
        "n_jobs": -1,
        "min_samples_split": 5, # Add to reduce overfitting
        "min_samples_leaf": 2
    })
    
    # Performance thresholds based on observed results
    min_acceptable_r2: Dict[str, float] = field(default_factory=lambda: {
        "points": 0.70,         # Aim for 70%+ on core stats
        "assists": 0.80,        # Aim for 80%+ on assists
        "total_rebounds": 0.70, # Aim for 70%+ on rebounds
        "minutes": 0.60,        # Accept 60%+ on minutes
        "fg_made": 0.70,        # Aim for 70%+ on shooting
        "fg_attempted": 0.70,   # Aim for 70%+ on attempts
        "fg_pct": 0.80          # Aim for 80%+ on percentages
    })
    
    # Expected MAE ranges for validation
    expected_mae_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "points": (1.5, 3.0),           # ±1.5-3.0 points
        "assists": (0.3, 0.6),          # ±0.3-0.6 assists  
        "total_rebounds": (0.5, 1.0),   # ±0.5-1.0 rebounds
        "minutes": (3.0, 6.0),          # ±3-6 minutes
        "fg_made": (0.6, 1.2),          # ±0.6-1.2 field goals
        "fg_attempted": (1.2, 2.5),     # ±1.2-2.5 attempts
        "fg_pct": (0.04, 0.08)          # ±4-8 percentage points
    })
    
@dataclass
class PredictionConfig:
    """
    Configuration for prediction models with comprehensive validation.
    
    This class manages all configurable aspects of the prediction system,
    from target statistics to model hyperparameters and validation criteria.
    
    Attributes:
        target_stats (List[str]): Statistics to predict
        min_games_for_prediction (int): Minimum games required for valid prediction
        confidence_threshold (float): Minimum confidence for displaying predictions
        max_uncertainty (float): Maximum allowed prediction uncertainty
        model_weights (Dict[str, float]): Ensemble weights for different models
        feature_importance_threshold (float): Minimum feature importance to include
        validation_split (float): Fraction of data for validation
        random_state (int): Random seed for reproducibility
        n_cross_validation_folds (int): Number of CV folds for model selection
        max_training_time_minutes (int): Maximum time allowed for training
    """
    target_stats: List[str] = field(default_factory=lambda: [
        "points", "assists", "minutes", "fg_made", "fg_attempted", "fg_pct", "fg3_made", "fg3_attempted", "fg3_pct", "ft_made", "ft_attempted", "ft_pct", "off_rebounds", "def_rebounds", "total_rebounds", "steals", "blocks", "turnovers", "fouls", "plus_minus"
    ])
    min_games_for_prediction: int = 5
    confidence_threshold: float = 0.6
    max_uncertainty: float = 10.0
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.30,
        "lightgbm": 0.25,
        "random_forest": 0.25,
        "neural_network": 0.20
    })
    feature_importance_threshold: float = 0.01
    validation_split: float = 0.2
    random_state: int = 42
    n_cross_validation_folds: int = 5
    max_training_time_minutes: int = 60
    
    # Class-level constants
    SUPPORTED_STATS: ClassVar[List[str]] = [
        "points", "assists", "minutes", "fg_made", "fg_attempted", "fg_pct", "fg3_made", "fg3_attempted", "fg3_pct", "ft_made", "ft_attempted", "ft_pct", "off_rebounds", "def_rebounds", "total_rebounds", "steals", "blocks", "turnovers", "fouls", "plus_minus"
    ]
    DEFAULT_WEIGHTS: ClassVar[Dict[str, float]] = {
        "xgboost": 0.30,
        "lightgbm": 0.25, 
        "random_forest": 0.25,
        "neural_network": 0.20
    }
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._normalize_weights()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate target stats
        for stat in self.target_stats:
            if stat not in self.SUPPORTED_STATS:
                raise ValueError(f"Unsupported target stat: {stat}")
        
        # Validate thresholds
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"Confidence threshold must be 0-1: {self.confidence_threshold}")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Validation split must be 0-1: {self.validation_split}")
        
        if self.min_games_for_prediction < 1:
            raise ValueError("Minimum games for prediction must be >= 1")
        
        if self.max_uncertainty <= 0:
            raise ValueError("Max uncertainty must be positive")
        
        # Validate model weights
        if not self.model_weights:
            raise ValueError("Model weights cannot be empty")
        
        for model, weight in self.model_weights.items():
            if weight < 0:
                raise ValueError(f"Model weight cannot be negative: {model}={weight}")
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1.0."""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                model: weight / total_weight 
                for model, weight in self.model_weights.items()
            }
    
    def add_target_stat(self, stat: str) -> None:
        """
        Add a new target statistic.
        
        Args:
            stat (str): Statistic name to add
            
        Raises:
            ValueError: If statistic is not supported
        """
        if stat not in self.SUPPORTED_STATS:
            raise ValueError(f"Unsupported stat: {stat}")
        
        if stat not in self.target_stats:
            self.target_stats.append(stat)
    
    def update_model_weight(self, model: str, weight: float) -> None:
        """
        Update weight for a specific model.
        
        Args:
            model (str): Model name
            weight (float): New weight (will be normalized)
            
        Raises:
            ValueError: If weight is negative
        """
        if weight < 0:
            raise ValueError(f"Weight cannot be negative: {weight}")
        
        self.model_weights[model] = weight
        self._normalize_weights()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'target_stats': self.target_stats,
            'min_games_for_prediction': self.min_games_for_prediction,
            'confidence_threshold': self.confidence_threshold,
            'max_uncertainty': self.max_uncertainty,
            'model_weights': self.model_weights,
            'feature_importance_threshold': self.feature_importance_threshold,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'n_cross_validation_folds': self.n_cross_validation_folds,
            'max_training_time_minutes': self.max_training_time_minutes
        }


# Custom Exception Classes
class WNBADataError(Exception):
    """
    Custom exception for WNBA data-related errors.
    
    Raised when there are issues with data fetching, parsing,
    or validation that prevent normal operation.
    """
    pass


class WNBAModelError(Exception):
    """
    Custom exception for WNBA model-related errors.
    
    Raised when there are issues with model training, loading,
    or prediction that prevent normal operation.
    """
    pass


class WNBAPredictionError(Exception):
    """
    Custom exception for WNBA prediction-related errors.
    
    Raised when there are issues generating predictions,
    such as insufficient data or model failures.
    """
    pass


class WNBAConfigurationError(Exception):
    """
    Custom exception for configuration-related errors.
    
    Raised when configuration parameters are invalid or
    incompatible with system requirements.
    """
    pass


# Utility Functions
def validate_dataframe_columns(
    df: pd.DataFrame, 
    required_columns: List[str],
    raise_on_missing: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        raise_on_missing (bool, optional): Whether to raise exception on missing columns. Defaults to True.
        
    Returns:
        Tuple[bool, List[str]]: (all_present, missing_columns)
        
    Raises:
        WNBADataError: If required columns are missing and raise_on_missing=True
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    all_present = len(missing_columns) == 0
    
    if missing_columns and raise_on_missing:
        raise WNBADataError(f"Missing required columns: {missing_columns}")
    
    return all_present, missing_columns


def create_game_id(
    date: date, 
    home_team: str, 
    away_team: str
) -> str:
    """
    Create standardized game identifier.
    
    Args:
        date (date): Game date
        home_team (str): Home team abbreviation
        away_team (str): Away team abbreviation
        
    Returns:
        str: Standardized game ID in format 'YYYYMMDD_AWAY_HOME'
    """
    date_str = date.strftime('%Y%m%d')
    return f"{date_str}_{away_team}_{home_team}"


if __name__ == "__main__":
    """
    Demo usage of the data models.
    """
    print("🏀 WNBA Data Models - Demo")
    print("=" * 30)
    
    # Create sample configuration
    config = PredictionConfig()
    print(f"✅ Configuration created with {len(config.target_stats)} target stats")
    
    # Create sample game schedule
    schedule = GameSchedule(
        game_id="20250617_NY_LAS",
        date=date.today(),
        home_team="LAS",
        away_team="NY", 
        game_time="7:00 PM PT"
    )
    print(f"✅ Game schedule: {schedule.away_team} @ {schedule.home_team}")
    
    # Create sample prediction
    prediction = PlayerPrediction(
        game_id=schedule.game_id,
        player="A'ja Wilson",
        team="LAS",
        opponent="NY",
        home_away=HomeAway.HOME,
        predicted_points=22.5,
        predicted_rebounds=8.2,
        predicted_assists=2.1,
        points_uncertainty=4.1,
        rebounds_uncertainty=2.3,
        assists_uncertainty=1.2,
        confidence_score=0.85
    )
    
    print(f"✅ Prediction for {prediction.player}:")
    print(f"   Points: {prediction.predicted_points:.1f} ± {prediction.points_uncertainty:.1f}")
    print(f"   Confidence: {prediction.confidence_score:.1%}")
    
    print("\n🎉 Data models working correctly!")