#!/usr/bin/env python3
"""
WNBA Model Validation and Backtesting
Validates model performance using time series cross-validation and calibration analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from dataclasses import asdict
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ML metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve

from wnba_data_models import (
    ModelMetrics, PredictionConfig, WNBAModelError, WNBADataError
)
from wnba_prediction_models import WNBAPredictionModel
from wnba_feature_engineer import WNBAFeatureEngineer


class WNBAModelValidator:
    """
    Validates WNBA prediction models using time series cross-validation.
    
    Implements walk-forward validation to simulate real-world usage where
    models predict future games based on historical data.
    
    Attributes:
        config: Configuration for validation
        prediction_model: Model to validate
        feature_engineer: Feature engineering component
        validation_results: Results from validation runs
        logger: Logger instance
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        output_dir: str = "wnba_validation"
    ):
        """
        Initialize model validator.
        
        Args:
            config: Configuration object, uses defaults if None
            output_dir: Directory for validation outputs
        """
        self.config = config or PredictionConfig()
        self.output_dir = output_dir
        self.prediction_model = WNBAPredictionModel(config=self.config)
        self.feature_engineer = WNBAFeatureEngineer(config=self.config)
        
        self.validation_results: Dict[str, List[Dict]] = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def time_series_validation(
        self,
        game_logs_df: pd.DataFrame,
        test_weeks: int = 4,
        min_train_weeks: int = 8,
        step_weeks: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Perform walk-forward time series validation.
        
        Args:
            game_logs_df: Game logs DataFrame
            test_weeks: Number of weeks for each test period
            min_train_weeks: Minimum training weeks required
            step_weeks: Step size between validation periods
            
        Returns:
            Dictionary of validation results by statistic
            
        Raises:
            WNBAModelError: If validation fails
        """
        self.logger.info("Starting time series validation...")
        self.logger.info(f"Test period: {test_weeks} weeks, Min training: {min_train_weeks} weeks")
        
        try:
            # Prepare data
            game_logs_df = game_logs_df.copy()
            game_logs_df['date'] = pd.to_datetime(game_logs_df['date'])
            game_logs_df = game_logs_df.sort_values(['date', 'player']).reset_index(drop=True)
            
            # Get date range
            start_date = game_logs_df['date'].min()
            end_date = game_logs_df['date'].max()
            total_weeks = (end_date - start_date).days // 7
            
            self.logger.info(f"Data range: {start_date.date()} to {end_date.date()} ({total_weeks} weeks)")
            
            if total_weeks < min_train_weeks + test_weeks:
                raise WNBAModelError(f"Insufficient data: need {min_train_weeks + test_weeks} weeks, have {total_weeks}")
            
            # Create validation periods
            validation_periods = []
            for week in range(min_train_weeks, total_weeks - test_weeks + 1, step_weeks):
                train_end_date = start_date + timedelta(weeks=week)
                test_start_date = train_end_date
                test_end_date = test_start_date + timedelta(weeks=test_weeks)
                
                train_data = game_logs_df[game_logs_df['date'] < train_end_date]
                test_data = game_logs_df[
                    (game_logs_df['date'] >= test_start_date) & 
                    (game_logs_df['date'] < test_end_date)
                ]
                
                if len(train_data) < 100 or len(test_data) < 20:
                    continue
                
                validation_periods.append({
                    'period': len(validation_periods) + 1,
                    'train_end_date': train_end_date,
                    'test_start_date': test_start_date,
                    'test_end_date': test_end_date,
                    'train_data': train_data,
                    'test_data': test_data
                })
            
            self.logger.info(f"Created {len(validation_periods)} validation periods")
            
            if not validation_periods:
                raise WNBAModelError("No valid validation periods created")
            
            # Run validation for each period
            all_results = {stat: [] for stat in self.config.target_stats}
            
            for period in validation_periods:
                self.logger.info(f"Validating period {period['period']}: "
                               f"{period['test_start_date'].date()} to {period['test_end_date'].date()}")
                
                period_results = self._validate_single_period(period)
                
                for stat in self.config.target_stats:
                    if stat in period_results:
                        period_results[stat]['period'] = period['period']
                        period_results[stat]['test_start_date'] = period['test_start_date']
                        period_results[stat]['test_end_date'] = period['test_end_date']
                        all_results[stat].append(period_results[stat])
            
            self.validation_results = all_results
            self.logger.info(f"Validation complete: {len(validation_periods)} periods analyzed")
            
            return all_results
            
        except Exception as e:
            raise WNBAModelError(f"Time series validation failed: {e}")

    def _validate_single_period(self, period: Dict) -> Dict[str, Dict]:
        """
        Validate models for a single time period.
        
        Args:
            period: Dictionary with train/test data and metadata
            
        Returns:
            Dictionary of metrics by statistic
        """
        train_data = period['train_data']
        test_data = period['test_data']
        
        try:
            # Create temporary model for this period
            temp_model = WNBAPredictionModel(config=self.config)
            
            # Train on this period's data
            temp_model.train_all_models(train_data)
            
            # Make predictions on test data
            predictions_df = self._make_period_predictions(temp_model, test_data)
            
            # Calculate metrics for each statistic
            period_metrics = {}
            for stat in self.config.target_stats:
                metrics = self._calculate_prediction_metrics(predictions_df, stat)
                period_metrics[stat] = metrics
            
            return period_metrics
            
        except Exception as e:
            self.logger.warning(f"Period validation failed: {e}")
            return {}

    def _make_period_predictions(
        self, 
        model: WNBAPredictionModel, 
        test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions for test period data.
        
        Args:
            model: Trained model
            test_data: Test data
            
        Returns:
            DataFrame with predictions and actuals
        """
        predictions = []
        
        for _, row in test_data.iterrows():
            try:
                # Get feature vector for this player-game
                feature_vector = row[model.feature_columns].fillna(0.0).values
                
                # Make prediction
                prediction = model.predict_player_stats(feature_vector)
                
                pred_dict = {
                    'player': row['player'],
                    'date': row['date'],
                    'actual_points': row['points'],
                    'actual_rebounds': row['rebounds'],
                    'actual_assists': row['assists'],
                    'predicted_points': prediction.predicted_points,
                    'predicted_rebounds': prediction.predicted_rebounds,
                    'predicted_assists': prediction.predicted_assists,
                    'points_uncertainty': prediction.points_uncertainty,
                    'rebounds_uncertainty': prediction.rebounds_uncertainty,
                    'assists_uncertainty': prediction.assists_uncertainty
                }
                
                predictions.append(pred_dict)
                
            except Exception as e:
                self.logger.debug(f"Prediction failed for {row['player']}: {e}")
                continue
        
        return pd.DataFrame(predictions)

    def _calculate_prediction_metrics(
        self, 
        predictions_df: pd.DataFrame, 
        stat: str
    ) -> Dict[str, float]:
        """
        Calculate validation metrics for a specific statistic.
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            stat: Statistic name
            
        Returns:
            Dictionary of metrics
        """
        actual_col = f'actual_{stat}'
        pred_col = f'predicted_{stat}'
        unc_col = f'{stat}_uncertainty'
        
        if actual_col not in predictions_df.columns or pred_col not in predictions_df.columns:
            return {}
        
        actual = predictions_df[actual_col].dropna()
        predicted = predictions_df[pred_col].dropna()
        
        if len(actual) == 0 or len(predicted) == 0:
            return {}
        
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]
        uncertainty = predictions_df[unc_col].iloc[:min_len]
        
        try:
            # Basic metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # Prediction interval coverage
            ci_lower = predicted - 1.96 * uncertainty
            ci_upper = predicted + 1.96 * uncertainty
            coverage_95 = ((actual >= ci_lower) & (actual <= ci_upper)).mean()
            
            # Calibration (Brier score approximation)
            median_actual = actual.median()
            binary_actual = (actual > median_actual).astype(int)
            binary_prob = (predicted > median_actual).astype(float)
            brier_score = np.mean((binary_prob - binary_actual) ** 2)
            
            # Additional metrics
            mean_uncertainty = uncertainty.mean()
            prediction_spread = predicted.std()
            
            return {
                'mae': mae,
                'mse': mse,
                'r2_score': r2,
                'coverage_95': coverage_95,
                'brier_score': brier_score,
                'mean_uncertainty': mean_uncertainty,
                'prediction_spread': prediction_spread,
                'n_predictions': len(actual)
            }
            
        except Exception as e:
            self.logger.warning(f"Metrics calculation failed for {stat}: {e}")
            return {}

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation summary and assessment
        """
        if not self.validation_results:
            raise WNBAModelError("No validation results available. Run validation first.")
        
        self.logger.info("Generating validation report...")
        
        report = {
            'summary': {},
            'detailed_metrics': {},
            'assessment': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        for stat in self.config.target_stats:
            if stat not in self.validation_results or not self.validation_results[stat]:
                continue
            
            stat_results = pd.DataFrame(self.validation_results[stat])
            
            summary = {
                'mean_mae': stat_results['mae'].mean(),
                'std_mae': stat_results['mae'].std(),
                'mean_r2': stat_results['r2_score'].mean(),
                'std_r2': stat_results['r2_score'].std(),
                'mean_coverage': stat_results['coverage_95'].mean(),
                'mean_brier': stat_results['brier_score'].mean(),
                'total_predictions': stat_results['n_predictions'].sum(),
                'n_periods': len(stat_results)
            }
            
            report['summary'][stat] = summary
            report['detailed_metrics'][stat] = stat_results.to_dict('records')
        
        # Overall assessment
        all_brier_scores = []
        all_maes = []
        all_r2s = []
        
        for stat in self.config.target_stats:
            if stat in report['summary']:
                all_brier_scores.append(report['summary'][stat]['mean_brier'])
                all_maes.append(report['summary'][stat]['mean_mae'])
                all_r2s.append(report['summary'][stat]['mean_r2'])
        
        if all_brier_scores:
            avg_brier = np.mean(all_brier_scores)
            avg_mae = np.mean(all_maes)
            avg_r2 = np.mean(all_r2s)
            
            # Grade based on research benchmarks
            if avg_brier < 0.12:
                grade = "EXCELLENT"
                assessment = "Meets research standards for sports prediction"
            elif avg_brier < 0.20:
                grade = "GOOD"
                assessment = "Above average calibration"
            else:
                grade = "FAIR" 
                assessment = "Room for improvement"
            
            report['assessment'] = {
                'overall_grade': grade,
                'assessment': assessment,
                'avg_brier_score': avg_brier,
                'avg_mae': avg_mae,
                'avg_r2_score': avg_r2,
                'research_comparison': {
                    'darko_benchmark': 0.10,
                    'xgboost_synergy_benchmark': 0.10,
                    'neural_attention_benchmark': 0.12,
                    'our_model': avg_brier
                }
            }
            
            # Generate recommendations
            recommendations = []
            
            if avg_brier > 0.15:
                recommendations.append("Consider improving calibration with better uncertainty quantification")
            
            if avg_r2 < 0.5:
                recommendations.append("Low R² suggests need for better features or more complex models")
            
            for stat in self.config.target_stats:
                if stat in report['summary']:
                    coverage = report['summary'][stat]['mean_coverage']
                    if coverage < 0.90 or coverage > 0.98:
                        recommendations.append(f"Adjust uncertainty estimation for {stat} (coverage: {coverage:.2%})")
            
            if not recommendations:
                recommendations.append("Model performance meets benchmarks - consider production deployment")
            
            report['recommendations'] = recommendations
        
        return report

    def create_validation_visualizations(self) -> Dict[str, str]:
        """
        Create validation performance visualizations.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.validation_results:
            raise WNBAModelError("No validation results available")
        
        self.logger.info("Creating validation visualizations...")
        
        viz_files = {}
        
        # Create matplotlib visualizations
        viz_files.update(self._create_matplotlib_plots())
        
        # Create plotly visualizations if available
        if PLOTLY_AVAILABLE:
            viz_files.update(self._create_plotly_plots())
        
        return viz_files

    def _create_matplotlib_plots(self) -> Dict[str, str]:
        """Create matplotlib-based validation plots."""
        viz_files = {}
        
        # Performance over time plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Over Time', fontsize=16)
        
        for i, stat in enumerate(self.config.target_stats):
            if stat not in self.validation_results:
                continue
            
            results_df = pd.DataFrame(self.validation_results[stat])
            if results_df.empty:
                continue
            
            # MAE plot
            ax_mae = axes[0, i]
            ax_mae.plot(results_df['period'], results_df['mae'], 'o-', color='blue')
            ax_mae.set_title(f'{stat.title()} - MAE')
            ax_mae.set_xlabel('Validation Period')
            ax_mae.set_ylabel('Mean Absolute Error')
            ax_mae.grid(True, alpha=0.3)
            
            # R² plot
            ax_r2 = axes[1, i]
            ax_r2.plot(results_df['period'], results_df['r2_score'], 'o-', color='green')
            ax_r2.set_title(f'{stat.title()} - R²')
            ax_r2.set_xlabel('Validation Period')
            ax_r2.set_ylabel('R² Score')
            ax_r2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        performance_file = os.path.join(self.output_dir, 'performance_over_time.png')
        plt.savefig(performance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_files['performance_plot'] = performance_file
        
        # Calibration plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for stat in self.config.target_stats:
            if stat not in self.validation_results:
                continue
            
            results_df = pd.DataFrame(self.validation_results[stat])
            if not results_df.empty:
                ax.plot(results_df['period'], results_df['brier_score'], 
                       'o-', label=f'{stat.title()}', linewidth=2)
        
        ax.axhline(y=0.12, color='red', linestyle='--', 
                  label='Research Benchmark (0.12)', alpha=0.7)
        
        ax.set_title('Calibration Quality Over Time (Lower is Better)', fontsize=14)
        ax.set_xlabel('Validation Period')
        ax.set_ylabel('Brier Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        calibration_file = os.path.join(self.output_dir, 'calibration_analysis.png')
        plt.savefig(calibration_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_files['calibration_plot'] = calibration_file
        
        return viz_files

    def _create_plotly_plots(self) -> Dict[str, str]:
        """Create interactive plotly visualizations."""
        viz_files = {}
        
        # Interactive performance dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'{stat.title()} MAE' for stat in self.config.target_stats] +
                          [f'{stat.title()} R²' for stat in self.config.target_stats],
            vertical_spacing=0.15
        )
        
        colors = ['#FF6B35', '#4ECDC4', '#45B7D1']
        
        for i, stat in enumerate(self.config.target_stats):
            if stat not in self.validation_results:
                continue
            
            results_df = pd.DataFrame(self.validation_results[stat])
            if results_df.empty:
                continue
            
            # MAE trace
            fig.add_trace(
                go.Scatter(
                    x=results_df['period'],
                    y=results_df['mae'],
                    mode='lines+markers',
                    name=f'{stat.title()} MAE',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # R² trace
            fig.add_trace(
                go.Scatter(
                    x=results_df['period'],
                    y=results_df['r2_score'],
                    mode='lines+markers',
                    name=f'{stat.title()} R²',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=2, col=i+1
            )
        
        fig.update_layout(
            title="Interactive Model Performance Dashboard",
            height=600,
            template="plotly_white"
        )
        
        dashboard_file = os.path.join(self.output_dir, 'interactive_performance.html')
        fig.write_html(dashboard_file)
        viz_files['interactive_dashboard'] = dashboard_file
        
        return viz_files

    def export_validation_results(self) -> Dict[str, str]:
        """
        Export validation results to files.
        
        Returns:
            Dictionary mapping export types to file paths
        """
        if not self.validation_results:
            raise WNBAModelError("No validation results to export")
        
        self.logger.info("Exporting validation results...")
        
        export_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed results
        for stat in self.config.target_stats:
            if stat in self.validation_results and self.validation_results[stat]:
                results_df = pd.DataFrame(self.validation_results[stat])
                results_file = os.path.join(self.output_dir, f'{stat}_validation_results_{timestamp}.csv')
                results_df.to_csv(results_file, index=False)
                export_files[f'{stat}_results'] = results_file
        
        # Export summary report
        try:
            report = self.generate_validation_report()
            report_file = os.path.join(self.output_dir, f'validation_report_{timestamp}.json')
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            export_files['validation_report'] = report_file
            
        except Exception as e:
            self.logger.warning(f"Failed to export validation report: {e}")
        
        # Create visualizations
        try:
            viz_files = self.create_validation_visualizations()
            export_files.update(viz_files)
        except Exception as e:
            self.logger.warning(f"Failed to create visualizations: {e}")
        
        self.logger.info(f"Exported {len(export_files)} validation files")
        return export_files


def main():
    """
    Example usage of the model validator.
    """
    print("🎯 WNBA Model Validator")
    print("=" * 30)
    
    print("❌ No sample data provided - requires real game logs")
    print("💡 Example usage:")
    print("   validator = WNBAModelValidator()")
    print("   results = validator.time_series_validation(game_logs_df)")
    print("   report = validator.generate_validation_report()")
    print("   files = validator.export_validation_results()")


if __name__ == "__main__":
    main()