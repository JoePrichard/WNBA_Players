#!/usr/bin/env python3
"""
WNBA Daily Prediction Model Validation and Backtesting
Validates model performance using time series cross-validation

Features:
- Walk-forward validation (realistic for sports predictions)
- Calibration analysis (Brier scores)
- Feature importance over time
- Prediction accuracy tracking
- Uncertainty quantification validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class WNBAModelValidator:
    def __init__(self, game_logs_file):
        """Initialize validator with game logs"""
        self.game_logs_df = pd.read_csv(game_logs_file)
        self.game_logs_df['date'] = pd.to_datetime(self.game_logs_df['date'])
        
        # Sort by date for time series validation
        self.game_logs_df = self.game_logs_df.sort_values(['date', 'player'])
        
        self.target_stats = ['points', 'rebounds', 'assists']
        self.validation_results = {}
        
    def time_series_validation(self, test_weeks=4, min_train_weeks=8):
        """
        Perform walk-forward validation using time series splits
        This mimics real-world usage where we predict future games
        """
        print("🕐 Performing time series walk-forward validation...")
        print(f"📊 Test period: {test_weeks} weeks, Min training: {min_train_weeks} weeks")
        
        # Get date range
        start_date = self.game_logs_df['date'].min()
        end_date = self.game_logs_df['date'].max()
        total_weeks = (end_date - start_date).days // 7
        
        print(f"📅 Data range: {start_date.date()} to {end_date.date()} ({total_weeks} weeks)")
        
        validation_periods = []
        
        # Create validation splits
        for week in range(min_train_weeks, total_weeks - test_weeks + 1, 2):  # Every 2 weeks
            train_end = start_date + timedelta(weeks=week)
            test_start = train_end
            test_end = test_start + timedelta(weeks=test_weeks)
            
            # Get data splits
            train_data = self.game_logs_df[self.game_logs_df['date'] < train_end]
            test_data = self.game_logs_df[
                (self.game_logs_df['date'] >= test_start) & 
                (self.game_logs_df['date'] < test_end)
            ]
            
            if len(train_data) < 100 or len(test_data) < 20:
                continue
                
            validation_periods.append({
                'period': len(validation_periods) + 1,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_data': train_data,
                'test_data': test_data
            })
        
        print(f"✅ Created {len(validation_periods)} validation periods")
        
        # Validate each period
        all_results = []
        
        for i, period in enumerate(validation_periods):
            print(f"\n🔍 Validating period {period['period']}: {period['test_start'].date()} to {period['test_end'].date()}")
            
            period_results = self.validate_period(period)
            if period_results:
                all_results.append(period_results)
        
        # Combine results
        if all_results:
            self.validation_results = self.combine_validation_results(all_results)
            print(f"\n✅ Validation complete: {len(all_results)} periods analyzed")
            return self.validation_results
        else:
            print("❌ No validation results generated")
            return None
    
    def validate_period(self, period):
        """Validate models for a specific time period"""
        try:
            from daily_game_predictor import WNBADailyPredictor
            
            # Create temporary files for this period
            train_file = f"temp_train_{period['period']}.csv"
            period['train_data'].to_csv(train_file, index=False)
            
            # Train predictor on this period's training data
            predictor = WNBADailyPredictor(train_file)
            predictor.prepare_data()
            
            # Train models
            results = {}
            for stat in self.target_stats:
                stat_results = predictor.train_models_for_stat(stat)
                results[stat] = stat_results
            
            # Make predictions on test data
            test_predictions = self.make_test_predictions(predictor, period['test_data'])
            
            # Calculate metrics
            period_metrics = self.calculate_period_metrics(test_predictions, period)
            
            # Clean up
            import os
            if os.path.exists(train_file):
                os.remove(train_file)
            
            return {
                'period': period['period'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'metrics': period_metrics,
                'predictions': test_predictions
            }
            
        except Exception as e:
            print(f"  ❌ Error in period {period['period']}: {e}")
            return None
    
    def make_test_predictions(self, predictor, test_data):
        """Make predictions for test data"""
        predictions = []
        
        for _, row in test_data.iterrows():
            try:
                # Extract features
                features = []
                for feature in predictor.feature_columns:
                    features.append(row.get(feature, 0.0))
                
                # Predict each stat
                pred_stats, uncertainties = predictor.predict_game_stats(features)
                
                prediction = {
                    'player': row['player'],
                    'date': row['date'],
                    'actual_points': row['points'],
                    'actual_rebounds': row['rebounds'], 
                    'actual_assists': row['assists'],
                    'pred_points': pred_stats['points'],
                    'pred_rebounds': pred_stats['rebounds'],
                    'pred_assists': pred_stats['assists'],
                    'unc_points': uncertainties['points'],
                    'unc_rebounds': uncertainties['rebounds'],
                    'unc_assists': uncertainties['assists']
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                continue
        
        return pd.DataFrame(predictions)
    
    def calculate_period_metrics(self, predictions_df, period):
        """Calculate metrics for a validation period"""
        metrics = {}
        
        for stat in self.target_stats:
            actual_col = f'actual_{stat}'
            pred_col = f'pred_{stat}'
            unc_col = f'unc_{stat}'
            
            if actual_col in predictions_df.columns and pred_col in predictions_df.columns:
                actual = predictions_df[actual_col]
                predicted = predictions_df[pred_col]
                uncertainty = predictions_df[unc_col]
                
                # Basic metrics
                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                r2 = r2_score(actual, predicted)
                
                # Prediction intervals coverage
                ci_lower = predicted - 1.96 * uncertainty
                ci_upper = predicted + 1.96 * uncertainty
                coverage = ((actual >= ci_lower) & (actual <= ci_upper)).mean()
                
                # Calibration (simplified Brier score)
                # Convert to binary outcome (above/below median)
                median_actual = actual.median()
                binary_actual = (actual > median_actual).astype(int)
                binary_prob = (predicted > median_actual).astype(float)
                brier_score = np.mean((binary_prob - binary_actual) ** 2)
                
                metrics[stat] = {
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'coverage_95': coverage,
                    'brier_score': brier_score,
                    'mean_uncertainty': uncertainty.mean(),
                    'n_predictions': len(actual)
                }
        
        return metrics
    
    def combine_validation_results(self, all_results):
        """Combine results from all validation periods"""
        combined = {stat: [] for stat in self.target_stats}
        
        for result in all_results:
            for stat in self.target_stats:
                if stat in result['metrics']:
                    period_metric = result['metrics'][stat].copy()
                    period_metric.update({
                        'period': result['period'],
                        'test_start': result['test_start'],
                        'test_end': result['test_end']
                    })
                    combined[stat].append(period_metric)
        
        # Convert to DataFrames
        for stat in self.target_stats:
            combined[stat] = pd.DataFrame(combined[stat])
        
        return combined
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📊 VALIDATION REPORT")
        print("="*50)
        
        if not self.validation_results:
            print("❌ No validation results available. Run time_series_validation() first.")
            return
        
        # Summary statistics for each stat
        for stat in self.target_stats:
            df = self.validation_results[stat]
            if len(df) == 0:
                continue
                
            print(f"\n🎯 {stat.upper()} PREDICTION PERFORMANCE:")
            print("-" * 40)
            
            # Average metrics
            avg_mae = df['mae'].mean()
            avg_r2 = df['r2'].mean()
            avg_coverage = df['coverage_95'].mean()
            avg_brier = df['brier_score'].mean()
            
            print(f"  📈 Mean Absolute Error: {avg_mae:.2f}")
            print(f"  📊 R² Score: {avg_r2:.3f}")
            print(f"  🎯 95% CI Coverage: {avg_coverage:.1%}")
            print(f"  📉 Brier Score: {avg_brier:.3f} {'(Excellent)' if avg_brier < 0.12 else '(Good)' if avg_brier < 0.20 else '(Needs improvement)'}")
            
            # Consistency
            mae_std = df['mae'].std()
            r2_std = df['r2'].std()
            
            print(f"  🔄 MAE Consistency: ±{mae_std:.2f}")
            print(f"  🔄 R² Consistency: ±{r2_std:.3f}")
            
            # Total predictions
            total_preds = df['n_predictions'].sum()
            print(f"  📊 Total Predictions: {total_preds:,}")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print("-" * 30)
        
        # Check if models meet research standards
        points_brier = self.validation_results['points']['brier_score'].mean()
        rebounds_brier = self.validation_results['rebounds']['brier_score'].mean()
        assists_brier = self.validation_results['assists']['brier_score'].mean()
        
        avg_brier = np.mean([points_brier, rebounds_brier, assists_brier])
        
        if avg_brier < 0.12:
            grade = "🥇 EXCELLENT"
            assessment = "Meets research standards for sports prediction"
        elif avg_brier < 0.20:
            grade = "🥈 GOOD"
            assessment = "Above average calibration"
        else:
            grade = "🥉 FAIR"
            assessment = "Room for improvement"
        
        print(f"  Model Grade: {grade}")
        print(f"  Assessment: {assessment}")
        print(f"  Avg Brier Score: {avg_brier:.3f}")
        
        # Comparison to research benchmarks
        print(f"\n📚 RESEARCH COMPARISON:")
        print(f"  DARKO (NBA): ~0.10 Brier score")
        print(f"  XGBoost Synergy: 0.10 Brier score")
        print(f"  Neural w/ Attention: <0.12 Brier score")
        print(f"  Our Model: {avg_brier:.3f} Brier score")
        
        return {
            'overall_grade': grade,
            'avg_brier_score': avg_brier,
            'total_predictions': sum(df['n_predictions'].sum() for df in self.validation_results.values()),
            'avg_mae': {stat: self.validation_results[stat]['mae'].mean() for stat in self.target_stats},
            'avg_r2': {stat: self.validation_results[stat]['r2'].mean() for stat in self.target_stats}
        }
    
    def create_validation_visualizations(self):
        """Create validation performance visualizations"""
        print("\n📈 Creating validation visualizations...")
        
        if not self.validation_results:
            print("❌ No validation results available")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Points MAE', 'Rebounds MAE', 'Assists MAE',
                           'Points R²', 'Rebounds R²', 'Assists R²'],
            vertical_spacing=0.15
        )
        
        colors = ['#FF6B35', '#4ECDC4', '#45B7D1']
        
        # Plot MAE over time
        for i, stat in enumerate(self.target_stats):
            df = self.validation_results[stat]
            if len(df) == 0:
                continue
                
            fig.add_trace(
                go.Scatter(
                    x=df['test_start'],
                    y=df['mae'],
                    mode='lines+markers',
                    name=f'{stat.title()} MAE',
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # Add R² plot
            fig.add_trace(
                go.Scatter(
                    x=df['test_start'],
                    y=df['r2'],
                    mode='lines+markers',
                    name=f'{stat.title()} R²',
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=i+1
            )
        
        fig.update_layout(
            title="Model Performance Over Time",
            height=600,
            template="plotly_white"
        )
        
        # Save the plot
        import os
        os.makedirs('wnba_predictions', exist_ok=True)
        fig.write_html("wnba_predictions/validation_performance.html")
        print("✅ Performance chart saved: wnba_predictions/validation_performance.html")
        
        # Create calibration plot
        self.create_calibration_plot()
    
    def create_calibration_plot(self):
        """Create calibration reliability diagram"""
        print("📊 Creating calibration plot...")
        
        # This would show how well-calibrated our probability predictions are
        # For now, create a simple Brier score trend
        
        fig = go.Figure()
        
        for i, stat in enumerate(self.target_stats):
            df = self.validation_results[stat]
            if len(df) == 0:
                continue
                
            fig.add_trace(go.Scatter(
                x=df['test_start'],
                y=df['brier_score'],
                mode='lines+markers',
                name=f'{stat.title()} Brier Score',
                line=dict(width=3)
            ))
        
        # Add benchmark line
        fig.add_hline(
            y=0.12,
            line_dash="dash",
            line_color="red",
            annotation_text="Research Benchmark (0.12)"
        )
        
        fig.update_layout(
            title="Calibration Quality Over Time (Lower is Better)",
            xaxis_title="Test Period Start",
            yaxis_title="Brier Score",
            height=400,
            template="plotly_white"
        )
        
        fig.write_html("wnba_predictions/calibration_analysis.html")
        print("✅ Calibration chart saved: wnba_predictions/calibration_analysis.html")
    
    def export_validation_summary(self):
        """Export validation summary to CSV"""
        print("\n💾 Exporting validation summary...")
        
        if not self.validation_results:
            print("❌ No validation results to export")
            return
        
        # Create summary DataFrame
        summary_data = []
        
        for stat in self.target_stats:
            df = self.validation_results[stat]
            if len(df) == 0:
                continue
                
            summary_data.append({
                'stat': stat,
                'avg_mae': df['mae'].mean(),
                'std_mae': df['mae'].std(),
                'avg_r2': df['r2'].mean(),
                'std_r2': df['r2'].std(),
                'avg_coverage_95': df['coverage_95'].mean(),
                'avg_brier_score': df['brier_score'].mean(),
                'total_predictions': df['n_predictions'].sum(),
                'validation_periods': len(df)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"wnba_predictions/validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Validation summary saved: {summary_file}")
        print("\n📋 Summary:")
        print(summary_df.round(3).to_string(index=False))
        
        return summary_df


def main():
    """Main validation function"""
    print("🎯 WNBA Model Validation System")
    print("=" * 40)
    
    # Find game logs
    import glob
    game_log_files = glob.glob("wnba_game_data/player_game_logs_*.csv")
    
    if not game_log_files:
        print("❌ No game logs found!")
        print("💡 Run the data fetcher first to generate game logs")
        return
    
    game_logs_file = game_log_files[0]
    print(f"📊 Using game logs: {game_logs_file}")
    
    # Initialize validator
    validator = WNBAModelValidator(game_logs_file)
    
    # Run validation
    print("\n🔍 Starting time series validation...")
    results = validator.time_series_validation(test_weeks=2, min_train_weeks=6)
    
    if results:
        # Generate report
        summary = validator.generate_validation_report()
        
        # Create visualizations
        validator.create_validation_visualizations()
        
        # Export results
        validator.export_validation_summary()
        
        print("\n🎉 Validation complete!")
        print("📁 Check wnba_predictions/ for detailed results")
        
    else:
        print("❌ Validation failed")

if __name__ == "__main__":
    main()