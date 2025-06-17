#!/usr/bin/env python3
"""
WNBA Daily Predictions - One-Click Run Script
Automates the entire daily prediction workflow

Usage: python run_daily_predictions.py [--mode MODE]

Modes:
  setup    - Full setup (install, fetch data, train models)  
  update   - Update data and retrain models
  predict  - Generate today's predictions only
  dashboard - Launch dashboard (default)
"""

import os
import sys
import argparse
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

def print_banner():
    """Print application banner"""
    print("🏀" + "="*60 + "🏀")
    print("   WNBA DAILY GAME PREDICTIONS")
    print("🎯" + "="*60 + "🎯")
    print("📊 Points • Rebounds • Assists")
    print("🤖 ML Ensemble: XGBoost + Neural Networks + Bayesian")
    print("📈 Based on DARKO, Synergy Models, and Attention Networks")
    print("="*64)

def check_dependencies():
    """Quick dependency check"""
    required_packages = ['pandas', 'streamlit', 'torch', 'xgboost', 'plotly']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("💡 Run: python run_daily_predictions.py --mode setup")
        return False
    
    print("✅ All dependencies available")
    return True

def setup_mode():
    """Full setup mode"""
    print("\n🔧 FULL SETUP MODE")
    print("="*30)
    
    try:
        # Run the setup script
        from daily_prediction_setup import main as setup_main
        setup_main()
        return True
    except ImportError:
        print("❌ Setup script not found")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def update_mode():
    """Update data and retrain models"""
    print("\n🔄 UPDATE MODE")
    print("="*20)
    
    try:
        # 1. Fetch fresh data
        print("📊 Fetching latest game data...")
        from enhanced_data_fetcher import WNBAGameDataFetcher
        
        fetcher = WNBAGameDataFetcher()
        fetcher.fetch_all_game_data(2025)
        
        # 2. Retrain models
        print("\n🤖 Retraining models...")
        import glob
        
        game_log_files = glob.glob("wnba_game_data/player_game_logs_*.csv")
        if not game_log_files:
            print("❌ No game logs found")
            return False
        
        from daily_game_predictor import WNBADailyPredictor
        
        predictor = WNBADailyPredictor(game_log_files[0])
        predictor.train_all_models()
        
        print("✅ Update completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return False

def predict_mode():
    """Generate today's predictions"""
    print("\n🔮 PREDICTION MODE")
    print("="*25)
    
    try:
        import glob
        from datetime import datetime
        
        # Find latest models and data
        game_log_files = glob.glob("wnba_game_data/player_game_logs_*.csv")
        if not game_log_files:
            print("❌ No game data found. Run --mode update first")
            return False
        
        print(f"📊 Using data: {game_log_files[0]}")
        
        # Load predictor
        from daily_game_predictor import WNBADailyPredictor
        
        predictor = WNBADailyPredictor(game_log_files[0])
        
        # Check if models exist
        if not any(predictor.models.values()):
            print("🤖 No trained models found. Training now...")
            predictor.train_all_models()
        
        # Generate today's predictions
        sample_schedule = [
            {
                'game_id': f"{datetime.now().strftime('%Y-%m-%d')}_game1",
                'home_team': 'LAS',
                'away_team': 'NY',
                'game_time': '7:00 PM ET'
            },
            {
                'game_id': f"{datetime.now().strftime('%Y-%m-%d')}_game2",
                'home_team': 'ATL',
                'away_team': 'CHI',
                'game_time': '8:30 PM ET'
            }
        ]
        
        print("🔮 Generating predictions...")
        daily_preds = predictor.predict_daily_games(sample_schedule)
        
        if len(daily_preds) > 0:
            # Save predictions
            os.makedirs('wnba_predictions', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'wnba_predictions/daily_predictions_{timestamp}.csv'
            daily_preds.to_csv(output_file, index=False)
            
            print(f"✅ Predictions saved: {output_file}")
            print(f"📊 Generated {len(daily_preds)} player predictions")
            
            # Show sample
            print("\n📋 Sample Predictions:")
            sample_cols = ['player', 'team', 'predicted_points', 'predicted_rebounds', 'predicted_assists']
            print(daily_preds[sample_cols].head(8).to_string(index=False))
            
            return True
        else:
            print("❌ No predictions generated")
            return False
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def dashboard_mode():
    """Launch the dashboard"""
    print("\n🚀 DASHBOARD MODE")
    print("="*25)
    
    # Check if we have the required files
    required_files = ['daily_game_dashboard.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("🌐 Launching Streamlit dashboard...")
    print("📱 Browser should open automatically")
    print("🔗 URL: http://localhost:8501")
    print("💡 Press Ctrl+C to stop")
    print()
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", "daily_game_dashboard.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
            "--logger.level", "error"
        ]
        
        subprocess.check_call(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Dashboard error: {e}")
        print("💡 Try: streamlit run daily_game_dashboard.py")
    except FileNotFoundError:
        print("\n❌ Streamlit not installed")
        print("💡 Run: pip install streamlit")

def show_status():
    """Show system status"""
    print("\n📊 SYSTEM STATUS")
    print("="*25)
    
    # Check data files
    data_files = list(Path("wnba_game_data").glob("*.csv")) if Path("wnba_game_data").exists() else []
    print(f"📄 Data files: {len(data_files)}")
    
    # Check prediction files  
    pred_files = list(Path("wnba_predictions").glob("*.csv")) if Path("wnba_predictions").exists() else []
    print(f"🔮 Prediction files: {len(pred_files)}")
    
    # Check dependencies
    deps_ok = check_dependencies()
    print(f"📦 Dependencies: {'✅ OK' if deps_ok else '❌ Missing'}")
    
    # Latest prediction
    if pred_files:
        latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest_pred.stat().st_mtime)
        print(f"🕐 Latest predictions: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    return deps_ok and len(data_files) > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="WNBA Daily Game Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_daily_predictions.py                    # Launch dashboard
  python run_daily_predictions.py --mode setup       # Full setup
  python run_daily_predictions.py --mode update      # Update data & models  
  python run_daily_predictions.py --mode predict     # Generate predictions
  python run_daily_predictions.py --status           # Show system status
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['setup', 'update', 'predict', 'dashboard'],
        default='dashboard',
        help='Operation mode (default: dashboard)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true', 
        help='Show system status and exit'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Status check
    if args.status:
        show_status()
        return
    
    # Quick dependency check for non-setup modes
    if args.mode != 'setup':
        if not check_dependencies():
            print("\n💡 Run setup first: python run_daily_predictions.py --mode setup")
            return
    
    # Execute the requested mode
    success = False
    
    if args.mode == 'setup':
        success = setup_mode()
    elif args.mode == 'update':
        success = update_mode()
    elif args.mode == 'predict':
        success = predict_mode()
    elif args.mode == 'dashboard':
        dashboard_mode()  # This runs until interrupted
        success = True
    
    # Final status
    if success:
        print(f"\n🎉 {args.mode.upper()} completed successfully!")
        
        if args.mode != 'dashboard':
            print("💡 Next steps:")
            if args.mode == 'setup':
                print("  • Run: python run_daily_predictions.py --mode predict")
                print("  • Or: python run_daily_predictions.py (dashboard)")
            elif args.mode in ['update', 'predict']:
                print("  • Launch dashboard: python run_daily_predictions.py")
    else:
        print(f"\n❌ {args.mode.upper()} failed")
        print("💡 Check error messages above for troubleshooting")

if __name__ == "__main__":
    main()