# 🏀 WNBA Points Prediction Dashboard

## Interactive Streamlit Dashboard for WNBA Player Predictions

This dashboard uses advanced machine learning to predict WNBA player scoring with confidence intervals, featuring all 13 teams including the Golden State Valkyries.

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
python setup.py

# Follow the prompts - it will:
# 1. Install all dependencies
# 2. Train ML models  
# 3. Launch the dashboard
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (5-10 minutes)
python models.py

# 3. Launch dashboard
streamlit run app.py
```

### Option 3: Super Quick Start
```bash
# Just run this - handles everything automatically
python run.py
```

---

## 📁 File Structure

```
wnba-dashboard/
├── app.py                  # Main Streamlit application
├── models.py              # ML training script (from previous step)
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
├── run.py                # Quick launch script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── wnba_stats/           # Generated data files
    ├── wnba_players_*.csv
    └── wnba_teams_*.csv
```

---

## 🎯 Dashboard Features

### 🔮 Make Predictions
- **Existing Players**: Select any WNBA player and predict their scoring
- **Custom Players**: Create hypothetical players with custom statistics
- **Confidence Intervals**: 68% and 95% prediction ranges
- **Real-time Results**: Instant predictions with uncertainty quantification

### 🏆 Team Analysis
- **All 13 Teams**: Including Golden State Valkyries
- **Performance Metrics**: Scoring, efficiency, assists, rebounds
- **Interactive Charts**: Team comparisons and rankings
- **Valkyries Highlighting**: Special attention to the newest franchise

### ⭐ Player Rankings
- **Top Scorers**: League-leading performers
- **Player Comparison**: Head-to-head statistical analysis
- **Position Analysis**: Performance by player position
- **Interactive Visualizations**: Radar charts and scatter plots

### 🔍 Model Insights
- **Feature Importance**: What drives scoring predictions
- **Model Performance**: Accuracy metrics and validation
- **Algorithm Details**: Understanding the ML pipeline
- **Prediction Uncertainty**: How confident are the predictions

---

## 🎮 How to Use

### 1. Launch the Dashboard
```bash
streamlit run app.py
```

### 2. Navigate Through Sections
- Use the **sidebar** to switch between different analysis modes
- Each section offers unique insights and interactive features

### 3. Train Models (First Time)
- Click **"Train/Update Models"** in the sidebar
- Wait 5-10 minutes for training to complete
- Models are saved and reused automatically

### 4. Make Predictions
- **Existing Players**: Choose from dropdown, click predict
- **Custom Players**: Fill out the form with desired statistics
- View results with confidence intervals and explanations

---

## 🔧 Customization

### Changing Colors/Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### Adding New Features
Modify `app.py` to add:
- New prediction models
- Additional statistics
- Custom visualizations
- Enhanced UI components

### Model Configuration
Edit the model parameters in the `train_models_if_needed()` function:
```python
# Add new models or modify existing ones
models_to_train = {
    'your_model': YourModelClass(parameters),
    # ... existing models
}
```

---

## 📊 Data Sources

- **Player Statistics**: Comprehensive WNBA player data
- **Team Metrics**: Team-level performance statistics  
- **2025 Season**: Includes Golden State Valkyries
- **Regular Updates**: Data can be refreshed by re-running `models.py`

---

## 🤖 Machine Learning Models

The dashboard uses an ensemble of 6+ machine learning algorithms:

1. **XGBoost** - Gradient boosting (primary model)
2. **Random Forest** - Ensemble of decision trees
3. **Neural Network** - PyTorch deep learning with uncertainty
4. **LightGBM** - Microsoft gradient boosting
5. **Bayesian Ridge** - Probabilistic linear model
6. **Linear Regression** - Simple baseline model

### Ensemble Prediction
- Combines all models for robust predictions
- Weighted average based on individual model performance
- Monte Carlo dropout for uncertainty estimation

---

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- 📱 **Mobile Phones** - Touch-friendly interface
- 📱 **Tablets** - Optimized layout
- 💻 **Desktop** - Full feature access
- 🖥️ **Large Screens** - Enhanced visualizations

---

## 🔍 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"No data files found"**
```bash
python models.py  # Generate data first
```

**"Port already in use"**
```bash
streamlit run app.py --server.port 8502
```

**"Models not trained"**
- Click "Train/Update Models" in the sidebar
- Or run `python models.py` manually

### Performance Tips

- **First Launch**: May take longer due to model loading
- **Large Datasets**: Use filters to improve performance
- **Memory Usage**: Close other applications if running slowly
- **Browser**: Chrome/Firefox recommended for best experience

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker
```bash
# Build image
docker build -t wnba-dashboard .

# Run container
docker run -p 8501:8501 wnba-dashboard
```

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
git push heroku main
```

---

## 📈 Advanced Features

### Prediction Export
- Download prediction results as CSV
- Save custom player configurations
- Export visualizations as images

### Batch Predictions
- Predict multiple players at once
- Compare team rosters
- Analyze trade scenarios

### Statistical Analysis
- Correlation analysis
- Performance trends
- Outlier detection

---

## 🔮 Future Enhancements

Planned features for future versions:

- **🏆 Playoff Predictions** - Postseason performance modeling
- **📊 Advanced Metrics** - Player efficiency ratings, VORP, BPM
- **🔄 Live Updates** - Real-time data integration
- **👥 Multiplayer Mode** - Fantasy league integration
- **📱 Mobile App** - Native iOS/Android versions
- **🤖 AI Insights** - Natural language explanations

---

## 🤝 Contributing

Want to improve the dashboard?

1. **Fork the repository**
2. **Add new features** - Models, visualizations, UI improvements
3. **Test thoroughly** - Ensure everything works
4. **Submit pull request** - Share your improvements

### Ideas for Contributions
- New machine learning models
- Enhanced visualizations
- Mobile UI improvements
- Performance optimizations
- Additional statistics

---

## 📜 License

This project is open source and available under the MIT License.

---

## 🏀 About

Created for WNBA fans, fantasy players, analysts, and anyone interested in basketball analytics. This dashboard demonstrates the power of machine learning applied to sports prediction while maintaining transparency about uncertainty and model limitations.

**Built with**: Streamlit, PyTorch, XGBoost, Plotly, and lots of ❤️ for basketball!

---

## 📞 Support

Need help? Here are your options:

1. **Check this README** - Most questions answered here
2. **Run the setup script** - `python setup.py` handles most issues
3. **Check file structure** - Ensure all files are in place
4. **Verify Python version** - Requires Python 3.8+

---

## 🎉 Enjoy!

Have fun exploring WNBA predictions and discovering insights about your favorite players and teams! 

**Go Valkyries!** 🗡️⚔️

---

*Last updated: June 2025 | WNBA Season featuring Golden State Valkyries*