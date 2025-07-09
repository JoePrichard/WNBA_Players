# alternative_data_sources.py - Alternative WNBA Data Sources
#!/usr/bin/env python3
"""
Alternative WNBA Data Sources

When Basketball Reference doesn't have immediate detailed stats,
try these alternative approaches:

1. ESPN API endpoints
2. WNBA.com official API
3. Alternative sports data providers
4. Manual CSV import workflows
"""

import requests
import pandas as pd
from datetime import datetime
import json

class WNBAAlternativeData:
    """Alternative data sources for WNBA statistics"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WNBA-Analytics/1.0)',
            'Accept': 'application/json'
        })
    
    def try_espn_api(self, date: str) -> dict:
        """
        Try ESPN's WNBA API endpoints
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with game data or error info
        """
        date_formatted = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        
        endpoints = [
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date_formatted}",
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/schedule?dates={date_formatted}",
            f"https://www.espn.com/wnba/schedule/_/date/{date_formatted}"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.session.get(endpoint, timeout=10)
                if response.status_code == 200:
                    if 'application/json' in response.headers.get('content-type', ''):
                        data = response.json()
                        if data.get('events'):
                            return {
                                'success': True,
                                'source': 'ESPN API',
                                'endpoint': endpoint,
                                'games': len(data['events']),
                                'data': data
                            }
            except Exception as e:
                continue
        
        return {'success': False, 'error': 'No ESPN data found'}
    
    def try_wnba_official_api(self, date: str) -> dict:
        """
        Try WNBA.com official API endpoints
        
        Note: These may require API keys or have different access patterns
        """
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Try various WNBA.com endpoints
        endpoints = [
            f"https://www.wnba.com/stats/api/schedule/{date_obj.year}/{date_obj.month:02d}/{date_obj.day:02d}",
            f"https://stats.wnba.com/stats/scoreboardV2?DayOffset=0&GameDate={date}&LeagueID=10",
            f"https://data.wnba.com/data/5s/v2015/json/mobile_teams/wnba/2025/scores/gameday_{date_obj.strftime('%Y%m%d')}_scores.json"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.session.get(endpoint, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'source': 'WNBA Official',
                        'endpoint': endpoint,
                        'data': data
                    }
            except Exception as e:
                continue
        
        return {'success': False, 'error': 'No WNBA official data found'}
    
    def create_manual_csv_template(self) -> str:
        """
        Create a CSV template for manual data entry
        
        Returns:
            Path to created template file
        """
        columns = [
            'Date', 'Player', 'Team', 'Opponent', 'Home/Away',
            'MP', 'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
            'BLK', 'TOV', 'PF', '+/-'
        ]
        
        # Create sample row for reference
        sample_data = {
            'Date': '2025-07-03',
            'Player': 'Example Player',
            'Team': 'ATL',
            'Opponent': 'SEA',
            'Home/Away': 'Home',
            'MP': '32.5',
            'PTS': '18',
            'FG': '7',
            'FGA': '15',
            'FG%': '0.467',
            '3P': '2',
            '3PA': '5',
            '3P%': '0.400',
            'FT': '2',
            'FTA': '2',
            'FT%': '1.000',
            'ORB': '1',
            'DRB': '6',
            'TRB': '7',
            'AST': '4',
            'STL': '2',
            'BLK': '0',
            'TOV': '3',
            'PF': '2',
            '+/-': '+8'
        }
        
        df = pd.DataFrame([sample_data])
        
        # Add empty rows for manual entry
        for i in range(20):
            empty_row = {col: '' for col in columns}
            df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
        
        filename = 'wnba_manual_template.csv'
        df.to_csv(filename, index=False)
        
        return filename
    
    def suggest_data_sources(self) -> list:
        """
        Suggest alternative data sources and methods
        
        Returns:
            List of suggestions with details
        """
        suggestions = [
            {
                'method': 'ESPN Box Scores',
                'url': 'https://www.espn.com/wnba/scoreboard',
                'description': 'Manual extraction from ESPN game pages',
                'pros': 'Usually has immediate post-game stats',
                'cons': 'Requires manual work or custom scraping'
            },
            {
                'method': 'WNBA.com Stats',
                'url': 'https://www.wnba.com/stats',
                'description': 'Official WNBA statistics portal',
                'pros': 'Official source, comprehensive data',
                'cons': 'May require navigation through multiple pages'
            },
            {
                'method': 'Sports APIs',
                'url': 'Various (SportRadar, Rapid API, etc.)',
                'description': 'Commercial sports data APIs',
                'pros': 'Reliable, structured data',
                'cons': 'Often require paid subscriptions'
            },
            {
                'method': 'Manual CSV Entry',
                'url': 'Local file',
                'description': 'Manually enter key stats into CSV template',
                'pros': 'Guaranteed accuracy, immediate availability',
                'cons': 'Time-consuming, not scalable'
            },
            {
                'method': 'Wait for Basketball Reference',
                'url': 'https://www.basketball-reference.com/wnba/',
                'description': 'Wait 24-48 hours for detailed stats to appear',
                'pros': 'Most comprehensive historical data',
                'cons': 'Delayed availability'
            }
        ]
        
        return suggestions


def test_alternative_sources():
    """Test alternative data sources for a specific date"""
    test_date = '2025-07-03'
    
    print(f"🔍 Testing Alternative Data Sources for {test_date}")
    print("=" * 50)
    
    alt_data = WNBAAlternativeData()
    
    # Test ESPN API
    print("\n📺 Testing ESPN API...")
    espn_result = alt_data.try_espn_api(test_date)
    if espn_result['success']:
        print(f"✅ ESPN Success: Found {espn_result['games']} games")
        print(f"   Source: {espn_result['source']}")
    else:
        print(f"❌ ESPN Failed: {espn_result['error']}")
    
    # Test WNBA Official API
    print("\n🏀 Testing WNBA Official API...")
    wnba_result = alt_data.try_wnba_official_api(test_date)
    if wnba_result['success']:
        print(f"✅ WNBA Official Success")
        print(f"   Source: {wnba_result['source']}")
    else:
        print(f"❌ WNBA Official Failed: {wnba_result['error']}")
    
    # Create manual template
    print("\n📝 Creating Manual CSV Template...")
    template_file = alt_data.create_manual_csv_template()
    print(f"✅ Template created: {template_file}")
    
    # Show suggestions
    print("\n💡 Alternative Data Source Suggestions:")
    suggestions = alt_data.suggest_data_sources()
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['method']}")
        print(f"   URL: {suggestion['url']}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Pros: {suggestion['pros']}")
        print(f"   Cons: {suggestion['cons']}")


if __name__ == "__main__":
    test_alternative_sources()