#!/usr/bin/env python3
"""
WNBA Data Fetcher - Fixed for Real Basketball Reference Format
Handles actual Basketball Reference WNBA data formats and edge cases.
"""

import requests
import pandas as pd
import time
from datetime import datetime, date, timedelta
import os
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import asdict
import re

from wnba_data_models import (
    PlayerGameLog, TeamGameLog, GameSchedule, HomeAway, GameResult,
    WNBADataError
)


class WNBADataFetcher:
    """
    Fetches WNBA data from Basketball Reference with robust error handling.
    
    This improved version handles:
    - Multiple date formats from Basketball Reference
    - Flexible table parsing
    - Better error recovery
    - Graceful degradation when data is incomplete
    
    Attributes:
        session: HTTP session for requests
        base_url: Base URL for data source
        headers: HTTP headers for requests
        rate_limit_delay: Delay between requests in seconds
        max_retries: Maximum retry attempts for failed requests
        logger: Logger instance
    """
    
    def __init__(
        self,
        base_url: str = "https://www.basketball-reference.com/wnba",
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
        user_agent: str = "WNBA-Analytics-Bot/1.0"
    ):
        """
        Initialize the WNBA data fetcher.
        
        Args:
            base_url (str): Base URL for Basketball Reference WNBA data
            rate_limit_delay (float): Seconds to wait between requests
            max_retries (int): Maximum number of retry attempts
            user_agent (str): User agent string for requests
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str) -> Optional[str]:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            Optional[str]: HTML content if successful, None if failed
            
        Raises:
            WNBADataError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Fetching URL: {url} (attempt {attempt + 1})")
                time.sleep(self.rate_limit_delay)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                if response.status_code == 200:
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise WNBADataError(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                time.sleep(self.rate_limit_delay * (attempt + 1))
        
        return None

    def _parse_date_flexible(self, date_str: str) -> Optional[date]:
        """
        Parse date string using multiple possible formats.
        
        Basketball Reference uses various date formats, so we need to be flexible.
        
        Args:
            date_str (str): Date string to parse
            
        Returns:
            Optional[date]: Parsed date or None if parsing fails
        """
        if not date_str or date_str.strip() == "":
            return None
        
        # Common Basketball Reference date formats
        date_formats = [
            '%Y-%m-%d',                    # 2025-05-16
            '%a, %b %d, %Y',              # Fri, May 16, 2025
            '%B %d, %Y',                  # May 16, 2025
            '%m/%d/%Y',                   # 05/16/2025
            '%m-%d-%Y',                   # 05-16-2025
            '%d %b %Y',                   # 16 May 2025
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        
        # Try to extract date using regex as fallback
        date_patterns = [
            r'(\w+,?\s+\w+\s+\d+,?\s+\d{4})',  # Fri, May 16, 2025 or May 16, 2025
            r'(\d{4}-\d{2}-\d{2})',             # 2025-05-16
            r'(\d{1,2}/\d{1,2}/\d{4})',         # 5/16/2025
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    extracted_date = match.group(1)
                    for fmt in date_formats:
                        try:
                            return datetime.strptime(extracted_date, fmt).date()
                        except ValueError:
                            continue
                except:
                    continue
        
        self.logger.warning(f"Could not parse date: '{date_str}'")
        return None

    def _parse_basketball_reference_table(
        self, 
        soup: BeautifulSoup, 
        table_id: str,
        flexible: bool = True
    ) -> List[Dict]:
        """
        Parse Basketball Reference table with flexible column handling.
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object of the page
            table_id (str): ID of the table to parse
            flexible (bool): Whether to use flexible parsing for missing columns
            
        Returns:
            List[Dict]: List of dictionaries containing table data
            
        Raises:
            WNBADataError: If table not found or parsing fails critically
        """
        table = soup.find('table', {'id': table_id})
        if not table:
            # Try alternative table IDs
            alternative_ids = [
                table_id.replace('_', ''),
                table_id + '_table',
                'schedule',
                'games',
                'stats',
                'per_game_stats'
            ]
            
            for alt_id in alternative_ids:
                table = soup.find('table', {'id': alt_id})
                if table:
                    self.logger.info(f"Found table with alternative ID: {alt_id}")
                    break
            
            if not table:
                raise WNBADataError(f"No table found with ID '{table_id}' or alternatives")
        
        # Get headers with flexible approach
        headers = []
        thead = table.find('thead')
        
        if thead:
            # Try to find the actual header row (might not be the first)
            header_rows = thead.find_all('tr')
            for header_row in header_rows:
                potential_headers = []
                for th in header_row.find_all(['th', 'td']):
                    header_text = th.get('data-stat') or th.text.strip()
                    if header_text and header_text not in ['', 'Rk']:  # Skip rank columns
                        potential_headers.append(header_text)
                
                if len(potential_headers) > len(headers):
                    headers = potential_headers
        
        if not headers:
            # Fallback: try to infer headers from first data row
            tbody = table.find('tbody')
            if tbody:
                first_row = tbody.find('tr')
                if first_row:
                    headers = [f"col_{i}" for i, _ in enumerate(first_row.find_all(['td', 'th']))]
        
        if not headers:
            raise WNBADataError(f"No headers found for table '{table_id}'")
        
        self.logger.debug(f"Found headers: {headers[:10]}...")  # Log first 10 headers
        
        # Get data rows
        data = []
        tbody = table.find('tbody')
        if not tbody:
            return data
        
        for row in tbody.find_all('tr'):
            # Skip header rows within tbody and empty rows
            if 'thead' in row.get('class', []) or not row.find(['td', 'th']):
                continue
            
            row_data = {}
            cells = row.find_all(['th', 'td'])
            
            for i, cell in enumerate(cells):
                if i < len(headers):
                    header = headers[i]
                    value = cell.text.strip()
                    
                    # Clean up common Basketball Reference formatting
                    if value in ['', 'N/A', '--', '—']:
                        value = None
                    else:
                        # Try to convert to appropriate data type
                        value = self._convert_cell_value(value)
                    
                    row_data[header] = value
            
            if row_data and any(v is not None for v in row_data.values()):
                data.append(row_data)
        
        self.logger.info(f"Parsed {len(data)} rows from table '{table_id}'")
        return data

    def _convert_cell_value(self, value: str) -> Union[str, int, float, None]:
        """
        Convert cell value to appropriate Python type.
        
        Args:
            value (str): Raw cell value
            
        Returns:
            Union[str, int, float, None]: Converted value
        """
        if not value or value.strip() == "":
            return None
        
        value = value.strip()
        
        # Try integer conversion
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            if '.' in value or value.replace('.', '').replace('-', '').isdigit():
                return float(value)
        except ValueError:
            pass
        
        # Try percentage conversion
        if value.endswith('%'):
            try:
                return float(value[:-1]) / 100
            except ValueError:
                pass
        
        # Return as string
        return value

    def fetch_season_schedule(self, year: int) -> List[GameSchedule]:
        """
        Fetch season schedule with improved date parsing.
        
        Args:
            year (int): Season year (e.g., 2025)
            
        Returns:
            List[GameSchedule]: List of GameSchedule objects
            
        Raises:
            WNBADataError: If schedule data cannot be fetched
        """
        url = f"{self.base_url}/years/{year}_games.html"
        html = self._make_request(url)
        
        if not html:
            raise WNBADataError(f"Could not fetch schedule for year {year}")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            games_data = self._parse_basketball_reference_table(soup, 'games', flexible=True)
        except WNBADataError:
            # Try alternative table parsing
            try:
                games_data = self._parse_basketball_reference_table(soup, 'schedule', flexible=True)
            except WNBADataError:
                self.logger.warning(f"Could not find games table for year {year}")
                return []
        
        if not games_data:
            self.logger.warning(f"No games found for year {year}")
            return []
        
        schedule = []
        for game_data in games_data:
            try:
                # Try multiple possible column names for date
                date_value = None
                for date_col in ['date_game', 'date', 'Date', 'game_date']:
                    if date_col in game_data and game_data[date_col]:
                        date_value = game_data[date_col]
                        break
                
                if not date_value:
                    continue
                
                # Parse date using flexible parser
                game_date = self._parse_date_flexible(str(date_value))
                if not game_date:
                    continue
                
                # Try multiple possible column names for teams
                home_team = None
                away_team = None
                
                for home_col in ['home_team_name', 'home_team', 'Home', 'home']:
                    if home_col in game_data and game_data[home_col]:
                        home_team = str(game_data[home_col]).strip()
                        break
                
                for away_col in ['visitor_team_name', 'away_team', 'visitor', 'Away', 'away']:
                    if away_col in game_data and game_data[away_col]:
                        away_team = str(game_data[away_col]).strip()
                        break
                
                if not (home_team and away_team):
                    continue
                
                # Get game time
                game_time = "TBD"
                for time_col in ['game_start_time', 'start_time', 'time', 'Time']:
                    if time_col in game_data and game_data[time_col]:
                        game_time = str(game_data[time_col])
                        break
                
                # Get status
                status = "scheduled"
                for status_col in ['game_status', 'status', 'Status']:
                    if status_col in game_data and game_data[status_col]:
                        status = str(game_data[status_col]).lower()
                        break
                
                game_schedule = GameSchedule(
                    game_id=f"{game_date}_{away_team}_{home_team}",
                    date=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    game_time=game_time,
                    status=status
                )
                
                schedule.append(game_schedule)
                
            except Exception as e:
                self.logger.debug(f"Skipping game data due to error: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(schedule)} games for {year} season")
        return schedule

    def fetch_player_season_stats(self, year: int) -> pd.DataFrame:
        """
        Fetch player season statistics with flexible parsing.
        
        Args:
            year (int): Season year
            
        Returns:
            pd.DataFrame: DataFrame with player season stats
            
        Raises:
            WNBADataError: If player stats cannot be fetched
        """
        url = f"{self.base_url}/years/{year}_per_game.html"
        html = self._make_request(url)
        
        if not html:
            raise WNBADataError(f"Could not fetch player stats for year {year}")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            players_data = self._parse_basketball_reference_table(soup, 'per_game_stats', flexible=True)
        except WNBADataError:
            try:
                players_data = self._parse_basketball_reference_table(soup, 'stats', flexible=True)
            except WNBADataError:
                self.logger.warning(f"Could not find player stats table for year {year}")
                return pd.DataFrame()
        
        if not players_data:
            self.logger.warning(f"No player stats found for year {year}")
            return pd.DataFrame()
        
        df = pd.DataFrame(players_data)
        
        # Standardize column names for common stats
        column_mapping = {
            'player': ['player', 'Player', 'name', 'Name'],
            'team': ['team_id', 'team', 'Team', 'tm', 'Tm'],
            'points': ['pts', 'points', 'Points', 'PTS'],
            'rebounds': ['trb', 'rebounds', 'Rebounds', 'TRB', 'reb'],
            'assists': ['ast', 'assists', 'Assists', 'AST'],
            'minutes': ['mp', 'minutes', 'Minutes', 'MP', 'min'],
            'games': ['g', 'games', 'Games', 'G', 'gp']
        }
        
        for standard_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns:
                    df = df.rename(columns={col: standard_col})
                    break
        
        self.logger.info(f"Fetched stats for {len(df)} players for {year} season")
        return df

    def create_sample_player_data(self, year: int) -> pd.DataFrame:
        """
        Create sample player game log data when real data is not available.
        
        This is a fallback for development/testing purposes.
        
        Args:
            year (int): Season year
            
        Returns:
            pd.DataFrame: Sample player game log data
        """
        import random
        
        self.logger.info(f"Creating sample player data for {year} (development mode)")
        
        # Sample WNBA teams and players
        teams = {
            'LAS': ['A\'ja Wilson', 'Kelsey Plum', 'Jackie Young'],
            'NY': ['Sabrina Ionescu', 'Breanna Stewart', 'Jonquel Jones'],
            'CHI': ['Chennedy Carter', 'Angel Reese', 'Dana Evans'],
            'CONN': ['Alyssa Thomas', 'DeWanna Bonner', 'DiJonai Carrington'],
            'IND': ['Caitlin Clark', 'Aliyah Boston', 'Kelsey Mitchell']
        }
        
        data = []
        game_num = 1
        start_date = date(year, 5, 15)  # WNBA season typically starts mid-May
        
        for week in range(20):  # 20 weeks of season
            week_date = start_date + timedelta(weeks=week)
            
            for team, players in teams.items():
                for player in players:
                    # Generate realistic stats
                    if 'Wilson' in player or 'Stewart' in player:
                        # Star player stats
                        points = random.normalvariate(22, 4)
                        rebounds = random.normalvariate(9, 2)
                        assists = random.normalvariate(4, 1.5)
                        minutes = random.normalvariate(32, 3)
                    elif 'Clark' in player or 'Ionescu' in player:
                        # Point guard stats
                        points = random.normalvariate(18, 3)
                        rebounds = random.normalvariate(5, 1.5)
                        assists = random.normalvariate(8, 2)
                        minutes = random.normalvariate(30, 3)
                    else:
                        # Role player stats
                        points = random.normalvariate(12, 3)
                        rebounds = random.normalvariate(6, 2)
                        assists = random.normalvariate(3, 1)
                        minutes = random.normalvariate(25, 4)
                    
                    # Ensure non-negative values
                    points = max(0, points)
                    rebounds = max(0, rebounds)
                    assists = max(0, assists)
                    minutes = max(5, minutes)
                    
                    # Random opponent
                    opponents = [t for t in teams.keys() if t != team]
                    opponent = random.choice(opponents)
                    
                    data.append({
                        'player': player,
                        'team': team,
                        'game_num': game_num,
                        'date': week_date,
                        'opponent': opponent,
                        'home_away': random.choice(['H', 'A']),
                        'result': random.choice(['W', 'L']),
                        'minutes': round(minutes, 1),
                        'points': round(points, 1),
                        'rebounds': round(rebounds, 1),
                        'assists': round(assists, 1),
                        'fg_made': round(points / 2.2, 1),
                        'fg_attempted': round(points / 1.5, 1),
                        'ft_made': round(points * 0.15, 1),
                        'ft_attempted': round(points * 0.2, 1),
                        'turnovers': round(assists * 0.6, 1),
                        'steals': round(random.normalvariate(1.2, 0.5), 1),
                        'blocks': round(random.normalvariate(0.5, 0.3), 1),
                        'fouls': round(random.normalvariate(2.5, 0.8), 1),
                        'rest_days': random.randint(1, 4)
                    })
            
            game_num += 1
        
        df = pd.DataFrame(data)
        self.logger.info(f"Created sample data: {len(df)} game logs for {len(teams) * 3} players")
        return df

    def get_todays_games(self) -> List[GameSchedule]:
        """
        Get today's scheduled games.
        
        Returns:
            List[GameSchedule]: List of GameSchedule objects for today
            
        Raises:
            WNBADataError: If today's schedule cannot be fetched
        """
        today = date.today()
        current_year = today.year
        
        try:
            # Try current year first
            full_schedule = self.fetch_season_schedule(current_year)
            todays_games = [game for game in full_schedule if game.date == today]
            
            if not todays_games and today.month >= 10:
                # If no games and it's late in year, try next year
                next_year_schedule = self.fetch_season_schedule(current_year + 1)
                todays_games = [game for game in next_year_schedule if game.date == today]
            
            return todays_games
            
        except WNBADataError as e:
            self.logger.warning(f"Could not fetch today's games: {e}")
            return []

    def validate_data_availability(self, year: int) -> Dict[str, bool]:
        """
        Validate what data is available for a given year.
        
        Args:
            year (int): Year to check
            
        Returns:
            Dict[str, bool]: Dictionary indicating what data types are available
        """
        availability = {
            'schedule': False,
            'player_stats': False,
            'team_stats': False
        }
        
        try:
            schedule = self.fetch_season_schedule(year)
            availability['schedule'] = len(schedule) > 0
        except WNBADataError:
            pass
        
        try:
            player_stats = self.fetch_player_season_stats(year)
            availability['player_stats'] = not player_stats.empty
        except WNBADataError:
            pass
        
        return availability

    def export_data(
        self, 
        data: Union[List[GameSchedule], List[PlayerGameLog], List[TeamGameLog], pd.DataFrame],
        filename: str,
        output_dir: str = "wnba_game_data"
    ) -> str:
        """
        Export data to CSV file.
        
        Args:
            data: Data to export
            filename (str): Name of output file (without extension)
            output_dir (str): Output directory
            
        Returns:
            str: Path to exported file
            
        Raises:
            WNBADataError: If export fails
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{filename}_{timestamp}.csv")
        
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            elif isinstance(data, list) and data:
                # Convert dataclass objects to dictionaries
                if hasattr(data[0], '__dataclass_fields__'):
                    dict_data = [asdict(item) for item in data]
                    df = pd.DataFrame(dict_data)
                    df.to_csv(filepath, index=False)
                else:
                    df = pd.DataFrame(data)
                    df.to_csv(filepath, index=False)
            else:
                raise WNBADataError("No data provided for export")
            
            self.logger.info(f"Exported {len(data)} records to {filepath}")
            return filepath
            
        except Exception as e:
            raise WNBADataError(f"Failed to export data: {e}")


def main():
    """
    Test the improved data fetcher.
    """
    fetcher = WNBADataFetcher()
    
    current_year = 2025
    
    print(f"🏀 WNBA Data Fetcher - Testing Improved Version for {current_year}")
    print("=" * 60)
    
    # Check data availability
    availability = fetcher.validate_data_availability(current_year)
    
    print("📊 Data Availability Check:")
    for data_type, available in availability.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {data_type}: {status}")
    
    # Try to fetch schedule
    if availability['schedule']:
        try:
            schedule = fetcher.fetch_season_schedule(current_year)
            print(f"\n✅ Successfully fetched {len(schedule)} games")
            if schedule:
                print(f"   First game: {schedule[0].away_team} @ {schedule[0].home_team} on {schedule[0].date}")
        except Exception as e:
            print(f"\n❌ Schedule fetch failed: {e}")
    
    # If no real data available, create sample data for development
    if not any(availability.values()):
        print(f"\n💡 No real data available for {current_year}, creating sample data for development...")
        sample_data = fetcher.create_sample_player_data(current_year)
        sample_path = fetcher.export_data(sample_data, f"sample_player_logs_{current_year}")
        print(f"✅ Created sample data: {sample_path}")
    
    print("\n🎉 Data fetching test complete!")


if __name__ == "__main__":
    main()