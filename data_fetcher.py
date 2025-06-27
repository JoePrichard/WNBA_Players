# data_fetcher.py - WNBA Player Statistics Scraper (FIXED GAME DETECTION)
#!/usr/bin/env python3
"""
WNBA Player Statistics Scraper - Fixed Game Detection Version

CRITICAL FIXES:
- Updated team abbreviations to match current Basketball Reference
- Fixed URL format for game detection  
- Added multiple detection methods for robustness
- Better error handling and fallbacks
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import argparse
import logging
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wnba_scraper_working.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WNBAStatsScraper:
    """WNBA Statistics Scraper with FIXED game detection"""
    
    def __init__(self, delay: float = 10.0):
        """Initialize the scraper"""
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Base columns for output
        self.base_columns = ['Date', 'Player', 'Team', 'Opponent', 'Home/Away']
        self.all_stat_columns = set()
        
        # FIXED: Updated Basketball Reference team abbreviations for 2025
        self.bbref_teams = {
            'ATL': 'Atlanta Dream',
            'CHI': 'Chicago Sky', 
            'CONN': 'Connecticut Sun',  # Updated: CONN not CON
            'DAL': 'Dallas Wings',
            'IND': 'Indiana Fever',
            'LAS': 'Los Angeles Sparks',
            'LV': 'Las Vegas Aces',      # Updated: LV not LVA
            'MIN': 'Minnesota Lynx',
            'NY': 'New York Liberty',    # Updated: NY not NYL
            'PHX': 'Phoenix Mercury',    # Updated: PHX not PHO
            'SEA': 'Seattle Storm',
            'WAS': 'Washington Mystics',
            'GSV': 'Golden State Valkyries'
        }
        
        self.team_name_to_abbr = {v: k for k, v in self.bbref_teams.items()}
        self.team_abbrs = list(self.bbref_teams.keys())
    
    def discover_games_in_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Discover games by checking Basketball Reference URLs - FIXED VERSION
        
        CRITICAL FIXES:
        - Updated team abbreviations
        - Multiple URL format attempts
        - Better error handling
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        games = []
        current_date = start_dt
        
        total_days = (end_dt - start_dt).days + 1
        logger.info(f"Searching for games across {total_days} days")
        
        while current_date <= end_dt:
            day_number = (current_date - start_dt).days + 1
            date_str = current_date.strftime('%Y%m%d')
            
            logger.info(f"Day {day_number}/{total_days}: {current_date.strftime('%Y-%m-%d')}")
            
            # FIXED: Try multiple detection methods
            games_today = []
            
            # Method 1: Try Basketball Reference schedule page first
            schedule_games = self._check_schedule_page(current_date)
            if schedule_games:
                games_today.extend(schedule_games)
                logger.info(f"  Found {len(schedule_games)} games via schedule page")
            
            # Method 2: If no games found via schedule, try individual team URLs
            if not games_today:
                team_games = self._check_team_boxscores(current_date, date_str)
                games_today.extend(team_games)
                if team_games:
                    logger.info(f"  Found {len(team_games)} games via team boxscores")
            
            if games_today:
                games.extend(games_today)
                logger.info(f"  Total: {len(games_today)} games on {current_date.strftime('%Y-%m-%d')}")
            else:
                logger.debug(f"  No games found on {current_date.strftime('%Y-%m-%d')}")
            
            current_date += timedelta(days=1)
            time.sleep(min(self.delay, 5))  # Reduced delay for faster checking
        
        logger.info(f"Total games discovered: {len(games)}")
        return games
    
    def _check_schedule_page(self, game_date: datetime) -> List[Dict]:
        """
        Check Basketball Reference schedule page for games - NEW METHOD
        
        This is more reliable than individual boxscore URLs
        """
        try:
            # Try WNBA schedule page
            schedule_url = f"https://www.basketball-reference.com/wnba/years/{game_date.year}_games.html"
            
            logger.debug(f"    Checking schedule: {schedule_url}")
            response = self.session.get(schedule_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                return self._parse_schedule_page(soup, game_date)
            
        except Exception as e:
            logger.debug(f"    Schedule page check failed: {e}")
        
        return []
    
    def _parse_schedule_page(self, soup: BeautifulSoup, target_date: datetime) -> List[Dict]:
        """Parse schedule page to find games for target date"""
        games = []
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        try:
            # Look for schedule table
            schedule_table = soup.find('table', {'id': 'schedule'})
            if not schedule_table:
                return []
            
            tbody = schedule_table.find('tbody')
            if not tbody:
                return []
            
            for row in tbody.find_all('tr'):
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 4:
                        continue
                    
                    # Extract date from first cell
                    date_cell = cells[0].get_text(strip=True)
                    if not date_cell:
                        continue
                    
                    # Try to parse date
                    try:
                        # Basketball Reference uses format like "Wed, Oct 15, 2025"
                        parsed_date = datetime.strptime(date_cell, '%a, %b %d, %Y')
                    except ValueError:
                        try:
                            # Alternative format
                            parsed_date = datetime.strptime(date_cell, '%Y-%m-%d')
                        except ValueError:
                            continue
                    
                    # Check if this is our target date
                    if parsed_date.date() != target_date.date():
                        continue
                    
                    # Extract teams
                    visitor_cell = cells[2] if len(cells) > 2 else None
                    home_cell = cells[3] if len(cells) > 3 else None
                    
                    if not visitor_cell or not home_cell:
                        continue
                    
                    visitor_team = visitor_cell.get_text(strip=True)
                    home_team = home_cell.get_text(strip=True)
                    
                    # Convert team names to abbreviations
                    visitor_abbr = self._find_team_abbr_by_name(visitor_team)
                    home_abbr = self._find_team_abbr_by_name(home_team)
                    
                    if visitor_abbr and home_abbr:
                        game_info = {
                            'date': target_date_str,
                            'visitor': self.bbref_teams.get(visitor_abbr, visitor_team),
                            'home': self.bbref_teams.get(home_abbr, home_team),
                            'visitor_abbr': visitor_abbr,
                            'home_abbr': home_abbr,
                            'boxscore_url': f"https://www.basketball-reference.com/wnba/boxscores/{target_date.strftime('%Y%m%d')}0{home_abbr}.html"
                        }
                        games.append(game_info)
                        logger.info(f"    ✓ Found via schedule: {visitor_abbr} @ {home_abbr}")
                
                except Exception as e:
                    logger.debug(f"    Error parsing schedule row: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"    Error parsing schedule page: {e}")
        
        return games
    
    def _check_team_boxscores(self, game_date: datetime, date_str: str) -> List[Dict]:
        """
        Check individual team boxscore URLs - FALLBACK METHOD
        """
        games = []
        
        # Check each team as potential home team
        for home_abbr in self.team_abbrs:
            try:
                # FIXED: Try multiple URL formats
                url_formats = [
                    f"https://www.basketball-reference.com/wnba/boxscores/{date_str}0{home_abbr}.html",
                    f"https://www.basketball-reference.com/wnba/boxscores/{date_str}{home_abbr}.html",
                    f"https://www.basketball-reference.com/wnba/boxscores/{date_str}-{home_abbr}.html"
                ]
                
                for url_format in url_formats:
                    logger.debug(f"    Trying: {url_format}")
                    
                    response = self.session.get(url_format, timeout=8)
                    
                    if response.status_code == 200:
                        # Found a game! Extract team info
                        game_info = self._extract_game_info_from_response(response, game_date, url_format)
                        if game_info:
                            games.append(game_info)
                            logger.info(f"    ✓ Found via boxscore: {game_info['visitor_abbr']} @ {game_info['home_abbr']}")
                            break  # Found game for this team, move to next
                    
                    elif response.status_code == 404:
                        logger.debug(f"    404: {url_format}")
                    else:
                        logger.debug(f"    {response.status_code}: {url_format}")
                
                # Small delay between team checks
                time.sleep(1)
                
            except requests.RequestException as e:
                logger.debug(f"    Error checking {home_abbr}: {e}")
                continue
        
        return games
    
    def _extract_game_info_from_response(self, response, game_date: datetime, boxscore_url: str) -> Optional[Dict]:
        """Extract game information from Basketball Reference response"""
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Method 1: Extract from page title
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                logger.debug(f"    Page title: {title_text}")
                
                # Parse title like "Dallas Wings at Atlanta Dream, May 24, 2025 | Basketball-Reference.com"
                if ' at ' in title_text and 'Basketball-Reference.com' in title_text:
                    # Remove the Basketball-Reference suffix
                    game_part = title_text.split(' | ')[0]
                    
                    # Split on " at "
                    if ' at ' in game_part:
                        parts = game_part.split(' at ')
                        if len(parts) == 2:
                            visitor_name = parts[0].strip()
                            home_part = parts[1].strip()
                            
                            # Remove date from home part (e.g., "Atlanta Dream, May 24, 2025")
                            home_name = re.sub(r',.*$', '', home_part).strip()
                            
                            # Convert names to abbreviations
                            visitor_abbr = self._find_team_abbr_by_name(visitor_name)
                            home_abbr = self._find_team_abbr_by_name(home_name)
                            
                            if visitor_abbr and home_abbr:
                                return {
                                    'date': game_date.strftime('%Y-%m-%d'),
                                    'visitor': self.bbref_teams[visitor_abbr],
                                    'home': self.bbref_teams[home_abbr],
                                    'visitor_abbr': visitor_abbr,
                                    'home_abbr': home_abbr,
                                    'boxscore_url': boxscore_url
                                }
            
            # Method 2: Look for scorebox or other team indicators
            scorebox = soup.find('div', {'class': 'scorebox'})
            if scorebox:
                team_links = scorebox.find_all('a')
                team_names = []
                
                for link in team_links:
                    href = link.get('href', '')
                    if '/wnba/teams/' in href:
                        team_names.append(link.get_text(strip=True))
                
                if len(team_names) >= 2:
                    visitor_abbr = self._find_team_abbr_by_name(team_names[0])
                    home_abbr = self._find_team_abbr_by_name(team_names[1])
                    
                    if visitor_abbr and home_abbr:
                        return {
                            'date': game_date.strftime('%Y-%m-%d'),
                            'visitor': self.bbref_teams[visitor_abbr],
                            'home': self.bbref_teams[home_abbr],
                            'visitor_abbr': visitor_abbr,
                            'home_abbr': home_abbr,
                            'boxscore_url': boxscore_url
                        }
            
            logger.warning(f"Could not extract team info from {boxscore_url}")
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting game info: {e}")
            return None
    
    def _find_team_abbr_by_name(self, team_name: str) -> Optional[str]:
        """Find team abbreviation by name with improved fuzzy matching"""
        team_name = team_name.strip()
        
        # Direct lookup
        if team_name in self.team_name_to_abbr:
            return self.team_name_to_abbr[team_name]
        
        # FIXED: Updated mapping for current teams
        name_variations = {
            'Atlanta': 'ATL', 'Dream': 'ATL',
            'Chicago': 'CHI', 'Sky': 'CHI',
            'Connecticut': 'CONN', 'Sun': 'CONN',
            'Dallas': 'DAL', 'Wings': 'DAL',
            'Indiana': 'IND', 'Fever': 'IND',
            'Los Angeles': 'LAS', 'Sparks': 'LAS',
            'Las Vegas': 'LV', 'Aces': 'LV',
            'Minnesota': 'MIN', 'Lynx': 'MIN',
            'New York': 'NY', 'Liberty': 'NY',
            'Phoenix': 'PHX', 'Mercury': 'PHX',
            'Seattle': 'SEA', 'Storm': 'SEA',
            'Washington': 'WAS', 'Mystics': 'WAS'
        }
        
        # Check variations
        team_name_lower = team_name.lower()
        for variation, abbr in name_variations.items():
            if variation.lower() in team_name_lower:
                return abbr
        
        # Direct abbreviation check
        if team_name.upper() in self.bbref_teams:
            return team_name.upper()
        
        logger.debug(f"Could not find abbreviation for team: {team_name}")
        return None
    
    def scrape_game_boxscore(self, game: Dict) -> List[Dict]:
        """Scrape player statistics from a game boxscore"""
        try:
            logger.info(f"Scraping: {game['visitor']} @ {game['home']} ({game['date']})")
            
            response = self.session.get(game['boxscore_url'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            game_stats = []
            
            # Process both teams
            for team_name, team_abbr in [(game['visitor'], game['visitor_abbr']), 
                                       (game['home'], game['home_abbr'])]:
                team_stats = self._extract_team_stats(soup, team_name, team_abbr, 
                                                    game['date'], game['visitor'], game['home'])
                game_stats.extend(team_stats)
            
            logger.info(f"  Extracted {len(game_stats)} player records")
            return game_stats
            
        except Exception as e:
            logger.warning(f"Error scraping {game['boxscore_url']}: {e}")
            return []
    
    def _extract_team_stats(self, soup: BeautifulSoup, team_name: str, team_abbr: str,
                           date: str, visitor_team: str, home_team: str) -> List[Dict]:
        """Extract player statistics for a team"""
        team_stats = []
        
        possible_table_ids = [
            f"box-{team_abbr.upper()}-game-basic",
            f"box-{team_abbr.lower()}-game-basic",
            f"{team_abbr.upper()}_basic",
            f"{team_abbr.lower()}_basic",
            f"box_{team_abbr.upper()}_basic",
            f"box_{team_abbr.lower()}_basic"
        ]
        
        table = None
        for table_id in possible_table_ids:
            table = soup.find('table', {'id': table_id})
            if table:
                logger.debug(f"    Found table: {table_id}")
                break
        
        if not table:
            logger.warning(f"    No stats table found for {team_name}")
            return []
        
        # Determine home/away and opponent
        home_away = "Home" if team_name == home_team else "Away"
        opponent = home_team if team_name == visitor_team else visitor_team
        
        # Extract headers
        thead = table.find('thead')
        if not thead:
            logger.debug(f"    No thead found for {team_name}")
            return []

        headers = []
        header_rows = thead.find_all('tr')
        
        # Find the row with actual stat column names
        for row_idx, header_row in enumerate(header_rows):
            th_elements = header_row.find_all(['th', 'td'])
            
            row_headers = []
            for th in th_elements:
                header_text = th.get_text(strip=True)
                row_headers.append(header_text)
            
            # Check if this row has actual stat column names
            stat_indicators = ['MP', 'PTS', 'REB', 'AST', 'FG', 'MIN', 'Player']
            
            if any(indicator in row_headers for indicator in stat_indicators):
                headers = row_headers
                break
            elif len(row_headers) > 10:
                headers = row_headers
                break

        if not headers:
            logger.debug(f"    No valid header row found for {team_name}")
            return []
        
        # Create column mapping
        column_mapping = self._create_column_mapping(headers)
        
        # Extract player rows
        tbody = table.find('tbody')
        if not tbody:
            return []
        
        for row in tbody.find_all('tr'):
            if row.get('class') and 'thead' in row.get('class'):
                continue
                
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            
            # Get player name
            player_name = cells[0].get_text(strip=True)
            
            # Skip totals and empty rows
            if not player_name or player_name in ['Team Totals', 'Reserves', '', 'Starters']:
                continue
            
            # Extract stats
            player_stats = self._extract_player_stats(cells, column_mapping)
            
            # Add game context
            player_stats.update({
                'Date': date,
                'Player': player_name,
                'Team': team_name,
                'Opponent': opponent,
                'Home/Away': home_away
            })
            
            team_stats.append(player_stats)
        
        logger.debug(f"    Extracted {len(team_stats)} players for {team_name}")
        return team_stats
    
    def _create_column_mapping(self, headers: List[str]) -> Dict[str, int]:
        """Create mapping of stat names to column indices"""
        mapping = {}
        
        for i, header in enumerate(headers):
            if i == 0:  # Skip player name
                continue
                
            clean_header = header.strip()
            
            if clean_header:
                mapping[clean_header] = i
                self.all_stat_columns.add(clean_header)
        
        return mapping
    
    def _extract_player_stats(self, cells: List, column_mapping: Dict[str, int]) -> Dict[str, str]:
        """Extract statistics from player's row"""
        stats = {}
        
        for stat_name, col_idx in column_mapping.items():
            if col_idx < len(cells):
                value = cells[col_idx].get_text(strip=True)
                
                # Handle missing values
                if value in ['', '-', 'Did Not Play', 'Did Not Dress', '—']:
                    value = '0'
                elif stat_name == 'MP' and ':' in value:
                    # Convert MM:SS to decimal minutes
                    try:
                        parts = value.split(':')
                        minutes = int(parts[0]) + int(parts[1]) / 60
                        value = f"{minutes:.1f}"
                    except (ValueError, IndexError):
                        value = '0'
                
                stats[stat_name] = value
            else:
                stats[stat_name] = '0'
        
        return stats
    
    def scrape_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape player statistics for date range"""
        logger.info(f"Starting WNBA scraper: {start_date} to {end_date}")
        
        # Discover games
        games = self.discover_games_in_date_range(start_date, end_date)
        
        if not games:
            logger.warning("No games found for the specified date range")
            return pd.DataFrame()
        
        logger.info(f"Found {len(games)} games to scrape")
        
        # Print game list for confirmation
        print(f"\nFound games:")
        for i, game in enumerate(games, 1):
            print(f"  {i}. {game['date']}: {game['visitor']} @ {game['home']}")
        print()
        
        all_stats = []
        
        for i, game in enumerate(games, 1):
            logger.info(f"Processing game {i}/{len(games)}")
            
            game_stats = self.scrape_game_boxscore(game)
            if game_stats:
                all_stats.extend(game_stats)
            
            if i < len(games):
                time.sleep(self.delay)
        
        if not all_stats:
            logger.warning("No player statistics extracted")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_stats)
        
        # Organize columns
        final_columns = self.base_columns.copy()
        discovered_stats = sorted(list(self.all_stat_columns))
        final_columns.extend(discovered_stats)
        
        for col in final_columns:
            if col not in df.columns:
                df[col] = '0'
        
        df = df[final_columns]
        
        logger.info(f"Scraped {len(df)} player game records")
        logger.info(f"Statistics: {', '.join(discovered_stats)}")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV in wnba_game_data folder"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"wnba_stats_{timestamp}.csv"
        # Ensure the path is in wnba_game_data
        if not os.path.isabs(filename):
            filename = os.path.join('wnba_game_data', filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        return filename


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Scrape WNBA player statistics')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output CSV filename')
    parser.add_argument('--delay', type=float, default=10.0, help='Delay between requests')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
        datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD")
        return
    
    scraper = WNBAStatsScraper(delay=args.delay)
    
    try:
        df = scraper.scrape_date_range(args.start, args.end)
        
        if df.empty:
            print("No data found for the specified date range")
            return
        
        # Save results
        output_file = scraper.save_to_csv(df, args.output)
        
        # Print summary
        print(f"\n✓ Scraping completed successfully!")
        print(f"Records: {len(df)}")
        print(f"Output: {output_file}")
        
        # Show sample
        print(f"\nFirst 3 records:")
        print(df.head(3).to_string(index=False))
        
    except KeyboardInterrupt:
        print("\nScraping interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()