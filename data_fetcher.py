#!/usr/bin/env python3
"""
WNBA Data Fetcher — 2025‑07‑08 PATCH 5
=====================================
Fully working end‑to‑end scraper that:
* Finds every WNBA game in a date range on Basketball‑Reference (even when the
  schedule table is hidden in HTML comments).
* Extracts each team’s **basic box‑score** table (which is itself hidden in a
  comment) and saves individual player rows to CSV.

Run:
    python data_fetcher.py --start 2025-07-01 --end 2025-07-06 --debug
Output CSV lands in `./wnba_game_data/`.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from team_mapping import TeamNameMapper

###############################################################################
# 1.  HTTP helper
###############################################################################

class RequestManager:
    """Requests wrapper with polite rate‑limiting and debug logging."""

    def __init__(self, *, rate_limit: float = 2.0, timeout: int = 15, debug: bool = False):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.debug = debug
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def get(self, url: str) -> Optional[requests.Response]:
        """GET with rate‑limit and basic error handling."""
        delta = time.time() - self._last_call
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)
        if self.debug:
            logging.debug("GET %s", url)
        try:
            resp = self.session.get(url, timeout=self.timeout)
            self._last_call = time.time()
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # noqa: BLE001
            logging.warning("HTTP error %s -> %s", url, exc)
            return None

###############################################################################
# 2.  Team name standardisation
###############################################################################

# Remove TeamNameStandardizer class

###############################################################################
# 3.  Basketball‑Reference scraper
###############################################################################

class BasketballReference:
    BASE = "https://www.basketball-reference.com/wnba"

    def __init__(self, rm: RequestManager):
        self.rm = rm

    # ---------- schedule ----------
    def schedule(self, start: date, end: date) -> List[dict[str, Any]]:
        games: List[dict[str, Any]] = []
        res = self.rm.get(f"{self.BASE}/years/{start.year}_games.html")
        if not res:
            return games
        soup = BeautifulSoup(res.content, "lxml")
        table = soup.find("table", id="schedule")
        if not table:
            wrapper = soup.find("div", id="all_schedule")
            if wrapper:
                comment = next((c for c in wrapper.children if isinstance(c, Comment)), None)
                if comment:
                    table = BeautifulSoup(comment, "lxml").find("table", id="schedule")
        if not table:
            logging.warning("schedule table missing")
            return games
        for row in table.find("tbody").find_all("tr"):
            if "thead" in row.get("class", []):
                continue
            dcell = row.find(["th", "td"], attrs={"data-stat": "date_game"})
            vcell = row.find("td", attrs={"data-stat": "visitor_team_name"})
            hcell = row.find("td", attrs={"data-stat": "home_team_name"})
            if not (dcell and vcell and hcell):
                continue
            g_date = self._parse_date(dcell.get_text(strip=True), start.year)
            if not g_date or not (start <= g_date <= end):
                continue
            away = TeamNameMapper.to_abbreviation(vcell.get_text(strip=True))
            home = TeamNameMapper.to_abbreviation(hcell.get_text(strip=True))
            if not away or not home:
                logging.error(f"Unknown team(s) in schedule: away={vcell.get_text(strip=True)}, home={hcell.get_text(strip=True)}. Only real teams from team_mapping.py are allowed.")
                continue
            slug = TeamNameMapper.to_slug(home)
            dstr = g_date.strftime("%Y%m%d")
            games.append(
                {
                    "game_id": f"BR_{dstr}_{away}_{home}",
                    "date": g_date.isoformat(),
                    "away_team": away,
                    "home_team": home,
                    "box_url": f"{self.BASE}/boxscores/{dstr}0{slug}.html",
                }
            )
        logging.info("Basketball‑Reference: %d games", len(games))
        return games

    @staticmethod
    def _parse_date(txt: str, year: int) -> Optional[date]:
        txt = txt.split(",", 1)[1].strip() if "," in txt else txt
        m = re.match(r"([A-Za-z]{3})\s+(\d{1,2})", txt)
        if m:
            month = datetime.strptime(m.group(1), "%b").month
            return date(year, month, int(m.group(2)))
        m = re.match(r"(\d{1,2})/(\d{1,2})", txt)
        if m:
            return date(year, int(m.group(1)), int(m.group(2)))
        try:
            return date.fromisoformat(txt)
        except ValueError:
            return None

    # ---------- player‑tables ----------
    @staticmethod
    def _player_tables(soup: BeautifulSoup) -> List[BeautifulSoup]:
        pattern = re.compile(r"box-[a-z]{3}-game-basic", re.I)
        tables: List[BeautifulSoup] = list(soup.find_all("table", id=pattern))
        # Scan every HTML comment for hidden tables
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "box-" not in comment:
                continue
            frag = BeautifulSoup(comment, "lxml")
            tables.extend(frag.find_all("table", id=pattern))
        return tables

    # ---------- boxscore ----------
    def boxscore(self, game: dict[str, Any]) -> List[dict[str, Any]]:
        rows: List[dict[str, Any]] = []
        res = self.rm.get(game["box_url"])
        if not res:
            return rows
        soup = BeautifulSoup(res.content, "lxml")
        if "game has not been played" in soup.get_text().lower():
            return rows

        for tbl in self._player_tables(soup):
            # Get team abbreviation from table ID
            tbl_id = tbl.get("id", "")
            match = re.search(r"box-([a-z]{3})-game-basic", tbl_id, re.I)
            team_abbr_raw = match.group(1) if match else ""
            player_team = TeamNameMapper.to_abbreviation(team_abbr_raw)
            if not player_team:
                logging.error(f"Unknown team in boxscore: {team_abbr_raw}. Only real teams from team_mapping.py are allowed.")
                continue  # Skip unknown teams

            hdrs_raw = [th.get_text(strip=True) for th in tbl.find("thead").find_all("th")]
            hdrs = [h for h in hdrs_raw if h and h.lower() not in {"basic box score stats", "starters", "reserves"}]

            for tr in tbl.find("tbody").find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                player = cells[0].get_text(strip=True)
                if player.lower() in {"player", "totals", "team totals"} or len(player) < 2:
                    continue

                # Align headers and cells
                offset = max(0, len(hdrs) - len(cells))
                record = {"Player": player}
                for i, cell in enumerate(cells[1:], start=1):
                    header = hdrs[offset + i - 1] if offset + i - 1 < len(hdrs) else f"col_{i}"
                    record[header] = cell.get_text(strip=True)

                record.update(game)
                record["team"] = player_team
                record["opponent"] = game["home_team"] if player_team == game["away_team"] else game["away_team"]
                record["home_away"] = "A" if player_team == game["away_team"] else "H"
                rows.append(record)

        return rows


###############################################################################
# 4  Orchestrator helpers
###############################################################################

def scrape(start: str, end: str, *, delay: float = 2.0, debug: bool = False) -> pd.DataFrame:
    start_d: date = date.fromisoformat(start)
    end_d: date = date.fromisoformat(end)
    rm = RequestManager(rate_limit=delay, debug=debug)
    br = BasketballReference(rm)

    games = br.schedule(start_d, end_d)
    if not games:
        logging.warning("No games found in range %s – %s", start, end)
        return pd.DataFrame()

    all_rows: List[dict[str, Any]] = []
    for g in games:
        all_rows.extend(br.boxscore(g))

    if not all_rows:
        logging.warning("No player stats scraped — games may be too recent.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["scrape_ts"] = datetime.now().isoformat(timespec="seconds")
    return df

###############################################################################
# 5  CLI entry‑point
###############################################################################

def main() -> None:
    p = argparse.ArgumentParser(description="Scrape WNBA player box‑scores from Basketball‑Reference")
    p.add_argument("--start", required=True, help="start date YYYY‑MM‑DD")
    p.add_argument("--end", required=True, help="end date YYYY‑MM‑DD")
    p.add_argument("--output", help="custom CSV filename")
    p.add_argument("--delay", type=float, default=2.0, help="seconds between requests (default 2)")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    df = scrape(args.start, args.end, delay=args.delay, debug=args.debug)
    if df.empty:
        print("❌ No data scraped. Try a different date range or wait for box‑scores to be posted.")
        return

    out_dir = Path("wnba_game_data")
    out_dir.mkdir(exist_ok=True)
    fname = args.output or f"wnba_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"✅ {len(df)} rows saved → {out_dir/fname}")

###############################################################################
# 6. Wrapper class for integration with main_application
###############################################################################

class WNBAStatsScraper:
    """
    Wrapper around BasketballReference scraper for integration with the main pipeline.
    Provides clean methods for date-range scraping and saving to CSV.
    """

    def __init__(self, rate_limit: float = 2.0, debug: bool = False):
        self.rm = RequestManager(rate_limit=rate_limit, debug=debug)
        self.br = BasketballReference(self.rm)

    def scrape_date_range(self, start: str, end: str) -> pd.DataFrame:
        """
        Scrape WNBA player box score data for the given date range.

        Args:
            start (str): Start date in YYYY-MM-DD format
            end (str): End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: Combined DataFrame of all player stats
        """
        return scrape(start, end, delay=self.rm.rate_limit, debug=self.rm.debug)

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save a DataFrame to CSV in the `wnba_game_data` directory.

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename

        Returns:
            str: Path to the saved file
        """
        output_dir = Path("wnba_game_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        df.to_csv(path, index=False)
        return str(path)

if __name__ == "__main__":
    main()
