"""
team_mapping.py - Centralized WNBA Team Name/Abbreviation Mapping

Provides a single source of truth for all team name/abbreviation normalization and validation.
"""
from typing import Optional, Set, List, Dict, Tuple, Any
import logging

class TeamNameMapper:
    # Canonical mapping (based on data_fetcher.py, with harmonization for downstream use)
    _NAME_TO_ABBR = {
        "atlanta dream": "ATL", "dream": "ATL", "atl": "ATL",
        "chicago sky": "CHI", "sky": "CHI", "chi": "CHI",
        "connecticut sun": "CON", "sun": "CON", "conn": "CON",
        "las vegas aces": "LVA", "aces": "LVA", "vegas": "LVA", "lv": "LVA",
        "new york liberty": "NYL", "liberty": "NYL", "nyl": "NYL", "ny": "NYL",
        "phoenix mercury": "PHO", "mercury": "PHO", "pho": "PHO", "phx": "PHO",
        "dallas wings": "DAL", "wings": "DAL", "dal": "DAL",
        "indiana fever": "IND", "fever": "IND", "ind": "IND",
        "los angeles sparks": "LAS", "sparks": "LAS", "las": "LAS", "la sparks": "LAS",
        "minnesota lynx": "MIN", "lynx": "MIN", "min": "MIN",
        "seattle storm": "SEA", "storm": "SEA", "sea": "SEA",
        "washington mystics": "WAS", "mystics": "WAS", "was": "WAS",
        "golden state valkyries": "GSV", "valkyries": "GSV", "gsv": "GSV",
        # Add all valid uppercase abbreviations as keys mapping to themselves
        "ATL": "ATL", "CHI": "CHI", "CON": "CON", "DAL": "DAL", "IND": "IND", "LAS": "LAS", "LVA": "LVA", "MIN": "MIN", "NYL": "NYL", "PHO": "PHO", "SEA": "SEA", "WAS": "WAS", "GSV": "GSV",
        # Add all valid lowercase abbreviations as keys mapping to uppercase
        "atl": "ATL", "chi": "CHI", "con": "CON", "dal": "DAL", "ind": "IND", "las": "LAS", "lva": "LVA", "min": "MIN", "nyl": "NYL", "pho": "PHO", "sea": "SEA", "was": "WAS", "gsv": "GSV",
    }

    _VALID_ABBRS = {
        "ATL","CHI","CON","DAL","IND","LAS","LVA","MIN","NYL","PHO","SEA","WAS","GSV"
    }

    @classmethod
    def to_abbreviation(cls, name_or_abbr: str) -> Optional[str]:
        if not name_or_abbr:
            return None
        key = name_or_abbr.strip().lower()
        abbr = cls._NAME_TO_ABBR.get(key)
        if abbr:
            return abbr
        # Try partial/word-based matching
        for word in key.split():
            for k, v in cls._NAME_TO_ABBR.items():
                if word in k.split():
                    return v
        logging.warning(f"Unknown team: {name_or_abbr}, using as-is")
        return name_or_abbr.upper() if len(name_or_abbr) <= 4 else None

    @classmethod
    def is_valid_abbreviation(cls, abbr: str) -> bool:
        return abbr.upper() in cls._VALID_ABBRS

    @classmethod
    def all_abbreviations(cls) -> Set[str]:
        return set(cls._VALID_ABBRS)

    @classmethod
    def to_slug(cls, abbr: str) -> str:
        """Return the canonical slug for a team abbreviation (for URLs, etc). Always returns a string."""
        abbr_val = cls.to_abbreviation(abbr)
        if abbr_val:
            return abbr_val
        return abbr.upper() if abbr else "" 