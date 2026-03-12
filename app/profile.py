"""
User profile management and zodiac sign calculation.

Handles:
- Sun sign computation from birth date
- Profile validation and storage
- Age calculation (bonus personalization)
"""

from datetime import datetime, date
from typing import Optional


# Zodiac date boundaries: (month, day) = last day of that sign
ZODIAC_BOUNDARIES = [
    ((1, 20), "Capricorn"),
    ((2, 19), "Aquarius"),
    ((3, 20), "Pisces"),
    ((4, 20), "Aries"),
    ((5, 21), "Taurus"),
    ((6, 21), "Gemini"),
    ((7, 22), "Cancer"),
    ((8, 23), "Leo"),
    ((9, 23), "Virgo"),
    ((10, 23), "Libra"),
    ((11, 22), "Scorpio"),
    ((12, 22), "Sagittarius"),
    ((12, 31), "Capricorn"),
]


def compute_zodiac(birth_date_str: str) -> str:
    """
    Compute the Western/Sun zodiac sign from a birth date string.

    Args:
        birth_date_str: Date in "YYYY-MM-DD" format (e.g. "1995-08-20")

    Returns:
        Zodiac sign name (e.g. "Leo")

    Raises:
        ValueError: If date format is invalid
    """
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Invalid date format: '{birth_date_str}'. Expected YYYY-MM-DD."
        )

    month = birth_date.month
    day = birth_date.day

    for (boundary_month, boundary_day), sign in ZODIAC_BOUNDARIES:
        if month < boundary_month or (month == boundary_month and day <= boundary_day):
            return sign

    return "Capricorn"  # fallback (Dec 23-31)


def compute_age(birth_date_str: str) -> int:
    """Compute current age from birth date string."""
    birth = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    today = date.today()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


class UserProfile:
    """Manages user profile data for a session."""

    def __init__(self, raw_profile: dict):
        self.name: str = raw_profile.get("name", "User")
        self.birth_date: str = raw_profile.get("birth_date", "")
        self.birth_time: Optional[str] = raw_profile.get("birth_time")
        self.birth_place: Optional[str] = raw_profile.get("birth_place")
        self.preferred_language: str = raw_profile.get("preferred_language", "en")

        # Computed fields
        self.zodiac: str = compute_zodiac(self.birth_date) if self.birth_date else "Unknown"
        self.age: Optional[int] = compute_age(self.birth_date) if self.birth_date else None

    def to_dict(self) -> dict:
        """Serialize for storage or prompt building."""
        return {
            "name": self.name,
            "birth_date": self.birth_date,
            "birth_time": self.birth_time,
            "birth_place": self.birth_place,
            "preferred_language": self.preferred_language,
            "zodiac": self.zodiac,
            "age": self.age,
        }

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors = []
        if not self.birth_date:
            errors.append("birth_date is required (format: YYYY-MM-DD)")
        if self.preferred_language not in ("en", "hi"):
            errors.append("preferred_language must be 'en' or 'hi'")
        return errors
