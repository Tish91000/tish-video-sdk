"""date_managment: date-offset/localized-formatting helpers and a small
publish-date tracker (see CONTEXT.md).

Ships mostly in its original form -- below the SDK's usual quality bar in a
couple of ways called out below, but not broken. Error reporting (print()
rather than raised/logged) and plain, non-atomic file writes are left as-is
here since that's how every other module in this SDK already handles both;
changing only this module would be inconsistent rather than better. A
handful of issues genuinely isolated to this module were fixed in place
instead of deferred -- see get_offset_date/to_rfc_datetime_string,
PublishingTracker's data_directory, and Optional[...] type hints below.
Three loosely related pieces, kept as one module rather than split further:

- get_offset_date/to_rfc_datetime_string/compute_publish_datetime: stateless
  date-offset, RFC 3339 formatting, and schedule-datetime helpers -- plain
  module functions rather than a static-method class, matching the SDK's
  convention elsewhere (_load_mood_config, _build_google_service) for
  stateless helpers with no reason to be a class. compute_publish_datetime is
  Publisher.publish_video's internal building block for its content_date/
  hour_of_day/utc_offset_hours params (see publisher.py) -- the hour and
  offset are always caller-supplied, a channel's own publishing-schedule
  policy rather than domain-neutral content the SDK should default (unlike
  DateFormatter's month names below).
- DateFormatter: per-language month-name lookup and localized date-string
  formatting. The month-name table is generic, domain-neutral built-in
  content, not fixed SDK behavior -- same override pattern as MusicManager's
  moods (MUSIC_MOODS_PATH): point month_names_path/DATE_MONTH_NAMES_PATH at
  a JSON file (see configuration/month_names.example.json) to add or replace languages.
- PublishingTracker: reads/writes the most recently published date to a
  small text file, so a scheduling loop can tell whether a given date has
  already been published. Publisher.next_unpublished_date()/mark_published()
  are the only intended entry points for consuming apps -- they wrap a
  PublishingTracker internally (see publisher.py, ADR 0012) rather than
  application code constructing one directly, so a date-selection loop stays
  in one place regardless of which platform(s) Publisher ends up publishing
  to.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

# --- Constants ---

# File to store the last published date in ISO format (YYYY-MM-DD)
PUBLISHED_DATE_FILE = "data.txt"

# Built-in month-name tables, used unless month_names_path/DATE_MONTH_NAMES_PATH
# overrides them (see _load_month_names below). Keys are language codes (e.g.,
# 'en', 'fr') and values are dictionaries mapping month numbers (as strings)
# to their respective names.
DEFAULT_MONTH_NAMES_BY_LANGUAGE = {
    'fr': {
        '1': 'Janvier', '2': 'Février', '3': 'Mars', '4': 'Avril',
        '5': 'Mai', '6': 'Juin', '7': 'Juillet', '8': 'Août',
        '9': 'Septembre', '10': 'Octobre', '11': 'Novembre', '12': 'Décembre'
    },
    'en': {
        '1': 'January', '2': 'February', '3': 'March', '4': 'April',
        '5': 'May', '6': 'June', '7': 'July', '8': 'August',
        '9': 'September', '10': 'October', '11': 'November', '12': 'December'
    },
    'es': {
        '1': 'Enero', '2': 'Febrero', '3': 'Marzo', '4': 'Abril',
        '5': 'Mayo', '6': 'Junio', '7': 'Julio', '8': 'Agosto',
        '9': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre'
    },
    'it': {
        '1': 'Gennaio', '2': 'Febbraio', '3': 'Marzo', '4': 'Aprile',
        '5': 'Maggio', '6': 'Giugno', '7': 'Luglio', '8': 'Agosto',
        '9': 'Settembre', '10': 'Ottobre', '11': 'Novembre', '12': 'Dicembre'
    },
    'ta': {
        '1': 'தை', '2': 'வெள்ளை', '3': 'மார்ச்', '4': 'ஏப்ரல்',
        '5': 'மே', '6': 'ஜூன்', '7': 'ஜூலை', '8': 'ஆகஸ்ட்',
        '9': 'செப்டம்பர்', '10': 'அக்டோபர்', '11': 'நவம்பர்', '12': 'டிசம்பர்'
    }
}


def _load_month_names(month_names_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Load per-language month-name tables from month_names_path (see
    configuration/month_names.example.json for the shape), falling back to the built-in
    defaults when unset. month_names_path itself falls back to the
    DATE_MONTH_NAMES_PATH env var, then to the defaults alone when neither
    is set."""
    path = month_names_path or os.getenv("DATE_MONTH_NAMES_PATH")
    if not path:
        return DEFAULT_MONTH_NAMES_BY_LANGUAGE
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_offset_date(offset_days: int, base_date: Optional[datetime] = None) -> datetime:
    """
    Calculates a date by adding or subtracting a specified number of days
    from a base date.

    Args:
        offset_days (int): The number of days to add (positive) or subtract (negative).
        base_date (datetime, optional): The starting date for the calculation.
                                        Defaults to datetime.today() if None.

    Returns:
        datetime: A datetime object representing the date with the requested offset.
    """
    if base_date is None:
        base_date = datetime.today()
    return base_date + timedelta(days=offset_days)


def to_rfc_datetime_string(date_obj: datetime) -> str:
    """
    Converts a datetime object to an RFC 3339 / ISO 8601 formatted string in UTC.

    A naive date_obj (no tzinfo, e.g. from datetime.today()) is treated as
    local system time and converted to UTC. An aware date_obj is converted
    to UTC directly.

    Args:
        date_obj (datetime): The datetime object to convert.

    Returns:
        str: The RFC 3339 / ISO 8601 formatted string with 'Z' indicating UTC.
    """
    if date_obj.tzinfo is None:
        date_obj = date_obj.astimezone()
    return date_obj.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def compute_publish_datetime(publish_date: datetime, hour_of_day: int, utc_offset_hours: int = 0,
                              minute: int = 0) -> datetime:
    """
    Combines a date with an hour-of-day and a UTC offset into the datetime a
    video should go live at: midnight on publish_date, plus hour_of_day +
    utc_offset_hours (and minute).

    hour_of_day and utc_offset_hours are always caller-supplied -- a content
    channel's own publishing-schedule policy (when *this* audience wants
    content), not domain-neutral content the SDK should default the way
    DateFormatter's month names are.

    Args:
        publish_date (datetime): The date to publish on. Only its
            year/month/day are used; any existing time-of-day is discarded.
        hour_of_day (int): The caller's base publish hour.
        utc_offset_hours (int, optional): Added to hour_of_day (e.g. a
            per-language timezone adjustment the caller maintains). Defaults to 0.
        minute (int, optional): Minute of the hour. Defaults to 0.

    Returns:
        datetime: publish_date at hour (hour_of_day + utc_offset_hours):minute.
    """
    base = publish_date.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(hours=hour_of_day + utc_offset_hours, minutes=minute)


class DateFormatter:
    """
    A class for formatting dates in various languages and formats.
    """

    def __init__(self, default_language: str = 'en', month_names_path: Optional[str] = None):
        """
        Initialize the DateFormatter.

        Args:
            default_language (str): Default language code for formatting.
            month_names_path (str, optional): Path to a JSON file overriding
                the built-in month-name tables (see
                configuration/month_names.example.json for the shape). Falls
                back to the DATE_MONTH_NAMES_PATH env var, then to the
                built-in defaults, when unset.
        """
        self.default_language = default_language
        self.month_names = _load_month_names(month_names_path)

    def get_month_name(self, month_number: int, language_code: Optional[str] = None) -> str:
        """
        Retrieves the name of a month for a given language.

        Args:
            month_number (int): The month number (1-12).
            language_code (str, optional): The language code. Uses default if None.

        Returns:
            str: The name of the month in the specified language, or an error message
                 if the language or month number is invalid.
        """
        lang_code = language_code or self.default_language
        month_names = self.month_names.get(lang_code)
        if month_names:
            return month_names.get(str(month_number), f'Invalid month number: {month_number}')
        return f'Unsupported language: {lang_code}'

    def format_date_string(self, date_obj: datetime, language_code: Optional[str] = None) -> str:
        """
        Formats a datetime object into a localized date string.

        Args:
            date_obj (datetime): The datetime object to format.
            language_code (str, optional): The language code for formatting. Uses default if None.

        Returns:
            str: The formatted date string.

        Raises:
            ValueError: If the language code is not supported.
        """
        lang_code = language_code or self.default_language
        month_name = self.get_month_name(date_obj.month, lang_code)
        if month_name.startswith('Unsupported language') or month_name.startswith('Invalid month number'):
            raise ValueError(month_name)

        return f"{date_obj.day} {month_name} {date_obj.year}"

    def format_date_string_with_offset(self, language_code: Optional[str] = None, offset_days: int = 0) -> str:
        """
        Formats the current date with an offset into a localized date string.

        Args:
            language_code (str, optional): The language code for formatting. Uses default if None.
            offset_days (int, optional): The number of days to offset from today.

        Returns:
            str: The formatted date string with the specified offset.
        """
        date_obj = get_offset_date(offset_days, datetime.today())
        return self.format_date_string(date_obj, language_code)


class PublishingTracker:
    """
    A class for tracking publishing dates and states.
    """

    def __init__(self, data_file: Optional[str] = None, language_code: Optional[str] = None,
                 data_directory: Optional[str] = None):
        """
        Initialize the PublishingTracker.

        Args:
            data_file (str, optional): Path to the file storing published dates.
                Ignored when language_code is also given (language_code wins,
                same precedence as before this parameter existed).
            language_code (str, optional): Language code to determine the data file name.
                                           If provided, the filename is 'data_{language_code}.txt'.
            data_directory (str, optional): Directory the language_code-derived
                or default filename is placed in. Ignored when data_file is
                given (data_file is used as-is, absolute or relative). Unset,
                filenames are left bare, relative to the current working
                directory -- the previous (implicit) behavior.
        """
        if language_code:
            filename = f"data_{language_code}.txt"
            self.data_file = os.path.join(data_directory, filename) if data_directory else filename
        elif data_file:
            self.data_file = data_file
        else:
            self.data_file = os.path.join(data_directory, PUBLISHED_DATE_FILE) if data_directory else PUBLISHED_DATE_FILE

    def get_last_published_date(self) -> Optional[datetime]:
        """
        Reads the last published date from the data file.

        Returns:
            datetime | None: The last published date as a datetime object,
                             or None if the file does not exist, is empty,
                             or contains an invalid date format.
        """
        if not os.path.exists(self.data_file):
            return None
        try:
            with open(self.data_file, "r", encoding="utf-8") as file:
                date_string = file.read().strip()
                if not date_string:
                    return None
                return datetime.strptime(date_string, "%Y-%m-%d")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading or parsing last published date: {e}")
            return None

    def has_date_been_published(self, date_obj: datetime) -> bool:
        """
        Checks if a given date has already been marked as published.

        Args:
            date_obj (datetime): The date to check.

        Returns:
            bool: True if the date has been published, False otherwise.
        """
        published_date = self.get_last_published_date()
        if published_date:
            return date_obj.date() <= published_date.date()
        return False

    def has_date_been_published_str(self, date_str: str) -> bool:
        """
        Checks if a given date string has already been marked as published.

        Args:
            date_str (str): The date string to check, expected in ISO format (YYYY-MM-DD).

        Returns:
            bool: True if the date has been published, False otherwise.
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return self.has_date_been_published(date_obj)
        except ValueError as e:
            print(f"Invalid date string format: {e}")
            return False

    def mark_date_as_published(self, date_obj: datetime):
        """
        Writes the given date to the data file, marking it as published.

        Args:
            date_obj (datetime): The date to mark as published.
        """
        try:
            with open(self.data_file, "w", encoding="utf-8") as file:
                file.write(date_obj.strftime("%Y-%m-%d"))
        except IOError as e:
            print(f"Error writing published date to file: {e}")
