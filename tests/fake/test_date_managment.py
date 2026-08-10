"""Mocked date_managment tests -- no real network/API calls, safe to run anywhere.

Run with: pytest tests/fake
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from tish_video_sdk import date_managment
from tish_video_sdk.date_managment import DateFormatter, PublishingTracker, get_offset_date, to_rfc_datetime_string


@pytest.fixture(autouse=True)
def _no_ambient_month_names_path(monkeypatch):
    # tts.py's module-level load_dotenv() can leak a real DATE_MONTH_NAMES_PATH
    # from .env into os.environ once any other test module in this session
    # imports it -- DateFormatter() falls back to that env var, so without
    # this every bare DateFormatter() below would risk loading an unexpected
    # file instead of the built-in defaults its test expects.
    monkeypatch.delenv("DATE_MONTH_NAMES_PATH", raising=False)


class _FixedDatetime(datetime):
    """Stand-in for datetime with a frozen .today(), real .strptime()."""

    _fixed_today = datetime(2024, 3, 15)

    @classmethod
    def today(cls):
        return cls._fixed_today


class TestGetOffsetDate:
    def test_get_offset_date_defaults_to_today(self, monkeypatch):
        monkeypatch.setattr(date_managment, "datetime", _FixedDatetime)
        assert get_offset_date(5) == datetime(2024, 3, 20)

    def test_get_offset_date_negative_offset(self):
        base = datetime(2024, 3, 15)
        assert get_offset_date(-10, base_date=base) == datetime(2024, 3, 5)

    def test_get_offset_date_zero_offset_returns_base_date(self):
        base = datetime(2024, 1, 1)
        assert get_offset_date(0, base_date=base) == base


class TestToRfcDatetimeString:
    def test_aware_datetime_converted_to_utc(self):
        # Host-timezone-independent: a fixed +05:00 offset should always
        # shift back by 5 hours regardless of what machine this runs on.
        aware = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone(timedelta(hours=5)))
        assert to_rfc_datetime_string(aware) == "2024-03-15T05:30:00Z"

    def test_utc_datetime_unchanged(self):
        aware = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert to_rfc_datetime_string(aware) == "2024-03-15T10:30:00Z"

    def test_naive_datetime_treated_as_local_time_not_utc(self):
        # The old implementation just appended 'Z' to a naive datetime's
        # isoformat(), mislabeling local time as UTC. The fix delegates to
        # datetime.astimezone(), which treats a naive datetime as local
        # system time -- assert against that same stdlib conversion rather
        # than a hardcoded offset, since the correct answer depends on the
        # host's local timezone.
        naive = datetime(2024, 3, 15, 10, 30, 0)
        expected = naive.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert to_rfc_datetime_string(naive) == expected


class TestDateFormatter:
    def test_get_month_name_known_language(self):
        formatter = DateFormatter(default_language="fr")
        assert formatter.get_month_name(3) == "Mars"

    def test_get_month_name_explicit_language_overrides_default(self):
        formatter = DateFormatter(default_language="en")
        assert formatter.get_month_name(3, language_code="es") == "Marzo"

    def test_get_month_name_unsupported_language(self):
        formatter = DateFormatter()
        assert formatter.get_month_name(1, language_code="de") == "Unsupported language: de"

    def test_get_month_name_invalid_month_number(self):
        formatter = DateFormatter()
        assert formatter.get_month_name(13) == "Invalid month number: 13"

    def test_format_date_string(self):
        formatter = DateFormatter(default_language="en")
        assert formatter.format_date_string(datetime(2024, 3, 15)) == "15 March 2024"

    def test_format_date_string_unsupported_language_raises(self):
        formatter = DateFormatter()
        with pytest.raises(ValueError, match="Unsupported language"):
            formatter.format_date_string(datetime(2024, 3, 15), language_code="de")

    def test_format_date_string_with_offset(self, monkeypatch):
        monkeypatch.setattr(date_managment, "datetime", _FixedDatetime)
        formatter = DateFormatter(default_language="en")
        assert formatter.format_date_string_with_offset(offset_days=1) == "16 March 2024"

    def test_tamil_month_name(self):
        formatter = DateFormatter(default_language="ta")
        assert formatter.get_month_name(1) == "தை"

    def test_month_names_path_overrides_defaults(self, tmp_path):
        month_names_file = tmp_path / "month_names.json"
        month_names_file.write_text(
            json.dumps({"en": {"1": "Uno-ary"}}), encoding="utf-8"
        )
        formatter = DateFormatter(default_language="en", month_names_path=str(month_names_file))
        assert formatter.get_month_name(1) == "Uno-ary"
        # Languages not present in the override file are simply gone, same
        # as MusicManager's mood_packs_path -- a full replacement, not a merge.
        assert formatter.get_month_name(1, language_code="fr") == "Unsupported language: fr"

    def test_month_names_path_falls_back_to_env_var(self, tmp_path, monkeypatch):
        month_names_file = tmp_path / "month_names.json"
        month_names_file.write_text(
            json.dumps({"en": {"1": "Uno-ary"}}), encoding="utf-8"
        )
        monkeypatch.setenv("DATE_MONTH_NAMES_PATH", str(month_names_file))
        formatter = DateFormatter(default_language="en")
        assert formatter.get_month_name(1) == "Uno-ary"

    def test_no_override_uses_built_in_defaults(self):
        formatter = DateFormatter(default_language="en")
        assert formatter.month_names is date_managment.DEFAULT_MONTH_NAMES_BY_LANGUAGE


class TestComputePublishDatetime:
    def test_combines_date_hour_and_offset(self):
        publish_date = datetime(2024, 3, 15)
        result = date_managment.compute_publish_datetime(publish_date, hour_of_day=21, utc_offset_hours=2)
        assert result == datetime(2024, 3, 15, 23, 0)

    def test_negative_offset(self):
        publish_date = datetime(2024, 3, 15)
        result = date_managment.compute_publish_datetime(publish_date, hour_of_day=21, utc_offset_hours=-2)
        assert result == datetime(2024, 3, 15, 19, 0)

    def test_offset_crosses_into_next_day(self):
        publish_date = datetime(2024, 3, 15)
        result = date_managment.compute_publish_datetime(publish_date, hour_of_day=23, utc_offset_hours=2)
        assert result == datetime(2024, 3, 16, 1, 0)

    def test_minute_applied(self):
        publish_date = datetime(2024, 3, 15)
        result = date_managment.compute_publish_datetime(publish_date, hour_of_day=21, minute=5)
        assert result == datetime(2024, 3, 15, 21, 5)

    def test_existing_time_of_day_on_publish_date_is_discarded(self):
        publish_date = datetime(2024, 3, 15, 13, 45, 30)
        result = date_managment.compute_publish_datetime(publish_date, hour_of_day=21)
        assert result == datetime(2024, 3, 15, 21, 0)

    def test_default_offset_and_minute_are_zero(self):
        publish_date = datetime(2024, 3, 15)
        assert date_managment.compute_publish_datetime(publish_date, hour_of_day=9) == datetime(2024, 3, 15, 9, 0)


class TestPublishingTracker:
    def test_data_file_name_from_language_code(self):
        tracker = PublishingTracker(language_code="fr")
        assert tracker.data_file == "data_fr.txt"

    def test_data_file_defaults_to_published_date_file(self):
        tracker = PublishingTracker()
        assert tracker.data_file == date_managment.PUBLISHED_DATE_FILE

    def test_data_file_explicit_path_used_when_no_language_code(self, tmp_path):
        data_file = str(tmp_path / "custom.txt")
        tracker = PublishingTracker(data_file=data_file)
        assert tracker.data_file == data_file

    def test_language_code_wins_over_data_file_when_both_given(self):
        tracker = PublishingTracker(data_file="custom.txt", language_code="fr")
        assert tracker.data_file == "data_fr.txt"

    def test_data_directory_joins_default_filename(self, tmp_path):
        tracker = PublishingTracker(data_directory=str(tmp_path))
        assert tracker.data_file == str(tmp_path / date_managment.PUBLISHED_DATE_FILE)

    def test_data_directory_joins_language_code_filename(self, tmp_path):
        tracker = PublishingTracker(language_code="fr", data_directory=str(tmp_path))
        assert tracker.data_file == str(tmp_path / "data_fr.txt")

    def test_data_directory_ignored_when_explicit_data_file_given(self, tmp_path):
        data_file = str(tmp_path / "custom.txt")
        tracker = PublishingTracker(data_file=data_file, data_directory=str(tmp_path / "other"))
        assert tracker.data_file == data_file

    def test_get_last_published_date_missing_file_returns_none(self, tmp_path):
        tracker = PublishingTracker(data_file=str(tmp_path / "missing.txt"))
        assert tracker.get_last_published_date() is None

    def test_get_last_published_date_empty_file_returns_none(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.get_last_published_date() is None

    def test_get_last_published_date_invalid_format_returns_none(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("not-a-date", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.get_last_published_date() is None

    def test_get_last_published_date_valid_format(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("2024-03-15", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.get_last_published_date() == datetime(2024, 3, 15)

    def test_has_date_been_published_true_for_earlier_date(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("2024-03-15", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.has_date_been_published(datetime(2024, 3, 10)) is True

    def test_has_date_been_published_true_for_same_date(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("2024-03-15", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.has_date_been_published(datetime(2024, 3, 15)) is True

    def test_has_date_been_published_false_for_later_date(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("2024-03-15", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.has_date_been_published(datetime(2024, 3, 20)) is False

    def test_has_date_been_published_false_when_no_data(self, tmp_path):
        tracker = PublishingTracker(data_file=str(tmp_path / "missing.txt"))
        assert tracker.has_date_been_published(datetime(2024, 3, 20)) is False

    def test_has_date_been_published_str_valid(self, tmp_path):
        data_file = tmp_path / "data.txt"
        data_file.write_text("2024-03-15", encoding="utf-8")
        tracker = PublishingTracker(data_file=str(data_file))
        assert tracker.has_date_been_published_str("2024-03-10") is True

    def test_has_date_been_published_str_invalid_format_returns_false(self, tmp_path):
        tracker = PublishingTracker(data_file=str(tmp_path / "data.txt"))
        assert tracker.has_date_been_published_str("15/03/2024") is False

    def test_mark_date_as_published_writes_file(self, tmp_path):
        data_file = tmp_path / "data.txt"
        tracker = PublishingTracker(data_file=str(data_file))
        tracker.mark_date_as_published(datetime(2024, 3, 15))
        assert data_file.read_text(encoding="utf-8") == "2024-03-15"

    def test_mark_date_as_published_roundtrips_through_get_last_published_date(self, tmp_path):
        data_file = tmp_path / "data.txt"
        tracker = PublishingTracker(data_file=str(data_file))
        tracker.mark_date_as_published(datetime(2024, 3, 15))
        assert tracker.get_last_published_date() == datetime(2024, 3, 15)

    def test_mark_date_as_published_write_failure_does_not_raise(self, tmp_path):
        # tmp_path itself is a directory -- opening it for writing raises
        # IsADirectoryError (an OSError/IOError subclass), which
        # mark_date_as_published should catch and report rather than propagate.
        tracker = PublishingTracker(data_file=str(tmp_path))
        tracker.mark_date_as_published(datetime(2024, 3, 15))
