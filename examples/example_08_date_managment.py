"""Example: date-offset formatting and publish-date tracking with
date_managment.

Entirely local -- no external services or credentials involved. Demonstrates:
- get_offset_date/to_rfc_datetime_string: computing an offset date and
  formatting it as an RFC 3339 UTC string.
- DateFormatter: localized date strings in a few of the supported languages.
- compute_publish_datetime: combining a content date with an hour-of-day and
  a caller-supplied UTC offset into a schedule datetime -- the building block
  behind Publisher.publish_video's content_date/hour_of_day/utc_offset_hours
  params (see example_07_publisher.py), usable standalone too.
- PublishingTracker: marking a date as published, then checking later dates
  against it -- the kind of check a daily-content scheduling loop would run
  before deciding whether to publish today's item. data_directory keeps its
  tracking file next to this example's other output instead of wherever the
  process happens to be run from. A consuming app normally reaches this via
  Publisher.next_unpublished_date()/mark_published() instead of constructing
  PublishingTracker directly (see ADR 0012) -- shown here standalone since
  this module has no Publisher of its own to demonstrate with.
"""
import os
import sys
from datetime import datetime, timedelta

from tish_video_sdk.date_managment import (
    DateFormatter, PublishingTracker, compute_publish_datetime, get_offset_date, to_rfc_datetime_string,
)


def main():
    # Some of DateFormatter's month names (e.g. Tamil) fall outside what a
    # default Windows console's legacy codepage can print -- widen stdout to
    # UTF-8 so this example runs the same on every platform.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    print("--- get_offset_date / to_rfc_datetime_string ---")
    today = datetime.today()
    next_week = get_offset_date(7, base_date=today)
    print(f"Today:      {today.date()}")
    print(f"+7 days:    {next_week.date()}")
    print(f"RFC 3339:   {to_rfc_datetime_string(next_week)}")

    print("--- DateFormatter ---")
    for language_code in ("en", "fr", "es", "ta"):
        formatter = DateFormatter(default_language=language_code)
        print(f"{language_code}: {formatter.format_date_string(today)}")

    print("--- compute_publish_datetime ---")
    scheduled = compute_publish_datetime(today, hour_of_day=21, utc_offset_hours=-2, minute=5)
    print(f"Content date {today.date()}, hour_of_day=21, utc_offset_hours=-2 -> {scheduled}")

    print("--- PublishingTracker ---")
    tracker = PublishingTracker(data_directory=output_dir)

    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    tracker.mark_date_as_published(today)
    print(f"Marked {today.date()} as published (see {tracker.data_file}).")
    print(f"Already published as of yesterday ({yesterday.date()})? {tracker.has_date_been_published(yesterday)}")
    print(f"Already published as of tomorrow ({tomorrow.date()})? {tracker.has_date_been_published(tomorrow)}")


if __name__ == "__main__":
    main()
