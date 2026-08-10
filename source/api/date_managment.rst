Date managment
==============

.. currentmodule:: tish_video_sdk.date_managment

Date-offset formatting and a small publish-date tracker, carried over from the
original standalone script rather than fully reworked for this release — see
:func:`get_offset_date`, :func:`to_rfc_datetime_string`, :class:`DateFormatter`,
and :class:`PublishingTracker` below. Error reporting (``print()``) and plain
file writes are left matching the rest of this SDK's existing style rather
than reworked here.

:class:`~tish_video_sdk.publisher.Publisher` is this module's one internal
consumer: :meth:`Publisher.next_unpublished_date()
<tish_video_sdk.publisher.Publisher.next_unpublished_date>` and
:meth:`Publisher.mark_published() <tish_video_sdk.publisher.Publisher.mark_published>`
wrap :class:`PublishingTracker` internally, and
:meth:`Publisher.publish_video() <tish_video_sdk.publisher.Publisher.publish_video>`
calls :func:`compute_publish_datetime` internally for its ``content_date``/
``hour_of_day``/``utc_offset_hours`` params — see :doc:`publisher`. A project
can still use every symbol here directly if useful for something outside
that flow.

:func:`get_offset_date`, :func:`to_rfc_datetime_string`, and
:func:`compute_publish_datetime` are stateless date-offset, RFC 3339
formatting, and schedule-datetime helpers. ``to_rfc_datetime_string``
converts to true UTC — a naive ``datetime`` (e.g. from ``datetime.today()``)
is treated as local system time and converted, not just labeled ``Z`` as-is.
``compute_publish_datetime`` combines a date with an hour-of-day and a UTC
offset into the datetime a video should go live at; the hour and offset are
always caller-supplied — a content channel's own publishing-schedule policy,
not domain-neutral content the SDK should default the way
:class:`DateFormatter`'s month names are.
:class:`DateFormatter` formats a date as a localized string
(``"15 Mars 2024"``) using its own built-in month-name tables — currently
English, French, Spanish, Italian, and Tamil. The table is generic,
domain-neutral built-in content, not fixed SDK behavior — same override
pattern as :class:`~tish_video_sdk.music.MusicManager`'s moods
(``MUSIC_MOODS_PATH``): point ``month_names_path``/``DATE_MONTH_NAMES_PATH``
at a JSON file (see ``configuration/month_names.example.json`` and :doc:`../installation`)
to add or replace languages. :class:`PublishingTracker` reads and writes the
most recently published date to a small text file (one date, ``YYYY-MM-DD``),
so a caller can check whether a given date has already been published before
publishing it again — ``data_directory`` places that file in a chosen
directory instead of wherever the process happens to be run from. A
consuming app normally reaches this through
:class:`~tish_video_sdk.publisher.Publisher` instead of constructing
``PublishingTracker`` directly (see :doc:`publisher`).

.. automodule:: tish_video_sdk.date_managment
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from datetime import datetime

   from tish_video_sdk.date_managment import (
       DateFormatter, PublishingTracker, compute_publish_datetime, get_offset_date,
   )

   # Format a date a week from now:
   next_week = get_offset_date(7)
   formatter = DateFormatter(default_language="fr")
   print(formatter.format_date_string(next_week))  # "22 Mars 2024"

   # Compute a schedule datetime from a content date + hour-of-day + UTC offset:
   scheduled = compute_publish_datetime(next_week, hour_of_day=21, utc_offset_hours=-2)

   # Track and check publish dates (normally reached via Publisher instead --
   # see publisher.py's next_unpublished_date()/mark_published()):
   tracker = PublishingTracker(data_directory="./state")
   if not tracker.has_date_been_published(datetime.today()):
       tracker.mark_date_as_published(datetime.today())
