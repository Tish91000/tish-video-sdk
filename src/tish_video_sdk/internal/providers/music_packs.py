"""music_packs: music-source provider client(s) used by MusicManager (see
music.py). Currently just a thin Jamendo API client
(https://api.jamendo.com/v3.0/), used as an optional remote fallback when a
mood has no local track configured -- named for the provider group rather
than jamendo specifically so another remote music source can land here
later without a rename. Supporting module, not a feature entry point --
consumers configure this via MusicManager(jamendo_client_id=...) /
JAMENDO_CLIENT_ID, not by importing it directly.

NOTE: parameter names and response shape reflect Jamendo API v3.0 as
documented at https://devportal.jamendo.com/ at the time this was written.
Third-party API contracts drift -- if searches start failing, check the live
docs before assuming a bug here.
"""
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

API_TRACKS_URL = "https://api.jamendo.com/v3.0/tracks/"


@dataclass
class JamendoTrack:
    id: str
    name: str
    artist_name: str
    duration: int  # seconds
    audio_download_url: str
    license_ccurl: str


def search_track(mood_query: str, min_duration_s: int, client_id: str,
                  exclude_non_commercial: bool = True, instrumental_only: bool = True,
                  timeout: float = 10.0) -> Optional[JamendoTrack]:
    """Search Jamendo for one track matching mood_query (used as a fuzzy tag
    search), at least min_duration_s long. Returns None on no match or on
    any request failure -- callers should fall back to local/default music
    rather than propagate a network error.
    """
    if not client_id:
        return None

    # NOTE: two request params were tried here and dropped after observing
    # them zero out results against the live API in combination with other
    # params (not fully explained -- see the module docstring above):
    # "audiodownload_allowed": "true", and "durationbetween": "<min>_10000"
    # (e.g. ccnc=false + durationbetween=6_10000 returned 0 results for the
    # "meditative" tag, while ccnc=false alone returned 5). A usable URL is
    # instead guaranteed by filtering the response's own audiodownload/audio
    # field below, and min_duration_s becomes a client-side preference
    # rather than a server-side filter -- harmless either way, since
    # MusicManager already loops a too-short track to cover what it needs.
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": "10",
        "order": "popularity_total",
        "fuzzytags": mood_query,
    }
    if exclude_non_commercial:
        params["ccnc"] = "false"
    if instrumental_only:
        # Background music mixed under narration shouldn't carry its own
        # competing vocals/lyrics.
        params["vocalinstrumental"] = "instrumental"

    url = API_TRACKS_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"Jamendo search failed: {e}")
        return None

    candidates = [t for t in payload.get("results", []) if t.get("audiodownload") or t.get("audio")]
    if not candidates:
        return None

    # Prefer a candidate that's already long enough over one that would need
    # looping, when there's a choice -- but any candidate is usable.
    long_enough = [t for t in candidates if int(t.get("duration", 0)) >= max(min_duration_s, 0)]
    track = long_enough[0] if long_enough else candidates[0]

    return JamendoTrack(
        id=str(track.get("id", "")),
        name=track.get("name", "unknown"),
        artist_name=track.get("artist_name", "unknown"),
        duration=int(track.get("duration", 0)),
        audio_download_url=track.get("audiodownload") or track.get("audio"),
        license_ccurl=track.get("license_ccurl", ""),
    )


def download_track(track: JamendoTrack, dest_path: str, timeout: float = 30.0) -> None:
    """Download track's audio to dest_path. Raises on failure -- callers
    decide how to handle a failed download (e.g. falling back to local
    music)."""
    with urllib.request.urlopen(track.audio_download_url, timeout=timeout) as response:
        data = response.read()
    with open(dest_path, "wb") as f:
        f.write(data)
