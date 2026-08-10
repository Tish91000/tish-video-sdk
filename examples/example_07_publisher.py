"""Example: publish videos with Publisher (YouTube, Instagram).

Demonstrates:
- Configuring Publisher from env vars (see ../.env.template) -- only
  platforms with credentials given end up in publisher.available().
- Read-only YouTube operations (get_trending_videos, search_public_videos):
  safe to call for real, gated on YOUTUBE_CLIENT_SECRET_FILEPATH being set.
  First call opens a browser for one-time OAuth consent, then caches a
  token next to the credential file.
- publish_video()/publish_to_youtube()/publish_to_instagram(): NOT called
  here. Publishing uploads a real, publicly-visible video to a real
  channel/account -- unlike a search call, it can't be un-done by re-running
  the example, so this stays commented-out sample code rather than something
  that runs by default. Uncomment deliberately, with real arguments, when
  you actually want to publish something.
"""
import os

from dotenv import load_dotenv

from tish_video_sdk.publisher import Publisher

load_dotenv()


def main():
    publisher = Publisher(
        youtube_client_secret_filepath=os.getenv("YOUTUBE_CLIENT_SECRET_FILEPATH", ""),
        instagram_username=os.getenv("INSTAGRAM_USERNAME", ""),
        instagram_password=os.getenv("INSTAGRAM_PASSWORD", ""),
        instagram_session_file=os.getenv("INSTAGRAM_SESSION_FILE", ""),
    )
    print(f"Configured platforms: {publisher.available()}")

    print("--- YouTube trending videos (read-only, real API call) ---")
    if publisher.youtube:
        trends = publisher.youtube.get_trending_videos(region_code="US", max_results=3)
        for video in trends:
            print(f"  {video['title']} ({video['viewCount']} views)")
    else:
        print("No YOUTUBE_CLIENT_SECRET_FILEPATH configured; skipping.")

    print("--- YouTube public search (read-only, real API call) ---")
    if publisher.youtube:
        results = publisher.youtube.search_public_videos("mountain sunrise timelapse", max_results=3)
        for video in results:
            print(f"  {video['title']} ({video['id']})")
    else:
        print("No YOUTUBE_CLIENT_SECRET_FILEPATH configured; skipping.")

    print("--- Publishing (not run by default -- see module docstring) ---")
    print("  publisher.publish_to_youtube('video.mp4', title='...', description='...', tags=[...])")
    print("  publisher.publish_to_instagram('video.mp4', caption='...')")
    # if publisher.youtube:
    #     publisher.publish_to_youtube(
    #         "video.mp4", title="My Video", description="Description", tags=["tag1", "tag2"],
    #         privacy_status="private",  # keep private for a first real test
    #     )
    # if publisher.instagram:
    #     publisher.publish_to_instagram("video.mp4", caption="My caption")


if __name__ == "__main__":
    main()
