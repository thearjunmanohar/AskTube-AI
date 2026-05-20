"""
transcript_fetcher.py
---------------------
Handles fetching a YouTube transcript and saving it as transcript.txt.
"""

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Parse the YouTube video ID from a full URL or bare ID."""
    if "v=" in url:
        vid = url.split("v=")[1]
        vid = vid.split("&")[0]
        return vid
    # Assume it's already a bare video ID
    return url.strip()


def list_available_transcripts(video_id: str) -> list[dict]:
    """Return a list of available transcript metadata for a video."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    return [
        {
            "language": t.language,
            "language_code": t.language_code,
            "is_generated": t.is_generated,
        }
        for t in transcript_list
    ]


def fetch_transcript(video_id: str, language_codes: list[str] | None = None) -> str:
    """
    Fetch transcript text for the given video ID.

    Parameters
    ----------
    video_id : str
        YouTube video ID.
    language_codes : list[str], optional
        Preferred language codes in priority order.
        Defaults to ['en-US', 'en'].

    Returns
    -------
    str
        The full transcript as a single string.
    """
    if language_codes is None:
        language_codes = ["en-US", "en"]

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    transcript = transcript_list.find_transcript(language_codes).fetch()
    return "".join(segment.text for segment in transcript)


def save_transcript(text: str, path: str = "transcript.txt") -> None:
    """Write transcript text to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
