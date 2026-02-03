"""YouTube URL Processor - Extracts video transcripts and formats as Markdown.

This processor handles YouTube video ingestion by:
1. Extracting video ID from various YouTube URL formats
2. Fetching transcript (captions/subtitles) via youtube-transcript-api
3. Formatting transcript with timestamps as Markdown for downstream chunking

Usage:
    processor = YouTubeProcessor()
    markdown = await processor.process("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

Supported URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID

Note:
    This processor inherits from BaseDocumentProcessor but overrides with an
    async signature. The async process(url: str) method is used for URL-based
    transcript fetching. The sync process(file_bytes, file_name) is not implemented.

    No YouTube Data API key is required - the library uses public endpoints.

    Integration with ProcessorFactory happens in Story 5.5.

Error Handling:
    - InvalidYouTubeURLError: URL format not recognized or video ID cannot be extracted
    - TranscriptUnavailableError: Transcripts disabled by uploader or not available
"""

import logging
import re
from typing import override
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from ragitect.services.processor.base import BaseDocumentProcessor

logger = logging.getLogger(__name__)

# Language code to display name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh-Hans": "Chinese (Simplified)",
    "zh-Hant": "Chinese (Traditional)",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}


class InvalidYouTubeURLError(Exception):
    """Raised when URL is not a valid YouTube video URL.

    Causes:
    - URL doesn't match any YouTube format
    - Video ID cannot be extracted
    - Video is unavailable/deleted/private

    Attributes:
        message: Descriptive error message including URL or video ID
    """

    pass


class TranscriptUnavailableError(Exception):
    """Raised when transcript cannot be retrieved.

    Causes:
    - Transcripts disabled by uploader
    - No transcript available in any language
    - Video is age-restricted

    Attributes:
        message: Descriptive error message including URL for debugging
    """

    pass


class YouTubeProcessor(BaseDocumentProcessor):
    """Processor for extracting YouTube video transcripts as Markdown.

    Inherits from BaseDocumentProcessor but provides an async process(url: str)
    method instead of the sync process(file_bytes, file_name) method.

    Implements YouTube transcript extraction with:
    - Support for multiple URL formats (youtube.com, youtu.be, embed)
    - Language preference (English first, fallback to first available)
    - Timestamp preservation in [M:SS] or [H:MM:SS] format
    - No API key required (uses public transcript endpoints)

    The youtube-transcript-api library is synchronous, so API calls are
    wrapped with asyncio.run_in_executor for non-blocking async operation.

    Example:
        >>> processor = YouTubeProcessor()
        >>> markdown = await processor.process("https://youtu.be/dQw4w9WgXcQ")
        >>> print(markdown[:200])
        # YouTube Video Transcript

        **Video URL:** https://youtu.be/dQw4w9WgXcQ
        **Language:** English

        ---

        [0:00] We're no strangers to love
        ...
    """

    # Preferred language codes in order of preference
    PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]

    @override
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions.

        YouTubeProcessor is not file-based, so returns empty list.
        URL-based routing is handled separately from file extension routing.

        Returns:
            Empty list (not file-based)
        """
        return []

    async def process(self, url: str) -> str:
        """Fetch YouTube transcript and convert to Markdown.

        Args:
            url: YouTube video URL in any supported format

        Returns:
            Markdown string with video metadata and timestamped transcript

        Raises:
            InvalidYouTubeURLError: If URL is not a valid YouTube URL or video unavailable
            TranscriptUnavailableError: If transcript cannot be retrieved
        """
        logger.info(f"Processing YouTube URL: {url}")

        # Extract video ID from URL
        video_id = self._extract_video_id(url)
        logger.debug(f"Extracted video ID: {video_id}")

        try:
            # Get transcript with language preference
            transcript = self._get_transcript(video_id)
            language_code = self._get_transcript_language(video_id)

            # Format as Markdown
            markdown = self._format_markdown(transcript, url, video_id, language_code)

            logger.info(
                f"Successfully processed YouTube video {video_id} - {len(markdown)} chars extracted"
            )
            return markdown

        except TranscriptsDisabled:
            error_msg = f"Transcripts are disabled for this video: {url}"
            logger.error(error_msg)
            raise TranscriptUnavailableError(error_msg)

        except NoTranscriptFound:
            error_msg = f"No transcript found for video: {url}"
            logger.error(error_msg)
            raise TranscriptUnavailableError(error_msg)

        except VideoUnavailable:
            error_msg = f"Video unavailable: {url}"
            logger.error(error_msg)
            raise InvalidYouTubeURLError(error_msg)

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from various URL formats.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID

        Args:
            url: YouTube URL to parse

        Returns:
            11-character video ID

        Raises:
            InvalidYouTubeURLError: If video ID cannot be extracted
        """
        if not url:
            raise InvalidYouTubeURLError("Empty URL provided")

        # Pattern for youtu.be short URLs
        short_pattern = r"youtu\.be/([a-zA-Z0-9_-]{11})"

        # Pattern for embed/v URLs
        embed_pattern = r"youtube\.com/(?:embed|v)/([a-zA-Z0-9_-]{11})"

        # Check short URL format
        match = re.search(short_pattern, url)
        if match:
            return match.group(1)

        # Check embed format
        match = re.search(embed_pattern, url)
        if match:
            return match.group(1)

        # Parse standard watch URL
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id and len(video_id) == 11:
                return video_id

        raise InvalidYouTubeURLError(f"Could not extract video ID from URL: {url}")

    def _get_transcript(self, video_id: str) -> list[dict]:
        """Fetch transcript for video with language preference.

        Tries English transcripts first, falls back to first available language.

        Args:
            video_id: YouTube video ID

        Returns:
            List of transcript segments with 'text', 'start', 'duration' keys

        Raises:
            TranscriptsDisabled: If transcripts are disabled for this video
            NoTranscriptFound: If no transcript is available in any language
            VideoUnavailable: If video doesn't exist or is private
        """
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Try preferred languages first
        for lang_code in self.PREFERRED_LANGUAGES:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                fetched = transcript.fetch()
                # Convert FetchedTranscriptSnippet objects to dicts
                return [
                    {"text": s.text, "start": s.start, "duration": s.duration}
                    for s in fetched
                ]
            except NoTranscriptFound:
                continue

        # Fallback: get first available transcript
        for transcript in transcript_list:
            fetched = transcript.fetch()
            # Convert FetchedTranscriptSnippet objects to dicts
            return [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched
            ]

        raise NoTranscriptFound(video_id, [], "No transcripts available")

    def _get_transcript_language(self, video_id: str) -> str:
        """Get the language code of the fetched transcript.

        Uses same logic as _get_transcript to determine which language was used.

        Args:
            video_id: YouTube video ID

        Returns:
            Language code string (e.g., 'en', 'es', 'fr')
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            # Try preferred languages first
            for lang_code in self.PREFERRED_LANGUAGES:
                try:
                    transcript_list.find_transcript([lang_code])
                    return lang_code
                except NoTranscriptFound:
                    continue

            # Fallback: get first available transcript's language
            for transcript in transcript_list:
                return transcript.language_code

        except Exception:
            pass

        return "unknown"

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as [M:SS] or [H:MM:SS] timestamp.

        Args:
            seconds: Time in seconds (float)

        Returns:
            Formatted timestamp string like [1:05] or [1:01:05]
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if hours > 0:
            return f"[{hours}:{minutes:02d}:{secs:02d}]"
        return f"[{minutes}:{secs:02d}]"

    def _get_language_display_name(self, language_code: str) -> str:
        """Get human-readable language name from code.

        Args:
            language_code: ISO language code (e.g., 'en', 'es')

        Returns:
            Human-readable language name or the code if not found
        """
        return LANGUAGE_NAMES.get(language_code, language_code)

    def _format_markdown(
        self,
        transcript: list[dict],
        url: str,
        video_id: str,
        language_code: str,
    ) -> str:
        """Format transcript segments as Markdown with timestamps.

        Args:
            transcript: List of transcript segments from API
            url: Original YouTube URL
            video_id: Extracted video ID
            language_code: Detected transcript language

        Returns:
            Formatted Markdown string with metadata and transcript
        """
        lines = []

        # Header
        lines.append("# YouTube Video Transcript")
        lines.append("")

        # Metadata
        lines.append(f"**Video URL:** {url}")
        language_name = self._get_language_display_name(language_code)
        lines.append(f"**Language:** {language_name}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Transcript with timestamps
        for segment in transcript:
            timestamp = self._format_timestamp(segment["start"])
            text = segment["text"].strip()
            lines.append(f"{timestamp} {text}")

        return "\n".join(lines)
