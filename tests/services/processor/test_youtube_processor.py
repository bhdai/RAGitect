"""Tests for YouTubeProcessor - YouTube transcript extraction and markdown formatting.

Red-Green-Refactor TDD: These tests define expected behavior before implementation.
"""

import pytest
from unittest.mock import patch

# Module-level markers as per project-context.md
pytestmark = [pytest.mark.asyncio]


class TestYouTubeProcessorInterface:
    """Test YouTubeProcessor class interface and method signatures (AC1)"""

    def test_class_exists(self):
        """YouTubeProcessor class should be importable"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert processor is not None

    def test_inherits_from_base_document_processor(self):
        """YouTubeProcessor should inherit from BaseDocumentProcessor"""
        from ragitect.services.processor.base import BaseDocumentProcessor
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert isinstance(processor, BaseDocumentProcessor)

    def test_supported_formats_returns_empty_list(self):
        """YouTubeProcessor is not file-based, returns empty list"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        formats = processor.supported_formats()
        assert formats == []

    async def test_process_method_signature_async(self):
        """process() should be async and accept url string, return Markdown string"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [{"text": "Test", "start": 0.0, "duration": 1.0}]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )
                    assert isinstance(result, str)


class TestVideoIdExtraction:
    """Test video ID extraction from various URL formats (AC1)"""

    def test_extract_from_standard_url(self):
        """Extract video ID from standard youtube.com/watch?v= format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert processor._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_short_url(self):
        """Extract video ID from youtu.be/ short format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert processor._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_embed_url(self):
        """Extract video ID from youtube.com/embed/ format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert processor._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_url_with_extra_params(self):
        """Extract video ID from URL with additional query parameters"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&list=PLtest"
        assert processor._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_short_url_with_params(self):
        """Extract video ID from youtu.be with query parameters"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://youtu.be/dQw4w9WgXcQ?t=30"
        assert processor._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises_error(self):
        """Invalid URL should raise InvalidYouTubeURLError"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            InvalidYouTubeURLError,
        )

        processor = YouTubeProcessor()
        with pytest.raises(InvalidYouTubeURLError):
            processor._extract_video_id("https://example.com/not-youtube")

    def test_empty_url_raises_error(self):
        """Empty URL should raise InvalidYouTubeURLError"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            InvalidYouTubeURLError,
        )

        processor = YouTubeProcessor()
        with pytest.raises(InvalidYouTubeURLError):
            processor._extract_video_id("")

    def test_url_without_video_id_raises_error(self):
        """YouTube URL without video ID should raise error"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            InvalidYouTubeURLError,
        )

        processor = YouTubeProcessor()
        with pytest.raises(InvalidYouTubeURLError):
            processor._extract_video_id("https://www.youtube.com/watch")


class TestTranscriptExtraction:
    """Test transcript extraction and formatting (AC2, AC3)"""

    async def test_process_returns_markdown(self):
        """process() should return Markdown formatted string"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test", "start": 2.0, "duration": 3.0},
        ]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    assert isinstance(result, str)
                    assert len(result) > 0
                    # Should contain heading
                    assert "#" in result

    async def test_markdown_includes_timestamps(self):
        """Markdown output should include timestamps in [MM:SS] format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [
            {"text": "First segment", "start": 5.0, "duration": 2.0},
            {"text": "Second segment", "start": 65.0, "duration": 3.0},
        ]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    # Check for timestamp format [0:05] or [00:05]
                    assert "[0:05]" in result or "[00:05]" in result
                    # Check for [1:05] format (65 seconds)
                    assert "[1:05]" in result or "[01:05]" in result

    async def test_markdown_includes_video_url(self):
        """Markdown output should include video URL as metadata"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [{"text": "Test", "start": 0.0, "duration": 1.0}]
        test_url = "https://www.youtube.com/watch?v=testVIDEOidX"

        with patch.object(processor, "_extract_video_id", return_value="testVIDEOidX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(test_url)

                    assert "testVIDEOidX" in result or test_url in result

    async def test_markdown_includes_transcript_text(self):
        """Markdown output should include transcript text content"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [
            {"text": "Never gonna give you up", "start": 0.0, "duration": 2.0},
            {"text": "Never gonna let you down", "start": 2.0, "duration": 2.0},
        ]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    assert "Never gonna give you up" in result
                    assert "Never gonna let you down" in result

    async def test_hour_long_video_timestamp_format(self):
        """Videos over 1 hour should use [H:MM:SS] timestamp format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [
            {"text": "Introduction", "start": 0.0, "duration": 2.0},
            {"text": "After one hour", "start": 3665.0, "duration": 2.0},  # 1:01:05
        ]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    # Should have [H:MM:SS] format for hour+ videos
                    assert "[1:01:05]" in result


class TestTimestampFormatting:
    """Test timestamp formatting helper function"""

    def test_format_timestamp_seconds_only(self):
        """Format seconds as [M:SS]"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert processor._format_timestamp(5.5) == "[0:05]"
        assert processor._format_timestamp(45) == "[0:45]"

    def test_format_timestamp_minutes_and_seconds(self):
        """Format minutes and seconds as [M:SS]"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert processor._format_timestamp(65) == "[1:05]"
        assert processor._format_timestamp(600) == "[10:00]"

    def test_format_timestamp_hours_format(self):
        """Format hours as [H:MM:SS]"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert processor._format_timestamp(3661) == "[1:01:01]"
        assert processor._format_timestamp(7200) == "[2:00:00]"

    def test_format_timestamp_zero(self):
        """Format zero seconds as [0:00]"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        assert processor._format_timestamp(0) == "[0:00]"


class TestLanguagePreference:
    """Test multi-language support and preference (AC5)"""

    async def test_english_transcript_preferred(self):
        """English transcript should be preferred if available"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [{"text": "English content", "start": 0.0, "duration": 1.0}]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="en"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    # Should indicate English language
                    assert "English" in result or "en" in result.lower()

    async def test_fallback_to_first_available_language(self):
        """If English unavailable, use first available language"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [
            {"text": "Contenido en español", "start": 0.0, "duration": 1.0}
        ]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="es"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    # Should work and include the content
                    assert "Contenido en español" in result
                    # Should indicate the language used
                    assert "Spanish" in result or "es" in result.lower()

    async def test_markdown_includes_language_metadata(self):
        """Markdown output should include detected language as metadata"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()

        mock_transcript = [{"text": "Content", "start": 0.0, "duration": 1.0}]

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", return_value=mock_transcript
            ):
                with patch.object(
                    processor, "_get_transcript_language", return_value="fr"
                ):
                    result = await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                    # Should include language info
                    assert "Language" in result or "language" in result


class TestErrorHandling:
    """Test error handling for various failure scenarios (AC4)"""

    async def test_transcripts_disabled_raises_error(self):
        """Transcripts disabled by uploader should raise TranscriptUnavailableError"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            TranscriptUnavailableError,
        )
        from youtube_transcript_api._errors import TranscriptsDisabled

        processor = YouTubeProcessor()

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", side_effect=TranscriptsDisabled("videoId")
            ):
                with pytest.raises(TranscriptUnavailableError) as exc_info:
                    await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                assert "disabled" in str(exc_info.value).lower()
                assert "test123abcX" in str(exc_info.value)

    async def test_no_transcript_found_raises_error(self):
        """No transcript available should raise TranscriptUnavailableError"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            TranscriptUnavailableError,
        )
        from youtube_transcript_api._errors import NoTranscriptFound

        processor = YouTubeProcessor()

        mock_exception = NoTranscriptFound("videoId", ["en"], "Requested")

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(processor, "_get_transcript", side_effect=mock_exception):
                with pytest.raises(TranscriptUnavailableError) as exc_info:
                    await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                assert (
                    "not found" in str(exc_info.value).lower()
                    or "no transcript" in str(exc_info.value).lower()
                )

    async def test_video_unavailable_raises_invalid_url_error(self):
        """Video unavailable should raise InvalidYouTubeURLError"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            InvalidYouTubeURLError,
        )
        from youtube_transcript_api._errors import VideoUnavailable

        processor = YouTubeProcessor()

        with patch.object(processor, "_extract_video_id", return_value="test123abcX"):
            with patch.object(
                processor, "_get_transcript", side_effect=VideoUnavailable("videoId")
            ):
                with pytest.raises(InvalidYouTubeURLError) as exc_info:
                    await processor.process(
                        "https://www.youtube.com/watch?v=test123abcX"
                    )

                assert (
                    "test123abcX" in str(exc_info.value)
                    or "unavailable" in str(exc_info.value).lower()
                )

    async def test_exception_messages_contain_url(self):
        """All exception messages should include the URL for debugging"""
        from ragitect.services.processor.youtube_processor import (
            YouTubeProcessor,
            TranscriptUnavailableError,
        )
        from youtube_transcript_api._errors import TranscriptsDisabled

        processor = YouTubeProcessor()
        test_video_id = "debugTestId"

        with patch.object(processor, "_extract_video_id", return_value=test_video_id):
            with patch.object(
                processor,
                "_get_transcript",
                side_effect=TranscriptsDisabled(test_video_id),
            ):
                with pytest.raises(TranscriptUnavailableError) as exc_info:
                    await processor.process(
                        f"https://www.youtube.com/watch?v={test_video_id}"
                    )

                # Video ID or URL should be in error message for debugging
                assert test_video_id in str(exc_info.value)


class TestExceptions:
    """Test custom exception classes exist and are properly defined"""

    def test_invalid_youtube_url_error_exists(self):
        """InvalidYouTubeURLError exception class should exist"""
        from ragitect.services.processor.youtube_processor import InvalidYouTubeURLError

        error = InvalidYouTubeURLError("Test error message")
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"

    def test_transcript_unavailable_error_exists(self):
        """TranscriptUnavailableError exception class should exist"""
        from ragitect.services.processor.youtube_processor import (
            TranscriptUnavailableError,
        )

        error = TranscriptUnavailableError("Test error message")
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"


@pytest.mark.integration
class TestYouTubeProcessorIntegration:
    """Integration tests with real YouTube API (require network access) (AC6)"""

    async def test_process_real_video_with_captions(self):
        """Integration test: fetch real YouTube transcript"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        # Use a well-known video with captions (TED talk or similar)
        # Rick Astley - Never Gonna Give You Up has captions
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        markdown = await processor.process(url)

        # Verify substantial content
        assert len(markdown) > 100
        # Verify has heading
        assert "#" in markdown
        # Verify has timestamps
        assert "[" in markdown and "]" in markdown

    async def test_real_video_timestamps_preserved(self):
        """Integration test: verify timestamps are in correct format"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        markdown = await processor.process(url)

        # Check for timestamp pattern [M:SS] or [MM:SS] or [H:MM:SS]
        import re

        timestamp_pattern = r"\[\d{1,2}:\d{2}(?::\d{2})?\]"
        timestamps = re.findall(timestamp_pattern, markdown)
        assert len(timestamps) > 0, (
            "Should have timestamps in [M:SS] or [H:MM:SS] format"
        )

    async def test_real_video_includes_metadata(self):
        """Integration test: verify markdown includes video metadata"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        markdown = await processor.process(url)

        # Should include video URL or ID
        assert "dQw4w9WgXcQ" in markdown or "youtube.com" in markdown.lower()
        # Should include language info
        assert "Language" in markdown or "language" in markdown.lower()

    async def test_markdown_compatible_with_chunking(self):
        """Integration test: verify markdown works with DocumentProcessor chunking"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor
        from ragitect.services.document_processor import split_markdown_document

        processor = YouTubeProcessor()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # Fetch and extract
        markdown = await processor.process(url)

        # Test with existing chunker
        chunks = split_markdown_document(
            raw_text=markdown,
            chunk_size=512,
            overlap=50,
        )

        # Verify chunking works
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all(isinstance(chunk, str) for chunk in chunks), (
            "Chunks should be strings"
        )
        assert all(len(chunk) > 0 for chunk in chunks), "Each chunk should have content"

    async def test_youtu_be_short_url_works(self):
        """Integration test: youtu.be short URL format works"""
        from ragitect.services.processor.youtube_processor import YouTubeProcessor

        processor = YouTubeProcessor()
        # Same video, short URL format
        url = "https://youtu.be/dQw4w9WgXcQ"

        markdown = await processor.process(url)

        assert len(markdown) > 100
        assert "#" in markdown
