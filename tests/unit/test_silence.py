"""
Unit tests for silence detection.
"""

import pytest
from munajjam.transcription.silence import detect_silences


class TestDetectSilences:
    """Test silence detection functions."""

    def test_detect_silences_file_not_found(self):
        """Test silence detection with non-existent file."""
        with pytest.raises(Exception):
            detect_silences("nonexistent_file.wav")

    def test_silence_tuple_format(self, sample_silences):
        """Test that silences are in (start_ms, end_ms) format."""
        for start, end in sample_silences:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end
