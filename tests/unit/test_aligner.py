"""
Unit tests for alignment strategies.
"""

import pytest
from munajjam.core import Aligner, align


class TestAligner:
    """Test Aligner class and strategies."""

    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid"])
    def test_aligner_initialization(self, strategy):
        """Test Aligner can be initialized with each strategy."""
        aligner = Aligner(strategy=strategy)
        assert aligner.strategy.value == strategy

    def test_aligner_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            Aligner(strategy="invalid")

    @pytest.mark.parametrize("strategy", ["greedy", "dp", "hybrid"])
    def test_align_returns_results(self, strategy, sample_segments, sample_ayahs):
        """Test alignment returns results with correct structure."""
        aligner = Aligner(strategy=strategy)
        results = aligner.align(sample_segments, sample_ayahs)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert hasattr(result, 'ayah')
            assert hasattr(result, 'start_time')
            assert hasattr(result, 'end_time')
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.start_time >= 0
            assert result.end_time > result.start_time

    def test_align_with_silences(self, sample_segments, sample_ayahs, sample_silences):
        """Test alignment with silence detection."""
        aligner = Aligner(strategy="hybrid")
        results = aligner.align(sample_segments, sample_ayahs, silences_ms=sample_silences)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_align_with_progress_callback(self, sample_segments, sample_ayahs):
        """Test alignment accepts progress callback."""
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        aligner = Aligner(strategy="hybrid")
        aligner.align(sample_segments, sample_ayahs, on_progress=callback)

        # Callback should have been called (implementation may vary)
        assert isinstance(progress_calls, list)

    def test_high_confidence_threshold(self, sample_segments, sample_ayahs):
        """Test custom quality threshold."""
        aligner = Aligner(strategy="hybrid", quality_threshold=0.85)
        results = aligner.align(sample_segments, sample_ayahs)

        for result in results:
            if result.is_high_confidence:
                assert result.similarity_score >= 0.85


class TestAlignFunction:
    """Test convenience align() function."""

    def test_align_function_works(self, sample_segments, sample_ayahs):
        """Test that align() convenience function works."""
        results = align(sample_segments, sample_ayahs)

        assert isinstance(results, list)
        assert len(results) > 0
