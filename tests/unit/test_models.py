"""
Unit tests for data models.
"""

import pytest
from munajjam.models import Segment, SegmentType, Ayah, AlignmentResult


class TestSegment:
    """Test Segment model."""

    def test_segment_creation(self):
        """Test creating a segment with all fields."""
        segment = Segment(
            id=1, surah_id=1, start=0.0, end=5.0,
            text="بسم الله", type=SegmentType.AYAH
        )

        assert segment.id == 1
        assert segment.surah_id == 1
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "بسم الله"
        assert segment.type == SegmentType.AYAH
        assert segment.duration == 5.0

    @pytest.mark.parametrize("seg_type", [SegmentType.AYAH, SegmentType.BASMALA, SegmentType.ISTIADHA])
    def test_segment_types(self, seg_type):
        """Test all segment types are valid."""
        segment = Segment(id=1, surah_id=1, start=0.0, end=5.0, text="text", type=seg_type)
        assert segment.type == seg_type


class TestAyah:
    """Test Ayah model."""

    def test_ayah_creation(self):
        """Test creating an ayah."""
        ayah = Ayah(id=1, surah_id=1, ayah_number=1, text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")

        assert ayah.id == 1
        assert ayah.surah_id == 1
        assert ayah.ayah_number == 1
        assert "بِسْمِ" in ayah.text

    @pytest.mark.parametrize("invalid_field,value", [
        ("id", 0),
        ("surah_id", 0),
        ("surah_id", 999),
    ])
    def test_ayah_validation(self, invalid_field, value):
        """Test ayah validation for invalid values."""
        kwargs = {"id": 1, "surah_id": 1, "ayah_number": 1, "text": "text"}
        kwargs[invalid_field] = value

        with pytest.raises(Exception):
            Ayah(**kwargs)

    def test_ayah_text_not_empty(self):
        """Test that ayah text cannot be empty."""
        with pytest.raises(Exception):
            Ayah(id=1, surah_id=1, ayah_number=1, text="")


class TestAlignmentResult:
    """Test AlignmentResult model."""

    def test_alignment_result_creation(self, sample_ayahs):
        """Test creating an alignment result."""
        result = AlignmentResult(
            ayah=sample_ayahs[0], start_time=0.0, end_time=5.0,
            transcribed_text="بسم الله", similarity_score=1.0
        )

        assert result.ayah == sample_ayahs[0]
        assert result.start_time == 0.0
        assert result.end_time == 5.0
        assert result.duration == 5.0
        assert result.similarity_score == 1.0
        assert result.overlap_detected is False

    @pytest.mark.parametrize("score,expected_confidence", [
        (0.9, True),
        (0.7, False),
    ])
    def test_high_confidence_threshold(self, sample_ayahs, score, expected_confidence):
        """Test high confidence detection."""
        result = AlignmentResult(
            ayah=sample_ayahs[0], start_time=0.0, end_time=5.0,
            transcribed_text="text", similarity_score=score
        )
        assert result.is_high_confidence == expected_confidence

    def test_invalid_similarity_score(self, sample_ayahs):
        """Test that similarity score must be in [0, 1]."""
        with pytest.raises(Exception):
            AlignmentResult(
                ayah=sample_ayahs[0], start_time=0.0, end_time=5.0,
                transcribed_text="text", similarity_score=1.5
            )
