"""
Unit tests for zone-level realignment helpers.
"""

from munajjam.core.zone_realigner import (
    _find_problem_runs,
    refine_low_confidence_zones_with_ctc,
)
from munajjam.models import AlignmentResult, Ayah


def _make_result(
    ayah_number: int,
    start: float,
    end: float,
    similarity: float,
    text: str = "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
) -> AlignmentResult:
    ayah = Ayah(
        id=ayah_number,
        surah_id=112,
        ayah_number=ayah_number,
        text=text,
    )
    return AlignmentResult(
        ayah=ayah,
        start_time=start,
        end_time=end,
        transcribed_text=text,
        similarity_score=similarity,
        overlap_detected=False,
    )


def test_find_problem_runs_detects_low_similarity_sequence():
    results = [
        _make_result(1, 0.0, 3.0, 0.95),
        _make_result(2, 3.2, 6.0, 0.62),
        _make_result(3, 6.2, 9.0, 0.58),
        _make_result(4, 9.2, 12.0, 0.92),
    ]

    runs = _find_problem_runs(
        results=results,
        similarity_threshold=0.75,
        min_consecutive=2,
        max_pace_ratio=2.5,
    )

    assert runs == [(1, 3)]


def test_refine_low_confidence_zones_with_ctc_noop_when_unavailable(monkeypatch):
    results = [
        _make_result(1, 0.0, 3.0, 0.95),
        _make_result(2, 3.2, 6.0, 0.62),
        _make_result(3, 6.2, 9.0, 0.58),
    ]

    # Force CTC path to skip without touching timestamps.
    monkeypatch.setattr(
        "munajjam.core.forced_aligner.is_available",
        lambda: False,
    )

    updated, refined = refine_low_confidence_zones_with_ctc(
        results=results,
        audio_path="dummy.mp3",
        similarity_threshold=0.75,
        min_consecutive=2,
    )

    assert refined == 0
    assert [r.start_time for r in updated] == [r.start_time for r in results]
    assert [r.end_time for r in updated] == [r.end_time for r in results]

