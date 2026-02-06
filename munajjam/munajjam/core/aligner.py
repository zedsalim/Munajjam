"""
Unified Aligner class for Quran audio alignment.

This module provides a single, simple interface for aligning transcribed
audio segments with reference Quran ayahs. It supports multiple alignment
strategies and handles all post-processing (zone realignment, overlap fixing).

Usage:
    from munajjam.core import Aligner

    aligner = Aligner(strategy="hybrid")
    results = aligner.align(segments, ayahs, silences_ms=silences)
"""

from enum import Enum
from typing import Callable

from ..models import Segment, Ayah, AlignmentResult
from .hybrid import HybridStats


class AlignmentStrategy(str, Enum):
    """Available alignment strategies."""
    GREEDY = "greedy"  # Fast, simple greedy matching
    DP = "dp"  # Dynamic programming for optimal alignment
    HYBRID = "hybrid"  # DP with fallback to greedy (recommended)
    WORD_DP = "word_dp"  # Word-level DP for sub-segment precision
    CTC_SEG = "ctc_seg"  # CTC segmentation — acoustic-based alignment
    AUTO = "auto"  # Automatically pick best strategy based on surah size


class Aligner:
    """
    Unified alignment interface.

    Provides a single entry point for all alignment operations with
    configurable strategy and automatic post-processing.

    Attributes:
        strategy: The alignment strategy to use
        quality_threshold: Similarity threshold for high-quality alignment (0.0-1.0)
        fix_drift: Whether to run zone realignment to fix timing drift
        fix_overlaps: Whether to fix overlapping ayah timings

    Example:
        >>> aligner = Aligner(strategy="hybrid")
        >>> results = aligner.align(segments, ayahs)
        >>> for r in results:
        ...     print(f"Ayah {r.ayah.ayah_number}: {r.start_time:.2f}s - {r.end_time:.2f}s")
    """

    def __init__(
        self,
        strategy: str | AlignmentStrategy = AlignmentStrategy.AUTO,
        quality_threshold: float = 0.85,
        fix_drift: bool = True,
        fix_overlaps: bool = True,
        min_gap: float = 0.3,
        ctc_refine: bool = False,
        energy_snap: bool = False,
        audio_path: str | None = None,
    ):
        """
        Initialize the Aligner.

        Args:
            strategy: Alignment strategy ("greedy", "dp", "hybrid", "word_dp", "ctc_seg", or "auto")
            quality_threshold: Similarity threshold for quality checks (0.0-1.0)
            fix_drift: Run zone realignment to fix timing drift in long surahs
            fix_overlaps: Fix any overlapping ayah timings
            min_gap: Minimum gap in seconds between consecutive ayahs (default 0.3)
            ctc_refine: Run CTC forced alignment as a refinement pass (requires torchaudio)
            energy_snap: Snap boundaries to energy minima for precise timing (requires audio_path)
            audio_path: Path to audio file (required if ctc_refine=True or energy_snap=True)
        """
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.fix_drift = fix_drift
        self.fix_overlaps = fix_overlaps
        self.min_gap = min_gap
        self.ctc_refine = ctc_refine
        self.energy_snap = energy_snap
        self.audio_path = audio_path

        # Stats from last alignment (only populated for hybrid strategy)
        self._last_stats: HybridStats | None = None

    @property
    def last_stats(self) -> HybridStats | None:
        """Get stats from the last hybrid alignment, or None if not applicable."""
        return self._last_stats

    def _select_strategy(self, ayahs: list[Ayah]) -> AlignmentStrategy:
        """Pick the best concrete strategy based on surah characteristics.

        When audio_path is available and CTC segmentation is supported,
        prefer CTC_SEG as it uses acoustic evidence directly.
        Otherwise: word_dp for short-ayah surahs, hybrid for long-ayah ones.
        """
        # Prefer CTC segmentation when audio is available
        if self.audio_path:
            from .ctc_segmentation import is_available as ctc_available
            if ctc_available():
                return AlignmentStrategy.CTC_SEG

        total_words = sum(len(a.text.split()) for a in ayahs)
        avg_words = total_words / len(ayahs) if ayahs else 0
        if total_words > 4000 and avg_words > 15:
            return AlignmentStrategy.HYBRID
        return AlignmentStrategy.WORD_DP

    def align(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[AlignmentResult]:
        """
        Align transcribed segments to reference ayahs.

        This is the main method for performing alignment. It automatically
        applies the configured strategy and post-processing steps.

        Args:
            segments: List of transcribed Segment objects
            ayahs: List of reference Ayah objects (in order)
            silences_ms: Optional silence periods in milliseconds [(start, end), ...]
            on_progress: Optional callback for progress updates (current, total)

        Returns:
            List of AlignmentResult objects with timing and similarity info
        """
        if not segments or not ayahs:
            return []

        # Clear previous stats
        self._last_stats = None

        # Resolve AUTO to a concrete strategy
        strategy = self.strategy
        if strategy == AlignmentStrategy.AUTO:
            strategy = self._select_strategy(ayahs)

        # Run alignment based on strategy
        if strategy == AlignmentStrategy.GREEDY:
            results = self._align_greedy(segments, ayahs, silences_ms)
        elif strategy == AlignmentStrategy.DP:
            results = self._align_dp(segments, ayahs, silences_ms, on_progress)
        elif strategy == AlignmentStrategy.WORD_DP:
            results = self._align_word_dp(segments, ayahs, silences_ms, on_progress)
        elif strategy == AlignmentStrategy.CTC_SEG:
            results = self._align_ctc_seg(segments, ayahs, silences_ms, on_progress)
        else:  # HYBRID
            results = self._align_hybrid(segments, ayahs, silences_ms, on_progress)

        # Post-processing
        if self.fix_drift and results:
            results = self._apply_drift_fix(results, segments, ayahs)

        # CTC forced alignment refinement pass
        if self.ctc_refine and self.audio_path and results:
            self._apply_ctc_refinement(results)

        # Snap boundaries to energy minima for precise timing
        if self.energy_snap and self.audio_path and results:
            self._apply_energy_snap(results)

        # Snap ayah boundaries to actual silence periods (fixes timestamp drift)
        if silences_ms and results:
            self._snap_to_silences(results, silences_ms)

        if self.fix_overlaps and results:
            self._apply_overlap_fix(results)

        return results

    def _align_greedy(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
    ) -> list[AlignmentResult]:
        """Run greedy alignment."""
        from .aligner_greedy import align_segments

        return align_segments(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
        )

    def _align_dp(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run DP alignment."""
        from .dp_core import align_segments_dp_with_constraints

        return align_segments_dp_with_constraints(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            on_progress=on_progress,
        )

    def _align_word_dp(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run word-level DP alignment."""
        from .word_level_dp import align_segments_word_dp

        return align_segments_word_dp(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            on_progress=on_progress,
        )

    def _align_hybrid(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run hybrid alignment (DP + fallback)."""
        from .hybrid import align_segments_hybrid

        results, stats = align_segments_hybrid(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
            quality_threshold=self.quality_threshold,
            on_progress=on_progress,
        )
        self._last_stats = stats
        return results

    def _align_ctc_seg(
        self,
        segments: list[Segment],
        ayahs: list[Ayah],
        silences_ms: list[tuple[int, int]] | None,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[AlignmentResult]:
        """Run CTC segmentation with word-DP validation.

        Phase 2+3 hybrid pipeline:
        1. CTC segmentation for coarse per-ayah boundaries.
        2. Word-DP alignment for comparison.
        3. Per-ayah: keep whichever result has higher similarity.
        Falls back entirely to word-DP if CTC is unavailable.
        """
        from .ctc_segmentation import (
            ctc_segment_ayahs,
            ctc_segment_to_alignment_results,
            is_available as ctc_available,
        )
        from .matcher import similarity as _sim

        # Always run word-DP as baseline/fallback
        word_dp_results = self._align_word_dp(
            segments, ayahs, silences_ms, on_progress,
        )

        if not self.audio_path or not ctc_available():
            return word_dp_results

        # Run CTC segmentation
        ayah_texts = [a.text for a in ayahs]
        boundaries = ctc_segment_ayahs(self.audio_path, ayah_texts)

        if boundaries is None:
            return word_dp_results

        ctc_results = ctc_segment_to_alignment_results(
            boundaries, ayahs, segments,
        )

        if not ctc_results:
            return word_dp_results

        # Per-ayah fusion: keep the result with higher similarity.
        # Build lookup by ayah number.
        ctc_by_ayah = {r.ayah.ayah_number: r for r in ctc_results}
        dp_by_ayah = {r.ayah.ayah_number: r for r in word_dp_results}

        fused: list[AlignmentResult] = []
        for ayah in ayahs:
            ctc_r = ctc_by_ayah.get(ayah.ayah_number)
            dp_r = dp_by_ayah.get(ayah.ayah_number)

            if ctc_r and dp_r:
                # Pick whichever has higher similarity
                if ctc_r.similarity_score >= dp_r.similarity_score:
                    fused.append(ctc_r)
                else:
                    fused.append(dp_r)
            elif dp_r:
                fused.append(dp_r)
            elif ctc_r:
                fused.append(ctc_r)

        return fused if fused else word_dp_results

    def _apply_drift_fix(
        self,
        results: list[AlignmentResult],
        segments: list[Segment],
        ayahs: list[Ayah],
    ) -> list[AlignmentResult]:
        """Apply zone realignment to fix timing drift."""
        from .zone_realigner import (
            iterative_realign_problem_zones,
            realign_from_anchors,
            realign_drift_zones_word_dp,
            refine_low_confidence_zones_with_ctc,
        )

        # First pass: iterative zone realignment with adaptive thresholds
        results, _ = iterative_realign_problem_zones(
            results=results,
            segments=segments,
            ayahs=ayahs,
            passes=3,
            initial_threshold=self.quality_threshold,
            buffer_seconds=10.0,
        )

        # Second pass: anchor-based realignment with confidence weighting
        results, _ = realign_from_anchors(
            results=results,
            segments=segments,
            ayahs=ayahs,
            min_gap_size=3,
            buffer_seconds=5.0,
        )

        # Third pass: zone-level word-DP fallback for drifted pace regions.
        results, _ = realign_drift_zones_word_dp(
            results=results,
            segments=segments,
            ayahs=ayahs,
            min_consecutive=4,
            max_pace_ratio=2.5,
        )

        # Optional fourth pass: CTC refinement only on problematic zones.
        if self.ctc_refine and self.audio_path:
            results, _ = refine_low_confidence_zones_with_ctc(
                results=results,
                audio_path=self.audio_path,
                similarity_threshold=max(0.5, self.quality_threshold - 0.1),
                min_consecutive=2,
                min_ctc_score=0.3,
                max_pace_ratio=2.5,
            )

        return results

    def _apply_energy_snap(self, results: list[AlignmentResult]) -> int:
        """Snap boundaries to local energy minima for precise timing."""
        from .zone_realigner import snap_boundaries_to_energy
        from ..transcription.silence import compute_energy_envelope

        try:
            envelope = compute_energy_envelope(self.audio_path)
        except Exception:
            return 0

        return snap_boundaries_to_energy(results, envelope, max_snap_distance=1.0)

    def _apply_ctc_refinement(self, results: list[AlignmentResult]) -> int:
        """Refine timestamps using CTC forced alignment (if available)."""
        from .forced_aligner import refine_alignment_results, is_available

        if not is_available():
            return 0

        return refine_alignment_results(
            results=results,
            audio_path=self.audio_path,
            min_similarity=0.5,
            min_ctc_score=0.3,
        )

    def _snap_to_silences(
        self,
        results: list[AlignmentResult],
        silences_ms: list[tuple[int, int]],
    ) -> int:
        """Snap ayah boundaries to actual silence periods to fix drift."""
        from .zone_realigner import snap_boundaries_to_silences

        return snap_boundaries_to_silences(results, silences_ms)

    def _apply_overlap_fix(self, results: list[AlignmentResult]) -> int:
        """Fix overlapping ayah timings and ensure minimum gaps in-place."""
        from .zone_realigner import fix_overlaps

        return fix_overlaps(results, min_gap=self.min_gap)


# Convenience function for simple usage
def align(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    strategy: str = "auto",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    Convenience function for alignment with default settings.

    This is equivalent to:
        Aligner(strategy=strategy).align(segments, ayahs, silences_ms)

    Args:
        segments: List of transcribed Segment objects
        ayahs: List of reference Ayah objects
        silences_ms: Optional silence periods in milliseconds
        strategy: Alignment strategy ("greedy", "dp", "hybrid", "word_dp", "ctc_seg", or "auto")
        on_progress: Optional progress callback

    Returns:
        List of AlignmentResult objects
    """
    aligner = Aligner(strategy=strategy)
    return aligner.align(segments, ayahs, silences_ms, on_progress)
