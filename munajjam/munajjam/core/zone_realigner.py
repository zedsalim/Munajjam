"""
Zone Re-alignment Module

Fixes drift issues in long surahs by identifying and re-aligning problematic regions.

WHAT IS DRIFT?
--------------
In long surahs (especially 30+ minutes), timing can gradually drift away from
the true alignment. This happens because:
1. Small errors accumulate over time
2. The reciter's pace may change throughout the recording
3. Pauses and emphasis patterns vary

Example of drift:
- Ayahs 1-50: Good alignment (similarity ~0.95)
- Ayahs 51-75: Drift begins (similarity drops to ~0.75)
- Ayahs 76-100: Severe drift (similarity ~0.60)
- Ayahs 101-120: Good again after a natural break

SOLUTION: ZONE RE-ALIGNMENT
---------------------------
Instead of re-aligning the entire surah (expensive), we:
1. Identify "problem zones" - consecutive low-confidence ayahs
2. Re-align ONLY those zones using segments in their time range
3. Keep the better result for each ayah (old vs. new)

This is much faster and preserves good alignments while fixing drift.

TWO STRATEGIES:
--------------
1. Problem Zone Re-alignment: Fixes obvious low-confidence regions
2. Anchor-based Re-alignment: Uses high-confidence ayahs as anchors
   and re-aligns the gaps between them
"""

from dataclasses import dataclass
from munajjam.models import Segment, Ayah, AlignmentResult
from .dp_core import align_segments_dp


@dataclass
class ZoneStats:
    """Statistics from zone re-alignment."""
    zones_found: int = 0
    zones_improved: int = 0
    ayahs_improved: int = 0
    ayahs_unchanged: int = 0
    ayahs_degraded: int = 0


@dataclass
class ProblemZone:
    """A zone of consecutive low-confidence ayahs."""
    start_idx: int  # Index in results list
    end_idx: int    # Index in results list (exclusive)
    start_ayah: int
    end_ayah: int
    avg_similarity: float
    start_time: float
    end_time: float


def identify_problem_zones(
    results: list[AlignmentResult],
    min_consecutive: int = 3,
    quality_threshold: float = 0.85,
) -> list[ProblemZone]:
    """
    Find sequences of consecutive low-confidence ayahs (drift zones).

    A "problem zone" is a region where alignment quality has degraded,
    typically due to timing drift in long recordings.

    Example scenario:
    ```
    Ayah 48: similarity 0.92 ✓
    Ayah 49: similarity 0.88 ✓
    Ayah 50: similarity 0.78 ← START OF ZONE
    Ayah 51: similarity 0.75 ← IN ZONE
    Ayah 52: similarity 0.72 ← IN ZONE
    Ayah 53: similarity 0.90 ✓ ← END OF ZONE
    ```

    If min_consecutive=3 and quality_threshold=0.85, ayahs 50-52 form a zone.

    Args:
        results: List of alignment results (in order)
        min_consecutive: Minimum consecutive low-confidence ayahs to form a zone.
                        Default 3 prevents single outliers from triggering re-alignment.
        quality_threshold: Similarity below which an ayah is "low confidence".
                          Default 0.85 means 85% similarity is the cutoff.

    Returns:
        List of ProblemZone objects, each containing:
        - start_idx, end_idx: Slice indices in results list
        - start_ayah, end_ayah: Ayah numbers
        - avg_similarity: Average similarity in the zone
        - start_time, end_time: Time boundaries for re-alignment
    """
    zones = []
    current_zone_start = None  # Index where current zone started
    current_zone_sims = []     # Similarity scores in current zone

    for i, result in enumerate(results):
        is_low = result.similarity_score < quality_threshold

        if is_low:
            # This ayah is low-confidence
            if current_zone_start is None:
                # Start a new zone
                current_zone_start = i
                current_zone_sims = [result.similarity_score]
            else:
                # Continue existing zone
                current_zone_sims.append(result.similarity_score)
        else:
            # This ayah is high-confidence - potential end of zone
            if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
                # We have a valid zone (enough consecutive low-conf ayahs)
                zone_results = results[current_zone_start:i]
                zones.append(ProblemZone(
                    start_idx=current_zone_start,
                    end_idx=i,
                    start_ayah=zone_results[0].ayah.ayah_number,
                    end_ayah=zone_results[-1].ayah.ayah_number,
                    avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
                    start_time=zone_results[0].start_time,
                    end_time=zone_results[-1].end_time,
                ))
            # Reset zone tracking
            current_zone_start = None
            current_zone_sims = []

    # Handle zone that extends to the end of the surah
    if current_zone_start is not None and len(current_zone_sims) >= min_consecutive:
        zone_results = results[current_zone_start:]
        zones.append(ProblemZone(
            start_idx=current_zone_start,
            end_idx=len(results),
            start_ayah=zone_results[0].ayah.ayah_number,
            end_ayah=zone_results[-1].ayah.ayah_number,
            avg_similarity=sum(current_zone_sims) / len(current_zone_sims),
            start_time=zone_results[0].start_time,
            end_time=zone_results[-1].end_time,
        ))

    return zones


def find_segments_for_zone(
    segments: list[Segment],
    zone: ProblemZone,
    buffer_seconds: float = 10.0,
) -> list[Segment]:
    """
    Find all segments that fall within a problem zone's time range.
    
    Args:
        segments: All segments
        zone: The problem zone
        buffer_seconds: Extra time to include at boundaries
    
    Returns:
        List of segments in the zone's time range
    """
    zone_start = zone.start_time - buffer_seconds
    zone_end = zone.end_time + buffer_seconds
    
    return [s for s in segments if s.start >= zone_start and s.end <= zone_end]


def realign_zone(
    zone: ProblemZone,
    zone_segments: list[Segment],
    ayahs: list[Ayah],
) -> list[AlignmentResult]:
    """
    Re-align a problem zone using the DP aligner.
    
    Args:
        zone: The problem zone
        zone_segments: Segments in the zone's time range
        ayahs: All ayahs (will filter to zone's ayahs)
    
    Returns:
        New alignment results for the zone
    """
    # Get the ayahs for this zone (1-indexed)
    zone_ayahs = [a for a in ayahs if zone.start_ayah <= a.ayah_number <= zone.end_ayah]
    
    if not zone_ayahs or not zone_segments:
        return []
    
    # Run DP alignment on this small zone
    results = align_segments_dp(zone_segments, zone_ayahs)
    
    return results


def realign_problem_zones(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    min_consecutive: int = 3,
    quality_threshold: float = 0.85,
    buffer_seconds: float = 10.0,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Re-align problem zones in the results to fix timing drift.

    This is the MAIN ENTRY POINT for zone-based drift correction.

    How it works:
    1. Scan through results to find "problem zones" (consecutive low-conf ayahs)
    2. For each zone:
       a. Extract segments in that time range (with buffer)
       b. Extract ayahs in that zone
       c. Re-run DP alignment on just that small region
       d. Compare old vs new results for each ayah
       e. Keep whichever result is better
    3. Return updated results with statistics

    Example:
    ```
    Initial results: [0.95, 0.92, 0.78, 0.75, 0.72, 0.91, 0.94]
                                    ↑____________↑
                                    Problem zone (ayahs 3-5)

    After re-alignment: [0.95, 0.92, 0.89, 0.87, 0.86, 0.91, 0.94]
                                         ↑____________↑
                                         Improved!
    ```

    Args:
        results: Initial alignment results (potentially with drift)
        segments: All transcribed segments (needed for re-alignment)
        ayahs: All reference ayahs (needed for re-alignment)
        min_consecutive: Min consecutive low-conf ayahs to form a zone (default 3)
        quality_threshold: Similarity below which is "low confidence" (default 0.85)
        buffer_seconds: Extra time to include at zone boundaries for context (default 10s)

    Returns:
        Tuple of:
        - Updated alignment results (with improved zones)
        - ZoneStats object with statistics about what was improved
    """
    stats = ZoneStats()

    # ============================================================================
    # STEP 1: Identify Problem Zones
    # ============================================================================
    # Scan through all results to find consecutive low-confidence ayahs
    zones = identify_problem_zones(results, min_consecutive, quality_threshold)
    stats.zones_found = len(zones)

    # If no problem zones, return original results unchanged
    if not zones:
        return results, stats

    # Create a mutable copy of results (we'll update it in-place)
    new_results = list(results)

    # Create quick lookup table: ayah_number -> Ayah object
    ayah_by_num = {a.ayah_number: a for a in ayahs}

    # ============================================================================
    # STEP 2: Re-align Each Problem Zone
    # ============================================================================
    for zone in zones:
        # --- Extract segments for this zone (with buffer for context) ---
        zone_segments = find_segments_for_zone(segments, zone, buffer_seconds)

        # Safety check: If we don't have enough segments, try a larger buffer
        # This can happen if the zone boundaries are tight
        if len(zone_segments) < (zone.end_ayah - zone.start_ayah + 1):
            zone_segments = find_segments_for_zone(segments, zone, buffer_seconds * 3)

        if not zone_segments:
            # Still no segments - skip this zone
            continue

        # --- Extract ayahs for this zone ---
        zone_ayahs = [ayah_by_num[n] for n in range(zone.start_ayah, zone.end_ayah + 1)
                      if n in ayah_by_num]

        # --- Re-run DP alignment on just this small region ---
        # This is much faster than re-aligning the entire surah
        new_zone_results = align_segments_dp(zone_segments, zone_ayahs)

        if not new_zone_results:
            # Re-alignment failed - skip this zone
            continue

        # ============================================================================
        # STEP 3: Compare Old vs New Results (Keep Better)
        # ============================================================================
        zone_improved = False
        new_by_ayah = {r.ayah.ayah_number: r for r in new_zone_results}

        # For each ayah in the zone, compare old vs new
        for i in range(zone.start_idx, zone.end_idx):
            if i >= len(new_results):
                break

            old_result = new_results[i]
            ayah_num = old_result.ayah.ayah_number

            if ayah_num in new_by_ayah:
                new_result = new_by_ayah[ayah_num]

                # Keep whichever result has higher similarity
                if new_result.similarity_score > old_result.similarity_score:
                    new_results[i] = new_result
                    stats.ayahs_improved += 1
                    zone_improved = True
                elif new_result.similarity_score < old_result.similarity_score:
                    # New result is worse - keep old result
                    stats.ayahs_degraded += 1
                else:
                    # Same score - keep old result
                    stats.ayahs_unchanged += 1

        if zone_improved:
            stats.zones_improved += 1

    return new_results, stats


def find_anchors(
    results: list[AlignmentResult],
    min_similarity: float = 0.95,
    min_wps: float = 0.8,
    max_wps: float = 2.0,
) -> list[tuple[int, AlignmentResult]]:
    """
    Find anchor points - high-confidence ayahs with normal recitation pace.

    Anchors are ayahs we're very confident about. We use them as reference
    points and re-align the regions between them.

    What makes a good anchor?
    1. High similarity (≥95%) - we're confident the timing is correct
    2. Normal WPS (words per second) - not too fast or too slow
       - Too fast (>2 WPS): might be compressed/rushed
       - Too slow (<0.8 WPS): might have pauses or timing issues

    Args:
        results: List of alignment results
        min_similarity: Minimum similarity to be an anchor (default 0.95)
        min_wps: Minimum words per second (default 0.8)
        max_wps: Maximum words per second (default 2.0)

    Returns:
        List of (index, result) tuples for anchor points
    """
    anchors = []
    for i, r in enumerate(results):
        words = len(r.ayah.text.split())
        duration = r.end_time - r.start_time
        wps = words / duration if duration > 0 else 0
        
        if r.similarity_score >= min_similarity and min_wps <= wps <= max_wps:
            anchors.append((i, r))
    
    return anchors


def realign_from_anchors(
    results: list[AlignmentResult],
    segments: list[Segment],
    ayahs: list[Ayah],
    min_gap_size: int = 3,
    buffer_seconds: float = 5.0,
) -> tuple[list[AlignmentResult], ZoneStats]:
    """
    Re-align regions between anchor points (anchor-based drift correction).

    This is a SECOND PASS after problem zone re-alignment. It's more aggressive
    and uses a different strategy:

    STRATEGY:
    1. Find "anchor" ayahs (very high confidence, normal pace)
    2. Identify gaps between anchors (3+ ayahs)
    3. Re-align each gap independently
    4. Keep better results

    Example:
    ```
    Ayah 1:  0.97 ← Anchor
    Ayah 2:  0.82
    Ayah 3:  0.79  ← Gap (low confidence)
    Ayah 4:  0.81
    Ayah 5:  0.96 ← Anchor
    Ayah 6:  0.75
    Ayah 7:  0.98 ← Anchor
    ```

    Gaps identified: [2-4] between anchors 1 and 5
    We re-align ayahs 2-4 using segments in that time range.

    This complements problem zone re-alignment by:
    - Using different criteria (anchors vs consecutive low-conf)
    - Being more targeted (gaps between known-good points)
    - Catching drift that doesn't form long consecutive zones

    Args:
        results: Alignment results (ideally after problem zone re-alignment)
        segments: All transcribed segments
        ayahs: All reference ayahs
        min_gap_size: Minimum gap size to re-align (default 3 ayahs)
        buffer_seconds: Extra time at boundaries (default 5s)

    Returns:
        Tuple of (updated results, statistics)
    """
    stats = ZoneStats()
    
    # Find anchors
    anchors = find_anchors(results)
    
    if len(anchors) < 2:
        return results, stats
    
    # Create mutable copy
    new_results = list(results)
    ayah_by_num = {a.ayah_number: a for a in ayahs}
    
    # Find gaps between anchors
    gaps = []
    prev_idx = -1
    prev_result = None
    
    for idx, result in anchors:
        if prev_idx >= 0:
            gap_size = idx - prev_idx - 1
            if gap_size >= min_gap_size:
                gaps.append({
                    'start_idx': prev_idx + 1,
                    'end_idx': idx,
                    'start_time': prev_result.end_time - buffer_seconds,
                    'end_time': result.start_time + buffer_seconds,
                })
        prev_idx = idx
        prev_result = result
    
    # Check for gap at the end (after last anchor to end of surah)
    if anchors:
        last_idx, last_result = anchors[-1]
        if len(results) - last_idx - 1 >= min_gap_size:
            gaps.append({
                'start_idx': last_idx + 1,
                'end_idx': len(results),
                'start_time': last_result.end_time - buffer_seconds,
                'end_time': segments[-1].end + buffer_seconds if segments else 0,
            })
    
    stats.zones_found = len(gaps)
    
    # Re-align each gap
    for gap in gaps:
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']
        time_start = max(0, gap['start_time'])
        time_end = gap['end_time']
        
        # Get segments and ayahs for this gap
        gap_segments = [s for s in segments if s.start >= time_start and s.end <= time_end]
        gap_ayah_nums = [new_results[i].ayah.ayah_number for i in range(start_idx, min(end_idx, len(new_results)))]
        gap_ayahs = [ayah_by_num[n] for n in gap_ayah_nums if n in ayah_by_num]
        
        if not gap_segments or not gap_ayahs:
            continue
        
        # Re-align
        new_gap_results = align_segments_dp(gap_segments, gap_ayahs)
        
        if not new_gap_results:
            continue
        
        # Update results where improved
        new_by_ayah = {r.ayah.ayah_number: r for r in new_gap_results}
        gap_improved = False
        
        for i in range(start_idx, min(end_idx, len(new_results))):
            old_result = new_results[i]
            ayah_num = old_result.ayah.ayah_number
            
            if ayah_num in new_by_ayah:
                new_result = new_by_ayah[ayah_num]
                
                if new_result.similarity_score > old_result.similarity_score + 0.01:
                    new_results[i] = new_result
                    stats.ayahs_improved += 1
                    gap_improved = True
                elif new_result.similarity_score < old_result.similarity_score - 0.01:
                    stats.ayahs_degraded += 1
                else:
                    stats.ayahs_unchanged += 1
        
        if gap_improved:
            stats.zones_improved += 1
    
    return new_results, stats


def fix_overlaps(results: list[AlignmentResult]) -> int:
    """
    Fix any overlapping ayah timings by adjusting boundaries.
    
    Returns number of overlaps fixed.
    """
    if len(results) < 2:
        return 0
    
    # Sort by ayah number
    results.sort(key=lambda r: r.ayah.ayah_number)
    
    fixes = 0
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        if curr.start_time < prev.end_time:
            # Fix by meeting at midpoint
            midpoint = (prev.end_time + curr.start_time) / 2
            
            # Update timing (create new results since AlignmentResult might be immutable)
            results[i-1] = AlignmentResult(
                ayah=prev.ayah,
                start_time=prev.start_time,
                end_time=round(midpoint - 0.05, 2),
                transcribed_text=prev.transcribed_text,
                similarity_score=prev.similarity_score,
                overlap_detected=prev.overlap_detected,
            )
            results[i] = AlignmentResult(
                ayah=curr.ayah,
                start_time=round(midpoint + 0.05, 2),
                end_time=curr.end_time,
                transcribed_text=curr.transcribed_text,
                similarity_score=curr.similarity_score,
                overlap_detected=curr.overlap_detected,
            )
            fixes += 1
    
    return fixes


# Export for use in other modules
__all__ = [
    'realign_problem_zones',
    'identify_problem_zones',
    'realign_from_anchors',
    'find_anchors',
    'fix_overlaps',
    'ProblemZone',
    'ZoneStats',
]
