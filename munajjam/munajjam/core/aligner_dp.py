"""
Dynamic Programming based aligner for Quran audio segments.

Uses a DTW-like approach to find the optimal alignment between
transcribed segments and reference ayahs, allowing multiple segments
to be merged into a single ayah.
"""

from dataclasses import dataclass
from typing import Callable

from ..models import Ayah, Segment, AlignmentResult
from .matcher import similarity, compute_coverage_ratio
from .arabic import normalize_arabic


@dataclass
class DPCell:
    """A cell in the DP matrix."""
    cost: float  # Accumulated cost to reach this cell
    merged_text: str  # Text merged so far for current ayah
    seg_start_idx: int  # Index of first segment in current ayah
    parent: tuple[int, int] | None  # Previous cell for backtracking


def compute_alignment_cost(
    merged_text: str, 
    ayah_text: str,
    critical_threshold: float = 0.4,
    critical_penalty: float = 0.5,
) -> float:
    """
    Compute the cost of aligning merged segments to an ayah.
    Lower cost = better alignment.
    
    Returns value between 0 (perfect match) and higher values for worse matches.
    
    Args:
        merged_text: The merged segment text
        ayah_text: The reference ayah text
        critical_threshold: Similarity below this triggers heavy penalty
        critical_penalty: Extra penalty added when below critical threshold
    """
    if not merged_text.strip():
        return 1.5  # High cost for empty text
    
    # Primary: text similarity
    sim = similarity(merged_text, ayah_text)
    
    # Secondary: coverage ratio penalty
    coverage = compute_coverage_ratio(merged_text, ayah_text)
    
    # Penalize under-coverage (missing words) and over-coverage (extra words)
    if coverage < 0.7:
        coverage_penalty = (0.7 - coverage) * 0.5  # Up to 0.35 penalty
    elif coverage > 1.3:
        coverage_penalty = (coverage - 1.3) * 0.3  # Penalty for too much text
    else:
        coverage_penalty = 0
    
    # Cost = 1 - similarity + coverage penalty
    cost = (1 - sim) + coverage_penalty
    
    # CRITICAL THRESHOLD: Heavy penalty for very low similarity
    # This discourages the DP from choosing paths with misaligned ayahs
    if sim < critical_threshold:
        cost += critical_penalty * (critical_threshold - sim) / critical_threshold
    
    return max(0.0, cost)


def _align_greedy_multi_ayah(
    segments: list[Segment],
    ayahs: list[Ayah],
) -> list[AlignmentResult]:
    """
    Bidirectional greedy alignment for when segments != ayahs.
    
    Handles two cases:
    1. Reciter merges multiple ayahs into one segment (segment > ayahs)
    2. Reciter splits one ayah across multiple segments (ayah > segments)
    
    Strategy: Try both merging segments and merging ayahs to find best match.
    """
    if not segments or not ayahs:
        return []
    
    results = []
    seg_idx = 0
    ayah_idx = 0
    
    while seg_idx < len(segments) and ayah_idx < len(ayahs):
        best_match_ayahs = None
        best_match_segs = None
        best_score = 0
        best_seg_end = seg_idx
        best_ayah_end = ayah_idx
        
        # Try different combinations:
        # 1. Single segment -> single ayah
        # 2. Single segment -> multiple ayahs (merged recitation)
        # 3. Multiple segments -> single ayah (split recitation)
        # 4. Multiple segments -> multiple ayahs
        
        # Search window for ayahs (look ahead)
        ayah_search_limit = min(ayah_idx + 4, len(ayahs))
        seg_search_limit = min(seg_idx + 4, len(segments))
        
        for num_segs in range(1, seg_search_limit - seg_idx + 1):
            merged_segs = segments[seg_idx:seg_idx + num_segs]
            merged_seg_text = " ".join(s.text for s in merged_segs)
            
            for num_ayahs in range(1, ayah_search_limit - ayah_idx + 1):
                concat_ayahs = ayahs[ayah_idx:ayah_idx + num_ayahs]
                concat_text = " ".join(a.text for a in concat_ayahs)
                
                score = similarity(merged_seg_text, concat_text)
                
                # Prefer matches that consume more content (closer to 1:1 ratio)
                # Penalize very unbalanced matches
                ratio_penalty = 0
                if num_segs > 1 and num_ayahs > 1:
                    ratio_penalty = 0.05  # Small penalty for complex merges
                
                adjusted_score = score - ratio_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match_ayahs = concat_ayahs
                    best_match_segs = merged_segs
                    best_seg_end = seg_idx + num_segs
                    best_ayah_end = ayah_idx + num_ayahs
        
        # Apply match if score is acceptable
        if best_match_ayahs and best_score > 0.35:
            # Calculate timing
            start_time = best_match_segs[0].start
            end_time = best_match_segs[-1].end
            merged_text = " ".join(s.text for s in best_match_segs)
            
            # Distribute time across matched ayahs proportionally
            total_words = sum(len(a.text.split()) for a in best_match_ayahs)
            current_time = start_time
            segment_duration = end_time - start_time
            
            for a in best_match_ayahs:
                ayah_words = len(a.text.split())
                ayah_duration = (ayah_words / total_words) * segment_duration if total_words > 0 else segment_duration / len(best_match_ayahs)
                
                result = AlignmentResult(
                    ayah=a,
                    start_time=round(current_time, 2),
                    end_time=round(current_time + ayah_duration, 2),
                    transcribed_text=merged_text if len(best_match_ayahs) == 1 else f"[{len(best_match_segs)} segs -> {len(best_match_ayahs)} ayahs]",
                    similarity_score=best_score,
                    overlap_detected=len(best_match_ayahs) > 1 or len(best_match_segs) > 1,
                )
                results.append(result)
                current_time += ayah_duration
            
            seg_idx = best_seg_end
            ayah_idx = best_ayah_end
        else:
            # No good match - advance segment index to try next segment
            seg_idx += 1
    
    return results


def align_segments_dp(
    segments: list[Segment],
    ayahs: list[Ayah],
    max_segments_per_ayah: int = 15,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    Align segments to ayahs using dynamic programming.
    
    Finds the globally optimal alignment that minimizes total cost,
    allowing multiple segments to be merged into each ayah.
    
    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        max_segments_per_ayah: Maximum segments that can be merged into one ayah
        on_progress: Optional callback for progress updates
    
    Returns:
        List of AlignmentResult objects
    """
    if not segments or not ayahs:
        return []
    
    # Filter out non-ayah segments (isti3aza, basmala for non-Fatiha)
    from ..models import SegmentType
    filtered_segments = []
    for seg in segments:
        # Skip isti3aza
        if seg.type == SegmentType.ISTI3AZA:
            continue
        # Skip basmala for surahs other than Al-Fatiha (surah 1)
        if seg.type == SegmentType.BASMALA and ayahs[0].surah_id != 1:
            continue
        filtered_segments.append(seg)
    
    segments = filtered_segments if filtered_segments else segments
    
    n_seg = len(segments)
    n_ayah = len(ayahs)
    
    # DP table: dp[i][j] = best way to align segments[0:i] to ayahs[0:j]
    # i = number of segments consumed (0 to n_seg)
    # j = number of ayahs aligned (0 to n_ayah)
    INF = float('inf')
    
    dp: dict[tuple[int, int], DPCell] = {}
    dp[(0, 0)] = DPCell(cost=0, merged_text="", seg_start_idx=0, parent=None)
    
    # Fill DP table
    for j in range(1, n_ayah + 1):  # For each ayah
        ayah = ayahs[j - 1]
        
        if on_progress:
            on_progress(j, n_ayah)
        
        for i in range(j, n_seg + 1):  # Need at least j segments for j ayahs
            best_cell = None
            best_cost = INF
            
            # Try merging k segments (1, 2, ..., max) into ayah j
            for k in range(1, min(max_segments_per_ayah, i - j + 2)):
                prev_i = i - k
                prev_j = j - 1
                
                if (prev_i, prev_j) not in dp:
                    continue
                
                prev_cell = dp[(prev_i, prev_j)]
                
                # Merge segments [prev_i : i] into ayah j
                merged_segments = segments[prev_i:i]
                merged_text = " ".join(seg.text for seg in merged_segments)
                
                # Compute alignment cost
                cost = compute_alignment_cost(merged_text, ayah.text)
                total_cost = prev_cell.cost + cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cell = DPCell(
                        cost=total_cost,
                        merged_text=merged_text,
                        seg_start_idx=prev_i,
                        parent=(prev_i, prev_j),
                    )
            
            if best_cell is not None:
                dp[(i, j)] = best_cell
    
    # Find best ending point (should be at n_ayah ayahs)
    # We might not use all segments if there's extra content
    best_end = None
    best_end_cost = INF
    
    for i in range(n_ayah, n_seg + 1):
        if (i, n_ayah) in dp:
            if dp[(i, n_ayah)].cost < best_end_cost:
                best_end_cost = dp[(i, n_ayah)].cost
                best_end = (i, n_ayah)
    
    if best_end is None:
        # Fallback: find the best partial alignment
        for j in range(n_ayah, 0, -1):
            for i in range(n_seg, 0, -1):
                if (i, j) in dp:
                    best_end = (i, j)
                    break
            if best_end:
                break
    
    if best_end is None:
        return []
    
    # Backtrack to reconstruct alignment
    path = []
    current = best_end
    
    while current and current in dp:
        cell = dp[current]
        i, j = current
        
        if cell.parent is not None:
            prev_i, prev_j = cell.parent
            # This cell represents aligning segments[prev_i:i] to ayah j
            path.append((prev_i, i, j, cell.merged_text))
        
        current = cell.parent
    
    path.reverse()
    
    # Convert path to AlignmentResult objects
    results = []
    for seg_start, seg_end, ayah_idx, merged_text in path:
        if seg_start >= len(segments) or seg_end > len(segments):
            continue
            
        ayah = ayahs[ayah_idx - 1]
        start_time = segments[seg_start].start
        end_time = segments[seg_end - 1].end
        
        sim_score = similarity(merged_text, ayah.text)
        
        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        results.append(result)
    
    return results


def _find_cascade_sequences(
    results: list[AlignmentResult],
    threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[tuple[int, int]]:
    """
    Find sequences of consecutive low-scoring ayahs (cascades).
    
    Args:
        results: List of alignment results
        threshold: Similarity threshold below which is considered low
        min_cascade_length: Minimum length to be considered a cascade
    
    Returns:
        List of (start_idx, end_idx) tuples for each cascade
    """
    cascades = []
    i = 0
    
    while i < len(results):
        if results[i].similarity_score < threshold:
            # Start of potential cascade
            start = i
            while i < len(results) and results[i].similarity_score < threshold:
                i += 1
            end = i
            
            if end - start >= min_cascade_length:
                cascades.append((start, end))
        else:
            i += 1
    
    return cascades


def _recover_cascade_with_resynce(
    segments: list["Segment"],
    ayahs: list["Ayah"],
    results: list[AlignmentResult],
    cascade_start: int,
    cascade_end: int,
    silences_sec: list[tuple[float, float]],
    context_ayahs: int = 1,  # Number of ayahs before/after to include for context
) -> list[AlignmentResult] | None:
    """
    Attempt to recover a cascade by re-aligning using silence boundaries.
    
    Strategy:
    1. Extend the cascade range by 1 ayah on each side for context
    2. Find the segment range covered by these ayahs
    3. Find silence boundaries within this range
    4. Re-run DP alignment on just this portion with emphasis on silence breaks
    
    Returns:
        New alignment results for the cascade region, or None if recovery failed
    """
    # Extend range for context
    extended_start = max(0, cascade_start - context_ayahs)
    extended_end = min(len(results), cascade_end + context_ayahs)
    
    # Get segment range for the extended ayah range
    seg_start_time = results[extended_start].start_time
    seg_end_time = results[extended_end - 1].end_time
    
    # Find segments in this time range
    seg_indices = []
    for idx, seg in enumerate(segments):
        if seg.start >= seg_start_time - 0.5 and seg.end <= seg_end_time + 0.5:
            seg_indices.append(idx)
    
    if not seg_indices:
        return None
    
    seg_range_start = min(seg_indices)
    seg_range_end = max(seg_indices) + 1
    
    # Extract the segments and ayahs for re-alignment
    sub_segments = segments[seg_range_start:seg_range_end]
    sub_ayahs = [results[i].ayah for i in range(extended_start, extended_end)]
    
    if len(sub_segments) < len(sub_ayahs):
        return None  # Can't recover without enough segments
    
    # Find silences in this time range
    relevant_silences = []
    for sil_start, sil_end in silences_sec:
        if seg_start_time <= sil_start <= seg_end_time:
            relevant_silences.append((sil_start, sil_end))
    
    # Re-run DP alignment on this sub-problem with stricter constraints
    # Use silence boundaries as preferred break points
    
    n_sub_seg = len(sub_segments)
    n_sub_ayah = len(sub_ayahs)
    
    INF = float('inf')
    dp: dict[tuple[int, int], tuple[float, str, int, tuple | None]] = {}
    dp[(0, 0)] = (0.0, "", 0, None)  # (cost, merged_text, seg_start, parent)
    
    # Build a set of segment indices that align with silences
    silence_aligned_ends = set()
    for idx, seg in enumerate(sub_segments):
        for sil_start, sil_end in relevant_silences:
            # Segment end aligns with silence start (within 0.3s tolerance)
            if abs(seg.end - sil_start) < 0.3:
                silence_aligned_ends.add(idx + 1)  # +1 because we use exclusive end
    
    max_segs = min(6, n_sub_seg)  # Limit merge attempts in recovery
    
    for j in range(1, n_sub_ayah + 1):
        for i in range(j, n_sub_seg + 1):
            best = None
            best_cost = INF
            
            for k in range(1, min(max_segs, i) + 1):
                prev_i = i - k
                prev_j = j - 1
                
                if (prev_i, prev_j) not in dp:
                    continue
                
                prev_cost, _, _, _ = dp[(prev_i, prev_j)]
                
                merged_text = " ".join(seg.text for seg in sub_segments[prev_i:i])
                cost = compute_alignment_cost(merged_text, sub_ayahs[j - 1].text)
                
                # Bonus for ending at silence boundary
                if i in silence_aligned_ends:
                    cost -= 0.15  # Reward silence-aligned boundaries
                
                total_cost = prev_cost + cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best = (total_cost, merged_text, prev_i, (prev_i, prev_j))
            
            if best is not None:
                dp[(i, j)] = best
    
    # Find best ending
    best_end = None
    best_end_cost = INF
    
    for i in range(n_sub_ayah, n_sub_seg + 1):
        if (i, n_sub_ayah) in dp:
            if dp[(i, n_sub_ayah)][0] < best_end_cost:
                best_end_cost = dp[(i, n_sub_ayah)][0]
                best_end = (i, n_sub_ayah)
    
    if best_end is None:
        return None
    
    # Backtrack and build new results
    path = []
    current = best_end
    
    while current and current in dp:
        cost, merged_text, seg_start, parent = dp[current]
        i, j = current
        
        if parent is not None:
            path.append((seg_start, i, j, merged_text))
        
        current = parent
    
    path.reverse()
    
    # Convert to results
    new_results = []
    for seg_start_idx, seg_end_idx, ayah_idx, merged_text in path:
        if seg_start_idx >= len(sub_segments) or seg_end_idx > len(sub_segments):
            continue
        
        ayah = sub_ayahs[ayah_idx - 1]
        start_time = sub_segments[seg_start_idx].start
        end_time = sub_segments[seg_end_idx - 1].end
        
        sim_score = similarity(merged_text, ayah.text)
        
        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        new_results.append(result)
    
    # Check if recovery improved the results
    if len(new_results) != extended_end - extended_start:
        return None  # Didn't align all ayahs
    
    old_results_range = results[extended_start:extended_end]
    
    # Conservative check: Don't accept recovery if ANY ayah degrades significantly
    for old, new in zip(old_results_range, new_results):
        drop = old.similarity_score - new.similarity_score
        
        # Strict protection for good ayahs (>= 75%): max 8% drop
        if old.similarity_score >= 0.75 and drop > 0.08:
            return None
        
        # Protect mediocre ayahs (50-75%): max 12% drop
        if old.similarity_score >= 0.5 and drop > 0.12:
            return None
        
        # Never let a good ayah (>=75%) drop below 70%
        if old.similarity_score >= 0.75 and new.similarity_score < 0.70:
            return None
    
    # Check overall improvement in the cascade region (not including context)
    # Focus on the actual cascade, not the context ayahs
    context = 1
    cascade_old_start = max(0, context)
    cascade_old_end = min(len(old_results_range), len(old_results_range) - context) if len(old_results_range) > 2 else len(old_results_range)
    
    cascade_new_start = cascade_old_start
    cascade_new_end = cascade_old_end
    
    old_cascade_sim = sum(r.similarity_score for r in old_results_range[cascade_old_start:cascade_old_end])
    new_cascade_sim = sum(r.similarity_score for r in new_results[cascade_new_start:cascade_new_end])
    
    cascade_len = cascade_old_end - cascade_old_start
    if cascade_len == 0:
        return None
    
    old_avg = old_cascade_sim / cascade_len
    new_avg = new_cascade_sim / cascade_len
    
    # Require significant improvement in the cascade region
    if new_avg > old_avg + 0.08:  # Require 8% improvement in cascade region
        return new_results
    
    return None


def _apply_cascade_recovery(
    segments: list["Segment"],
    ayahs: list["Ayah"],
    results: list[AlignmentResult],
    silences_ms: list[tuple[int, int]] | None = None,
    cascade_threshold: float = 0.7,
    min_cascade_length: int = 2,
) -> list[AlignmentResult]:
    """
    Post-process alignment results to recover cascaded failures.
    
    Detects sequences of consecutive low-scoring ayahs and attempts
    to re-align them using silence boundaries for better sync.
    
    Args:
        segments: Original segments
        ayahs: Original ayahs
        results: Initial alignment results
        silences_ms: Silence periods in milliseconds
        cascade_threshold: Similarity below which triggers cascade detection
        min_cascade_length: Minimum consecutive failures to be a cascade
    
    Returns:
        Improved alignment results
    """
    if not results:
        return results
    
    # Convert silences to seconds
    silences_sec = []
    if silences_ms:
        for start_ms, end_ms in silences_ms:
            silences_sec.append((start_ms / 1000.0, end_ms / 1000.0))
    
    # Find cascades
    cascades = _find_cascade_sequences(results, cascade_threshold, min_cascade_length)
    
    if not cascades:
        return results  # No cascades to recover
    
    # Process each cascade (in reverse order to maintain indices)
    improved_results = list(results)
    
    for cascade_start, cascade_end in reversed(cascades):
        recovery = _recover_cascade_with_resynce(
            segments,
            ayahs,
            improved_results,
            cascade_start,
            cascade_end,
            silences_sec,
        )
        
        if recovery:
            # Calculate the extended range that was recovered
            context = 1
            ext_start = max(0, cascade_start - context)
            ext_end = min(len(improved_results), cascade_end + context)
            
            # Replace the recovered range
            improved_results = (
                improved_results[:ext_start] +
                recovery +
                improved_results[ext_end:]
            )
    
    return improved_results


def align_segments_dp_with_constraints(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    max_segments_per_ayah: int = 8,  # Reduced from 15
    silence_bonus: float = 0.1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[AlignmentResult]:
    """
    DP alignment with silence constraints (OPTIMIZED).
    
    Optimizations:
    - Pre-compute all merged texts
    - Cache similarity computations
    - Limit search window (beam search style)
    - Skip impossible paths early
    
    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        silences_ms: List of (start_ms, end_ms) silence periods
        max_segments_per_ayah: Maximum segments per ayah
        silence_bonus: Cost reduction when boundary aligns with silence
        on_progress: Optional progress callback
    
    Returns:
        List of AlignmentResult objects
    """
    if not segments or not ayahs:
        return []
    
    # Filter out non-ayah segments (isti3aza, basmala for non-Fatiha)
    from ..models import SegmentType
    filtered_segments = []
    for seg in segments:
        if seg.type == SegmentType.ISTI3AZA:
            continue
        if seg.type == SegmentType.BASMALA and ayahs[0].surah_id != 1:
            continue
        filtered_segments.append(seg)
    
    segments = filtered_segments if filtered_segments else segments
    
    n_seg = len(segments)
    n_ayah = len(ayahs)
    
    # Handle case where we have fewer segments than ayahs
    # This happens when reciter merges multiple ayahs into one segment
    # Use greedy when segments < ayahs (DP requires segments >= ayahs)
    if n_seg < n_ayah:
        # Fall back to greedy alignment that allows multiple ayahs per segment
        return _align_greedy_multi_ayah(segments, ayahs)
    
    # OPTIMIZATION 1: Pre-compute merged texts for all possible segment ranges
    # This avoids re-joining strings repeatedly
    merged_cache: dict[tuple[int, int], str] = {}
    
    def get_merged_text(start: int, end: int) -> str:
        if (start, end) not in merged_cache:
            merged_cache[(start, end)] = " ".join(seg.text for seg in segments[start:end])
        return merged_cache[(start, end)]
    
    # OPTIMIZATION 2: Cache similarity computations
    sim_cache: dict[tuple[str, int], float] = {}
    
    def get_cost(merged_text: str, ayah_idx: int) -> float:
        cache_key = (merged_text[:100], ayah_idx)  # Truncate for cache key
        if cache_key not in sim_cache:
            sim_cache[cache_key] = compute_alignment_cost(merged_text, ayahs[ayah_idx].text)
        return sim_cache[cache_key]
    
    INF = float('inf')
    dp: dict[tuple[int, int], DPCell] = {}
    dp[(0, 0)] = DPCell(cost=0, merged_text="", seg_start_idx=0, parent=None)
    
    # Calculate expected segments per ayah ratio for better windowing
    avg_seg_per_ayah = n_seg / n_ayah
    
    for j in range(1, n_ayah + 1):
        if on_progress:
            on_progress(j, n_ayah)
        
        # RELAXED WINDOWING: Allow more flexibility in segment distribution
        # Instead of strict 1:1 minimum, use a sliding window based on average ratio
        # This handles cases where some ayahs use more/fewer segments than average
        
        # Minimum: at least j segments, but allow some slack
        min_i = max(1, j)
        
        # Maximum: use all remaining segments minus minimum for remaining ayahs
        # But add buffer to handle variable segment distribution
        remaining_ayahs = n_ayah - j
        min_segs_needed_for_remaining = max(1, remaining_ayahs)  # At least 1 seg per remaining ayah
        max_i = min(n_seg + 1, n_seg - min_segs_needed_for_remaining + 1 + max_segments_per_ayah)
        
        # Ensure valid range
        max_i = max(min_i + 1, max_i)
        
        for i in range(min_i, max_i):
            best_cell = None
            best_cost = INF
            
            # Limit merge attempts: allow merging up to max_segments_per_ayah
            # Don't overly restrict based on i-j difference
            max_k = min(max_segments_per_ayah, i)
            
            for k in range(1, max_k + 1):  # +1 to include max_k
                prev_i = i - k
                prev_j = j - 1
                
                if (prev_i, prev_j) not in dp:
                    continue
                
                prev_cell = dp[(prev_i, prev_j)]
                
                # Early pruning: skip if already too costly
                if prev_cell.cost > best_cost:
                    continue
                
                merged_text = get_merged_text(prev_i, i)
                cost = get_cost(merged_text, j - 1)
                
                total_cost = prev_cell.cost + cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cell = DPCell(
                        cost=total_cost,
                        merged_text=merged_text,
                        seg_start_idx=prev_i,
                        parent=(prev_i, prev_j),
                    )
            
            if best_cell is not None:
                dp[(i, j)] = best_cell
    
    # Find best ending
    best_end = None
    best_end_cost = INF
    
    for i in range(n_ayah, n_seg + 1):
        if (i, n_ayah) in dp:
            if dp[(i, n_ayah)].cost < best_end_cost:
                best_end_cost = dp[(i, n_ayah)].cost
                best_end = (i, n_ayah)
    
    if best_end is None:
        for j in range(n_ayah, 0, -1):
            for i in range(n_seg, 0, -1):
                if (i, j) in dp:
                    best_end = (i, j)
                    break
            if best_end:
                break
    
    if best_end is None:
        return []
    
    # Backtrack
    path = []
    current = best_end
    
    while current and current in dp:
        cell = dp[current]
        i, j = current
        
        if cell.parent is not None:
            prev_i, prev_j = cell.parent
            path.append((prev_i, i, j, cell.merged_text))
        
        current = cell.parent
    
    path.reverse()
    
    # Convert to results
    results = []
    for seg_start, seg_end, ayah_idx, merged_text in path:
        if seg_start >= len(segments) or seg_end > len(segments):
            continue
            
        ayah = ayahs[ayah_idx - 1]
        start_time = segments[seg_start].start
        end_time = segments[seg_end - 1].end
        
        sim_score = similarity(merged_text, ayah.text)
        
        result = AlignmentResult(
            ayah=ayah,
            start_time=start_time,
            end_time=end_time,
            transcribed_text=merged_text,
            similarity_score=sim_score,
            overlap_detected=False,
        )
        results.append(result)
    
    # POST-PROCESSING: Cascade Recovery
    # Detect sequences of consecutive low-scoring ayahs and attempt re-alignment
    results = _apply_cascade_recovery(
        segments=segments,
        ayahs=ayahs,
        results=results,
        silences_ms=silences_ms,
        cascade_threshold=0.7,
        min_cascade_length=2,
    )
    
    return results


def apply_hybrid_fallback(
    new_results: list[AlignmentResult],
    original_results: list[dict],
    threshold: float = 0.8,
) -> list[AlignmentResult]:
    """
    Apply hybrid fallback: for ayahs scoring below threshold in new results,
    use original results if they were better.
    
    Args:
        new_results: Results from the new alignment algorithm
        original_results: Original ayah results from file (list of dicts with 'similarity', 'start', 'end')
        threshold: Similarity threshold below which to consider fallback
    
    Returns:
        Merged results taking the best of both
    """
    if len(new_results) != len(original_results):
        return new_results  # Can't merge if lengths don't match
    
    for i, (new_r, orig) in enumerate(zip(new_results, original_results)):
        orig_sim = orig.get("similarity", 0)
        
        # If new result is below threshold and original was better, use original timing
        if new_r.similarity_score < threshold and orig_sim > new_r.similarity_score:
            # Update the result with original timing but keep the ayah reference
            new_results[i] = AlignmentResult(
                ayah=new_r.ayah,
                start_time=orig.get("start", new_r.start_time),
                end_time=orig.get("end", new_r.end_time),
                transcribed_text=new_r.transcribed_text,  # Keep transcribed text
                similarity_score=orig_sim,  # Use original similarity
                overlap_detected=new_r.overlap_detected,
            )
    
    return new_results


@dataclass
class HybridStats:
    """Statistics from hybrid alignment."""
    total_ayahs: int = 0
    dp_kept: int = 0  # High quality from DP, kept as-is
    old_fallback: int = 0  # Fell back to old aligner
    split_improved: int = 0  # Improved via split-and-restitch
    still_low: int = 0  # Remained low quality after all attempts
    
    def __str__(self) -> str:
        return (
            f"HybridStats(total={self.total_ayahs}, dp_kept={self.dp_kept}, "
            f"old_fallback={self.old_fallback}, split_improved={self.split_improved}, "
            f"still_low={self.still_low})"
        )


def _find_silences_in_range(
    silences_sec: list[tuple[float, float]],
    start_time: float,
    end_time: float,
    min_duration: float = 0.2,
) -> list[tuple[float, float]]:
    """Find silence periods within a given time range."""
    result = []
    for sil_start, sil_end in silences_sec:
        # Check if silence overlaps with range
        if sil_end > start_time and sil_start < end_time:
            # Clip to range
            clipped_start = max(sil_start, start_time)
            clipped_end = min(sil_end, end_time)
            duration = clipped_end - clipped_start
            if duration >= min_duration:
                result.append((clipped_start, clipped_end))
    return result


def _split_segments_at_silences(
    segments: list[Segment],
    silences_sec: list[tuple[float, float]],
    start_time: float,
    end_time: float,
) -> list[list[Segment]]:
    """
    Split segments into chunks based on silence boundaries.
    
    Returns a list of segment groups, where each group represents
    a chunk between silences.
    """
    # Get segments in the time range
    range_segments = [
        s for s in segments 
        if s.start >= start_time - 0.5 and s.end <= end_time + 0.5
    ]
    
    if not range_segments:
        return []
    
    # Find silences in range
    silences = _find_silences_in_range(silences_sec, start_time, end_time)
    
    if not silences:
        # No silences to split on - return all segments as one chunk
        return [range_segments]
    
    # Sort silences by start time
    silences.sort(key=lambda x: x[0])
    
    # Split segments into chunks at silence boundaries
    chunks = []
    current_chunk = []
    silence_idx = 0
    
    for seg in range_segments:
        # Check if this segment is before the next silence
        if silence_idx < len(silences):
            sil_start, sil_end = silences[silence_idx]
            
            if seg.end <= sil_start:
                # Segment is before silence - add to current chunk
                current_chunk.append(seg)
            elif seg.start >= sil_end:
                # Segment is after silence - start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [seg]
                silence_idx += 1
            else:
                # Segment spans silence - add to current chunk and start new
                current_chunk.append(seg)
                chunks.append(current_chunk)
                current_chunk = []
                silence_idx += 1
        else:
            # No more silences - add all remaining to current chunk
            current_chunk.append(seg)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [range_segments]


def _try_split_and_restitch(
    segments: list[Segment],
    ayah: Ayah,
    dp_result: AlignmentResult,
    silences_ms: list[tuple[int, int]] | None,
) -> AlignmentResult | None:
    """
    Try to improve alignment for a long ayah by splitting at silences.
    
    Strategy:
    1. Find silence boundaries within the ayah's time range
    2. Split into chunks
    3. Compute similarity for the merged text of all chunks
    4. If better than original, return improved result
    
    Returns:
        Improved AlignmentResult or None if no improvement
    """
    if not silences_ms:
        return None
    
    # Convert silences to seconds
    silences_sec = [(s / 1000.0, e / 1000.0) for s, e in silences_ms]
    
    # Split segments at silences within this ayah's time range
    chunks = _split_segments_at_silences(
        segments, 
        silences_sec, 
        dp_result.start_time, 
        dp_result.end_time
    )
    
    if len(chunks) <= 1:
        # No meaningful split possible
        return None
    
    # Merge all chunk texts
    all_texts = []
    for chunk in chunks:
        chunk_text = " ".join(seg.text for seg in chunk)
        if chunk_text.strip():
            all_texts.append(chunk_text)
    
    if not all_texts:
        return None
    
    merged_text = " ".join(all_texts)
    
    # Compute new similarity
    new_sim = similarity(merged_text, ayah.text)
    
    # Only accept if significantly better (at least 5% improvement)
    if new_sim > dp_result.similarity_score + 0.05:
        return AlignmentResult(
            ayah=ayah,
            start_time=dp_result.start_time,
            end_time=dp_result.end_time,
            transcribed_text=merged_text,
            similarity_score=new_sim,
            overlap_detected=dp_result.overlap_detected,
        )
    
    return None


def align_segments_hybrid(
    segments: list[Segment],
    ayahs: list[Ayah],
    silences_ms: list[tuple[int, int]] | None = None,
    quality_threshold: float = 0.85,
    long_ayah_words: int = 30,
    long_ayah_duration: float = 30.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[AlignmentResult], HybridStats]:
    """
    Hybrid alignment combining DP and old aligner with smart fallback.
    
    Strategy:
    1. Run DP aligner on all segments/ayahs
    2. For each ayah with similarity < quality_threshold:
       a. If long ayah (>30 words or >30s): try split-and-restitch
       b. Try old aligner as fallback
       c. Keep whichever result is best
    
    Args:
        segments: List of transcribed segments
        ayahs: List of reference ayahs
        silences_ms: Silence periods in milliseconds
        quality_threshold: Similarity below which to try fallback (default 0.85)
        long_ayah_words: Word count threshold for "long" ayahs
        long_ayah_duration: Duration threshold for "long" ayahs (seconds)
        on_progress: Optional progress callback (current, total)
    
    Returns:
        Tuple of (alignment_results, hybrid_stats)
    """
    from .aligner import align_segments  # Import old aligner
    
    stats = HybridStats(total_ayahs=len(ayahs))
    
    if not segments or not ayahs:
        return [], stats
    
    # Step 1: Run DP aligner
    dp_results = align_segments_dp_with_constraints(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
        on_progress=on_progress,
    )
    
    # If DP returned no results, fall back entirely to old aligner
    if not dp_results:
        old_results = align_segments(
            segments=segments,
            ayahs=ayahs,
            silences_ms=silences_ms,
        )
        stats.old_fallback = len(old_results)
        return old_results, stats
    
    # Step 2: Run old aligner to have fallback options
    old_results = align_segments(
        segments=segments,
        ayahs=ayahs,
        silences_ms=silences_ms,
    )
    
    # Build lookup for old results by ayah number
    old_by_ayah: dict[int, AlignmentResult] = {}
    for r in old_results:
        old_by_ayah[r.ayah.ayah_number] = r
    
    # Step 3: For each DP result, check quality and apply fallback if needed
    final_results = []
    
    for dp_r in dp_results:
        ayah = dp_r.ayah
        ayah_word_count = len(ayah.text.split())
        ayah_duration = dp_r.end_time - dp_r.start_time
        
        is_long_ayah = (ayah_word_count > long_ayah_words or 
                        ayah_duration > long_ayah_duration)
        
        # Check if DP result is good enough
        if dp_r.similarity_score >= quality_threshold:
            # Good quality - keep DP result
            final_results.append(dp_r)
            stats.dp_kept += 1
            continue
        
        # DP result is low quality - try to improve
        best_result = dp_r
        best_source = "dp"
        
        # Try 1: For long ayahs, try split-and-restitch
        if is_long_ayah:
            split_result = _try_split_and_restitch(
                segments, ayah, dp_r, silences_ms
            )
            if split_result and split_result.similarity_score > best_result.similarity_score:
                best_result = split_result
                best_source = "split"
        
        # Try 2: Check if old aligner did better
        old_r = old_by_ayah.get(ayah.ayah_number)
        if old_r and old_r.similarity_score > best_result.similarity_score:
            best_result = old_r
            best_source = "old"
        
        # Record stats based on final choice
        if best_source == "old":
            stats.old_fallback += 1
        elif best_source == "split":
            stats.split_improved += 1
        elif best_result.similarity_score < quality_threshold:
            stats.still_low += 1
        else:
            stats.dp_kept += 1
        
        final_results.append(best_result)
    
    return final_results, stats
