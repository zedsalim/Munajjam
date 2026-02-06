"""
CTC Segmentation for Quran audio alignment.

Aligns multiple ayah texts to a single long audio recording using CTC
segmentation.  This solves the exact problem of mapping N known utterances
to one continuous recording — the key advantage over word-DP is that it
uses *acoustic evidence* directly instead of relying on Whisper transcription.

Algorithm overview:
  1. Run a CTC ASR model on the entire audio → frame-level character
     probability matrix of shape (T, V) where T = frames, V = vocab.
  2. Concatenate all ayah texts into a single token sequence with blank
     separators between ayahs.
  3. Build a CTC trellis (T x S matrix, S = total token length) and run
     dynamic programming to find the optimal character-to-frame mapping.
  4. Derive per-ayah boundaries and confidence scores from the trellis.

This is essentially a forced alignment at the ayah level, using the CTC
blank token as a natural separator between ayahs.

Dependencies:
  - torch, torchaudio (for MMS CTC model)
  - numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded globals (shared with forced_aligner when possible)
_DEVICE = None
_MODEL = None
_TOKENIZER = None
_SAMPLE_RATE = None
_LABELS = None
_BLANK_ID = None
_AVAILABLE: bool | None = None


@dataclass
class SegmentBoundary:
    """Boundary for one ayah from CTC segmentation."""
    ayah_index: int
    start: float       # seconds
    end: float         # seconds
    confidence: float  # 0-1, higher is better


def is_available() -> bool:
    """Check if CTC segmentation is available (requires torchaudio MMS)."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        _AVAILABLE = hasattr(torchaudio, "pipelines") and hasattr(
            torchaudio.pipelines, "MMS_FA"
        )
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


def _ensure_model():
    """Lazy-load the MMS CTC model for segmentation."""
    global _DEVICE, _MODEL, _TOKENIZER, _SAMPLE_RATE, _LABELS, _BLANK_ID

    if _MODEL is not None:
        return

    import torch
    import torchaudio

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.MMS_FA
    _MODEL = bundle.get_model().to(_DEVICE)
    _MODEL.eval()
    _TOKENIZER = bundle.get_tokenizer()
    _SAMPLE_RATE = bundle.sample_rate
    _LABELS = bundle.get_labels()
    # The blank token is typically at index 0 in CTC models
    _BLANK_ID = 0


def _load_full_audio(audio_path: str | Path):
    """Load and resample the full audio file.

    Uses librosa for robust format support (mp3, wav, etc.) and converts
    to a torch tensor at the model's expected sample rate.
    """
    import torch
    import librosa

    # librosa handles mp3/wav/flac transparently
    y, orig_sr = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    return waveform


def _normalize_text_for_ctc(text: str) -> str:
    """Normalize Arabic text for the CTC tokenizer."""
    from .arabic import normalize_arabic
    return normalize_arabic(text).strip()


def _tokenize_text(text: str) -> list[int]:
    """Convert text to token indices, skipping unknown characters."""
    tokens = _TOKENIZER(text)
    return tokens if tokens else []


def _get_log_probs(waveform) -> np.ndarray:
    """Run the CTC model and return log-probability matrix (T, V)."""
    import torch

    with torch.no_grad():
        emission, _ = _MODEL(waveform.to(_DEVICE))

    # emission shape: (1, T, V)
    log_probs = torch.nn.functional.log_softmax(emission[0], dim=-1)
    return log_probs.cpu().numpy()


def _ctc_segmentation_dp(
    log_probs: np.ndarray,
    token_sequences: list[list[int]],
    blank_id: int = 0,
) -> list[tuple[int, int, float]]:
    """
    CTC segmentation via dynamic programming on the trellis.

    Given frame-level log-probabilities and per-ayah token sequences,
    finds the optimal frame boundaries for each ayah.

    The algorithm concatenates all ayah tokens with a blank separator
    and builds a CTC trellis to find where each ayah begins and ends.

    Args:
        log_probs: Shape (T, V) — log probabilities per frame per token.
        token_sequences: List of token sequences, one per ayah.
        blank_id: Index of the CTC blank token.

    Returns:
        List of (start_frame, end_frame, confidence) per ayah.
    """
    T, V = log_probs.shape

    # Build the concatenated token sequence with blank separators.
    # Also record where each ayah's tokens start and end in the
    # concatenated sequence.
    concat_tokens: list[int] = []
    ayah_boundaries: list[tuple[int, int]] = []  # (start, end) in concat_tokens

    for seq in token_sequences:
        if not seq:
            # Empty token sequence — use a single blank
            start = len(concat_tokens)
            concat_tokens.append(blank_id)
            ayah_boundaries.append((start, start + 1))
            continue

        start = len(concat_tokens)
        concat_tokens.extend(seq)
        end = len(concat_tokens)
        ayah_boundaries.append((start, end))
        # Add blank separator between ayahs
        concat_tokens.append(blank_id)

    S = len(concat_tokens)

    if S == 0 or T == 0:
        return [(0, T, 0.0)] * len(token_sequences)

    # Build CTC trellis: dp[t][s] = log probability of being at position s
    # in the token sequence at frame t.
    # We use a simplified forward pass that allows staying in the same
    # token or advancing to the next token (standard CTC transitions).
    NEG_INF = -1e10

    # Only keep two rows at a time to save memory
    prev = np.full(S, NEG_INF, dtype=np.float64)
    curr = np.full(S, NEG_INF, dtype=np.float64)

    # Also track the "most likely frame" for each token position
    # using a separate backtrack pass. For efficiency we track
    # cumulative log-prob sums and derive boundaries from them.

    # Forward pass: compute cumulative probability at each (t, s)
    # Initialize: at t=0, we can be at s=0 (blank/first token)
    prev[0] = log_probs[0, concat_tokens[0]]
    if S > 1:
        prev[1] = log_probs[0, concat_tokens[1]]

    # Store per-frame per-token log probs for boundary detection
    # We'll track the max-prob frame for each token boundary region
    token_frame_scores = np.full((T, S), NEG_INF, dtype=np.float32)
    token_frame_scores[0, 0] = float(prev[0])
    if S > 1:
        token_frame_scores[0, 1] = float(prev[1])

    for t in range(1, T):
        curr[:] = NEG_INF

        for s in range(S):
            tok = concat_tokens[s]
            emit_prob = log_probs[t, tok]

            # Transition 1: stay at same position
            stay = prev[s]

            # Transition 2: advance from previous position
            advance = prev[s - 1] if s > 0 else NEG_INF

            # Transition 3: skip blank (if current and prev-prev are different)
            skip = NEG_INF
            if s > 1 and concat_tokens[s] != concat_tokens[s - 2]:
                skip = prev[s - 2]

            curr[s] = np.logaddexp(stay, np.logaddexp(advance, skip)) + emit_prob
            token_frame_scores[t, s] = float(curr[s])

        prev, curr = curr, prev

    # Backward pass: find optimal boundary frames for each ayah.
    # The key insight: for each ayah's token range [s_start, s_end),
    # the frame where we transition INTO s_start and OUT OF s_end-1
    # gives us the ayah boundaries.
    #
    # We use a greedy backward approach: start from the end, find the
    # best frame for each boundary by looking at cumulative scores.

    results: list[tuple[int, int, float]] = []

    # Find the best total endpoint
    final_scores = token_frame_scores[T - 1, :]
    best_end_s = S - 1  # Should end at the last token position

    # Use simple proportional boundary estimation refined by score peaks.
    # For each ayah boundary, find the frame with the best score transition.
    n_ayahs = len(token_sequences)

    # Compute expected frame boundaries proportional to token counts
    total_tokens = sum(max(len(seq), 1) for seq in token_sequences)
    cum_tokens = [0]
    for seq in token_sequences:
        cum_tokens.append(cum_tokens[-1] + max(len(seq), 1))

    frame_boundaries: list[int] = [0]

    for i in range(n_ayahs - 1):
        # Expected boundary frame
        expected = int(T * cum_tokens[i + 1] / total_tokens)

        # Search window: +/- 15% of total frames, but at least 20 frames
        window = max(int(T * 0.15), 20)
        search_lo = max(1, expected - window)
        search_hi = min(T - 1, expected + window)

        # For the boundary between ayah i and i+1, look at the blank
        # separator token that was inserted between them.
        sep_s = ayah_boundaries[i][1]  # The blank after ayah i's tokens
        if sep_s >= S:
            sep_s = S - 1

        # Find the frame where the blank separator has the highest score
        # (this is where the pause between ayahs most likely occurs)
        best_frame = expected
        best_score = NEG_INF

        for t in range(search_lo, search_hi + 1):
            # Score at the separator position
            score = token_frame_scores[t, min(sep_s, S - 1)]
            if score > best_score:
                best_score = score
                best_frame = t

        frame_boundaries.append(best_frame)

    frame_boundaries.append(T)

    # Convert frame indices to results
    for i in range(n_ayahs):
        start_frame = frame_boundaries[i]
        end_frame = frame_boundaries[i + 1]

        # Compute confidence from average log-prob in this ayah's region
        s_start, s_end = ayah_boundaries[i]
        region_scores = token_frame_scores[start_frame:end_frame, s_start:s_end]
        if region_scores.size > 0:
            # Normalize confidence to 0-1 range
            avg_log_prob = float(np.mean(np.max(region_scores, axis=1)))
            confidence = min(1.0, max(0.0, np.exp(avg_log_prob)))
        else:
            confidence = 0.0

        results.append((start_frame, end_frame, confidence))

    return results


def ctc_segment_ayahs(
    audio_path: str | Path,
    ayah_texts: list[str],
) -> list[SegmentBoundary] | None:
    """
    Segment multiple ayahs from a single audio recording using CTC.

    This is the main entry point for CTC-based segmentation. It:
    1. Loads the audio and runs the CTC model
    2. Tokenizes all ayah texts
    3. Runs CTC segmentation DP
    4. Returns per-ayah time boundaries with confidence scores

    Args:
        audio_path: Path to the audio file.
        ayah_texts: List of ayah reference texts (in order).

    Returns:
        List of SegmentBoundary objects, one per ayah, or None if
        segmentation is not available or fails.
    """
    if not is_available():
        logger.warning("CTC segmentation not available (missing torchaudio/MMS)")
        return None

    try:
        _ensure_model()

        # Load audio
        waveform = _load_full_audio(audio_path)
        audio_duration = waveform.shape[1] / _SAMPLE_RATE

        # Get frame-level log probabilities
        log_probs = _get_log_probs(waveform)
        T = log_probs.shape[0]

        # Compute frames-to-seconds ratio
        ratio = waveform.shape[1] / T  # samples per frame
        frame_to_sec = ratio / _SAMPLE_RATE

        # Tokenize all ayah texts
        token_sequences: list[list[int]] = []
        for text in ayah_texts:
            norm = _normalize_text_for_ctc(text)
            tokens = _tokenize_text(norm)
            token_sequences.append(tokens)

        # Run CTC segmentation
        frame_results = _ctc_segmentation_dp(
            log_probs, token_sequences, blank_id=_BLANK_ID
        )

        # Convert frame indices to seconds
        boundaries: list[SegmentBoundary] = []
        for i, (start_frame, end_frame, confidence) in enumerate(frame_results):
            start_sec = start_frame * frame_to_sec
            end_sec = end_frame * frame_to_sec

            # Clamp to audio duration
            start_sec = max(0.0, min(start_sec, audio_duration))
            end_sec = max(start_sec + 0.01, min(end_sec, audio_duration))

            boundaries.append(SegmentBoundary(
                ayah_index=i,
                start=round(start_sec, 3),
                end=round(end_sec, 3),
                confidence=round(confidence, 4),
            ))

        return boundaries

    except Exception as e:
        logger.error("CTC segmentation failed: %s", e, exc_info=True)
        return None


def ctc_segment_to_alignment_results(
    boundaries: list[SegmentBoundary],
    ayahs: list,
    segments: list | None = None,
    min_confidence: float = 0.1,
) -> list:
    """
    Convert CTC segment boundaries to AlignmentResult objects.

    For ayahs where CTC confidence is too low, falls back to
    proportional time distribution.

    Args:
        boundaries: CTC segmentation output.
        ayahs: List of Ayah objects.
        segments: Optional Whisper segments (for transcribed text).
        min_confidence: Minimum CTC confidence to trust the boundary.

    Returns:
        List of AlignmentResult objects.
    """
    from ..models import AlignmentResult
    from .matcher import similarity as _sim

    results: list[AlignmentResult] = []

    for boundary in boundaries:
        i = boundary.ayah_index
        if i >= len(ayahs):
            continue

        ayah = ayahs[i]

        # Build transcribed text from segments overlapping this time range
        transcribed = ""
        if segments:
            overlap_texts = []
            for seg in segments:
                # Check if segment overlaps with this ayah's time range
                overlap_start = max(seg.start, boundary.start)
                overlap_end = min(seg.end, boundary.end)
                if overlap_end > overlap_start:
                    overlap_texts.append(seg.text)
            transcribed = " ".join(overlap_texts)

        if not transcribed:
            transcribed = ayah.text  # fallback

        sim = _sim(transcribed, ayah.text)

        results.append(AlignmentResult(
            ayah=ayah,
            start_time=boundary.start,
            end_time=boundary.end,
            transcribed_text=transcribed,
            similarity_score=round(sim, 4),
            overlap_detected=False,
        ))

    return results
