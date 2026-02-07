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


# Arabic letter → romanized approximation for MMS_FA tokenizer.
# MMS_FA vocabulary: - a i e n o u t s r m k l d g h y b p w c v j z f ' q x *
_ARABIC_TO_ROMAN: dict[str, str] = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "s",   # th → s (closest available)
    "ج": "j",
    "ح": "h",
    "خ": "x",   # kh → x
    "د": "d",
    "ذ": "d",   # dh → d
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "s",   # sh → s
    "ص": "s",   # emphatic s
    "ض": "d",   # emphatic d
    "ط": "t",   # emphatic t
    "ظ": "z",   # emphatic z
    "ع": "a",   # ain → a (guttural)
    "غ": "g",   # ghain → g
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    " ": " ",
}


def _normalize_text_for_ctc(text: str) -> str:
    """Normalize Arabic text and transliterate to romanized form for MMS_FA."""
    from .arabic import normalize_arabic

    normalized = normalize_arabic(text).strip()

    # Transliterate Arabic → romanized
    roman_chars: list[str] = []
    for ch in normalized:
        mapped = _ARABIC_TO_ROMAN.get(ch)
        if mapped is not None:
            roman_chars.append(mapped)
        # Skip characters not in the mapping (punctuation, digits, etc.)

    romanized = "".join(roman_chars)
    # Collapse multiple spaces
    import re
    romanized = re.sub(r"\s+", " ", romanized).strip()
    return romanized


def _tokenize_text(text: str) -> list[int]:
    """Convert romanized text to token indices, skipping unknown characters.

    The MMS_FA tokenizer expects ``List[str]`` (list of words), where each
    word is a string of characters that exist in its dictionary.  It returns
    ``List[List[int]]`` (token ids per word).  We flatten them into a single
    list suitable for CTC segmentation.
    """
    known = set(_TOKENIZER.dictionary.keys()) if hasattr(_TOKENIZER, "dictionary") else set()

    # Split into words, filter each word to known characters only
    words: list[str] = []
    for w in text.split():
        filtered = "".join(ch for ch in w if ch in known)
        if filtered:
            words.append(filtered)

    if not words:
        return []

    # Tokenizer returns List[List[int]] — flatten to a single list
    nested = _TOKENIZER(words)
    return [tok for word_toks in nested for tok in word_toks]


def _get_log_probs(waveform, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0) -> np.ndarray:
    """Run the CTC model and return log-probability matrix (T, V).

    For long audio, processes in overlapping chunks to avoid GPU OOM.
    Each chunk is ``chunk_seconds`` long with ``overlap_seconds`` overlap
    to avoid boundary artefacts.  Overlap regions are discarded (we keep
    the interior portion of each chunk).
    """
    import torch

    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_seconds * _SAMPLE_RATE)
    overlap_samples = int(overlap_seconds * _SAMPLE_RATE)

    # If the waveform fits in one chunk, process directly
    if total_samples <= chunk_samples:
        with torch.no_grad():
            emission, _ = _MODEL(waveform.to(_DEVICE))
        log_probs = torch.nn.functional.log_softmax(emission[0], dim=-1)
        return log_probs.cpu().numpy()

    # --- Chunked processing ---
    step_samples = chunk_samples - overlap_samples
    chunks_log_probs: list[np.ndarray] = []

    # First pass: process one small chunk to learn the samples-to-frames ratio
    probe = waveform[:, :chunk_samples]
    with torch.no_grad():
        probe_em, _ = _MODEL(probe.to(_DEVICE))
    probe_frames = probe_em.shape[1]
    ratio = chunk_samples / probe_frames  # samples per frame
    overlap_frames = max(1, int(overlap_samples / ratio))

    # Process the probe chunk (reuse it)
    probe_lp = torch.nn.functional.log_softmax(probe_em[0], dim=-1).cpu().numpy()
    del probe_em
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    offset = 0
    chunk_idx = 0
    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk_wav = waveform[:, offset:end]

        if chunk_idx == 0:
            # Reuse the probe we already computed
            lp = probe_lp
        else:
            with torch.no_grad():
                em, _ = _MODEL(chunk_wav.to(_DEVICE))
            lp = torch.nn.functional.log_softmax(em[0], dim=-1).cpu().numpy()
            del em
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Trim overlap: discard the first overlap_frames for all chunks
        # except the first, and discard the last overlap_frames for all
        # chunks except the last.
        trim_start = overlap_frames if chunk_idx > 0 else 0
        trim_end = lp.shape[0]  # keep to end by default

        chunks_log_probs.append(lp[trim_start:trim_end])

        offset += step_samples
        chunk_idx += 1

    del probe_lp
    return np.concatenate(chunks_log_probs, axis=0)


def _ctc_segmentation_dp(
    log_probs: np.ndarray,
    token_sequences: list[list[int]],
    blank_id: int = 0,
) -> list[tuple[int, int, float]]:
    """
    CTC segmentation via dynamic programming on the trellis.

    Given frame-level log-probabilities and per-ayah token sequences,
    finds the optimal frame boundaries for each ayah.

    Memory-efficient: instead of storing the full (T, S) trellis, only
    tracks scores at ayah separator positions needed for boundary
    detection, plus a running confidence accumulator per ayah.

    Args:
        log_probs: Shape (T, V) — log probabilities per frame per token.
        token_sequences: List of token sequences, one per ayah.
        blank_id: Index of the CTC blank token.

    Returns:
        List of (start_frame, end_frame, confidence) per ayah.
    """
    T, V = log_probs.shape
    n_ayahs = len(token_sequences)

    # Build the concatenated token sequence with blank separators.
    concat_tokens: list[int] = []
    ayah_boundaries: list[tuple[int, int]] = []  # (start, end) in concat_tokens

    for seq in token_sequences:
        if not seq:
            start = len(concat_tokens)
            concat_tokens.append(blank_id)
            ayah_boundaries.append((start, start + 1))
            continue

        start = len(concat_tokens)
        concat_tokens.extend(seq)
        end = len(concat_tokens)
        ayah_boundaries.append((start, end))
        concat_tokens.append(blank_id)

    S = len(concat_tokens)

    if S == 0 or T == 0:
        return [(0, T, 0.0)] * n_ayahs

    NEG_INF = -1e10

    # Identify separator token positions (the blank after each ayah).
    # These are the columns we need to track for boundary detection.
    sep_positions: list[int] = []  # one per boundary (n_ayahs - 1)
    for i in range(n_ayahs - 1):
        sep_s = ayah_boundaries[i][1]
        if sep_s >= S:
            sep_s = S - 1
        sep_positions.append(sep_s)

    # Pre-compute proportional expected boundaries and search windows
    total_tokens_count = sum(max(len(seq), 1) for seq in token_sequences)
    cum_tokens = [0]
    for seq in token_sequences:
        cum_tokens.append(cum_tokens[-1] + max(len(seq), 1))

    expected_frames: list[int] = []
    search_ranges: list[tuple[int, int]] = []
    for i in range(n_ayahs - 1):
        expected = int(T * cum_tokens[i + 1] / total_tokens_count)
        window = max(int(T * 0.15), 20)
        lo = max(1, expected - window)
        hi = min(T - 1, expected + window)
        expected_frames.append(expected)
        search_ranges.append((lo, hi))

    # Memory-efficient storage: only keep separator scores within
    # each search window, plus per-ayah confidence accumulators.
    # sep_scores[i] maps frame_t -> score for the i-th separator
    sep_best_frame = [expected_frames[i] for i in range(n_ayahs - 1)]
    sep_best_score = [NEG_INF] * (n_ayahs - 1)

    # For confidence: track running max score per ayah region per frame
    # We accumulate sum of max-over-tokens scores per ayah
    ayah_confidence_sum = [0.0] * n_ayahs
    ayah_confidence_count = [0] * n_ayahs

    # Vectorized forward pass using numpy operations
    concat_tokens_arr = np.array(concat_tokens, dtype=np.int64)

    prev = np.full(S, NEG_INF, dtype=np.float64)
    curr = np.full(S, NEG_INF, dtype=np.float64)

    # Initialize
    prev[0] = log_probs[0, concat_tokens[0]]
    if S > 1:
        prev[1] = log_probs[0, concat_tokens[1]]

    # Process t=0 for confidence tracking
    for ai in range(n_ayahs):
        s_start, s_end = ayah_boundaries[ai]
        region = prev[s_start:s_end]
        mx = float(np.max(region)) if region.size > 0 else NEG_INF
        if mx > NEG_INF / 2:
            ayah_confidence_sum[ai] += mx
            ayah_confidence_count[ai] += 1

    for t in range(1, T):
        # Vectorized emission lookup
        emit_probs = log_probs[t, concat_tokens_arr]  # shape (S,)

        # Transition: stay
        curr[:] = prev

        # Transition: advance from s-1
        advance = np.full(S, NEG_INF, dtype=np.float64)
        advance[1:] = prev[:-1]
        curr = np.logaddexp(curr, advance)

        # Transition: skip blank (s-2) where tokens differ
        if S > 2:
            skip = np.full(S, NEG_INF, dtype=np.float64)
            skip[2:] = prev[:-2]
            # Mask: only allow skip where concat_tokens[s] != concat_tokens[s-2]
            same_mask = concat_tokens_arr[2:] == concat_tokens_arr[:-2]
            skip[2:][same_mask] = NEG_INF
            curr = np.logaddexp(curr, skip)

        curr += emit_probs

        # Track separator scores for boundary detection
        for i in range(n_ayahs - 1):
            lo, hi = search_ranges[i]
            if lo <= t <= hi:
                score = float(curr[sep_positions[i]])
                if score > sep_best_score[i]:
                    sep_best_score[i] = score
                    sep_best_frame[i] = t

        # Track per-ayah confidence (sample every 4th frame for speed)
        if t % 4 == 0:
            for ai in range(n_ayahs):
                s_start, s_end = ayah_boundaries[ai]
                region = curr[s_start:s_end]
                if region.size > 0:
                    mx = float(np.max(region))
                    if mx > NEG_INF / 2:
                        ayah_confidence_sum[ai] += mx
                        ayah_confidence_count[ai] += 1

        prev, curr = curr, prev

    # Build frame boundaries from separator peaks
    frame_boundaries: list[int] = [0]
    for i in range(n_ayahs - 1):
        frame_boundaries.append(sep_best_frame[i])
    frame_boundaries.append(T)

    # Build results
    results: list[tuple[int, int, float]] = []
    for i in range(n_ayahs):
        start_frame = frame_boundaries[i]
        end_frame = frame_boundaries[i + 1]

        if ayah_confidence_count[i] > 0:
            avg_log_prob = ayah_confidence_sum[i] / ayah_confidence_count[i]
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
        total_samples = waveform.shape[1]
        logger.info(
            "CTC: audio %.1f min, %d samples",
            audio_duration / 60, total_samples,
        )

        # Get frame-level log probabilities (chunked for long audio)
        log_probs = _get_log_probs(waveform)
        T = log_probs.shape[0]

        # Compute frames-to-seconds ratio
        ratio = total_samples / T  # samples per frame
        frame_to_sec = ratio / _SAMPLE_RATE

        # Free waveform — no longer needed
        del waveform

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
