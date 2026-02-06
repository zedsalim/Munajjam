"""
CTC Forced Alignment using torchaudio MMS (Massively Multilingual Speech).

Aligns reference text directly to audio without a transcription step.
Used as a refinement pass after word-DP to improve boundary precision,
especially in problem zones where text similarity is ambiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_DEVICE = None
_BUNDLE = None
_MODEL = None
_TOKENIZER = None
_ALIGNER = None
_SAMPLE_RATE = None
_AVAILABLE: bool | None = None


@dataclass
class AlignedWord:
    """A word aligned to audio via CTC forced alignment."""
    word: str
    start: float   # seconds
    end: float     # seconds
    score: float   # alignment confidence (0-1)


@dataclass
class AlignedSpan:
    """An aligned span covering one or more words."""
    text: str
    start: float
    end: float
    score: float
    words: list[AlignedWord]


def is_available() -> bool:
    """Check if CTC forced alignment is available (torchaudio with MMS)."""
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
    """Lazy-load the MMS forced alignment model."""
    global _DEVICE, _BUNDLE, _MODEL, _TOKENIZER, _ALIGNER, _SAMPLE_RATE

    if _MODEL is not None:
        return

    import torch
    import torchaudio

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _BUNDLE = torchaudio.pipelines.MMS_FA
    _MODEL = _BUNDLE.get_model().to(_DEVICE)
    _MODEL.eval()
    _TOKENIZER = _BUNDLE.get_tokenizer()
    _ALIGNER = _BUNDLE.get_aligner()
    _SAMPLE_RATE = _BUNDLE.sample_rate


def _load_audio_segment(
    audio_path: str | Path,
    start_sec: float,
    end_sec: float,
):
    """Load a specific time range from an audio file.

    Uses librosa for robust format support (mp3, wav, etc.) and converts
    to a torch tensor at the model's expected sample rate.
    """
    import torch
    import librosa

    duration = end_sec - start_sec
    y, _ = librosa.load(
        str(audio_path),
        sr=_SAMPLE_RATE,
        mono=True,
        offset=start_sec,
        duration=duration,
    )
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    return waveform


def _normalize_for_ctc(text: str) -> str:
    """Normalize Arabic text for CTC tokenizer (lowercase romanization won't work).

    MMS uses romanized tokens. For Arabic, we pass the text as-is and let
    the tokenizer handle unknown characters gracefully. Characters not in
    the vocabulary are skipped.
    """
    # Strip diacritics — MMS doesn't know Arabic tashkeel
    from .arabic import normalize_arabic
    return normalize_arabic(text).strip()


def align_text_to_audio(
    audio_path: str | Path,
    text: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> AlignedSpan | None:
    """
    Align a text string to a region of audio using CTC forced alignment.

    Args:
        audio_path: Path to the audio file
        text: Reference text to align
        start_sec: Start of the audio region (seconds)
        end_sec: End of the audio region (seconds). None = end of file.

    Returns:
        AlignedSpan with per-word timing, or None if alignment fails.
    """
    if not is_available():
        return None

    try:
        import torch

        _ensure_model()

        if end_sec is None:
            import torchaudio
            info = torchaudio.info(str(audio_path))
            end_sec = info.num_frames / info.sample_rate

        waveform = _load_audio_segment(audio_path, start_sec, end_sec)
        if waveform.shape[1] < 400:  # Too short for meaningful alignment
            return None

        # Tokenize
        norm_text = _normalize_for_ctc(text)
        tokens = _TOKENIZER(norm_text)
        if not tokens:
            return None

        # Run model
        with torch.no_grad():
            emission, _ = _MODEL(waveform.to(_DEVICE))

        # Align
        token_spans = _ALIGNER(emission[0], tokens)

        # Convert token spans to word spans
        words_text = norm_text.split()
        ratio = waveform.shape[1] / emission.shape[1]
        duration = end_sec - start_sec

        # Map tokens back to words
        # MMS tokenizes at character level; we need to group back to words
        aligned_words: list[AlignedWord] = []
        token_idx = 0

        for word in words_text:
            word_chars = list(word.replace(" ", ""))
            n_chars = len(word_chars)

            if token_idx + n_chars > len(token_spans):
                break

            word_token_spans = token_spans[token_idx:token_idx + n_chars]
            if not word_token_spans:
                token_idx += n_chars
                continue

            # Word timing from first to last token
            w_start = word_token_spans[0].start
            w_end = word_token_spans[-1].end
            w_score = sum(s.score for s in word_token_spans) / len(word_token_spans)

            # Convert frame indices to seconds
            w_start_sec = start_sec + (w_start * ratio / _SAMPLE_RATE)
            w_end_sec = start_sec + (w_end * ratio / _SAMPLE_RATE)

            aligned_words.append(AlignedWord(
                word=word,
                start=round(w_start_sec, 3),
                end=round(w_end_sec, 3),
                score=round(w_score, 4),
            ))

            token_idx += n_chars

        if not aligned_words:
            return None

        avg_score = sum(w.score for w in aligned_words) / len(aligned_words)
        return AlignedSpan(
            text=norm_text,
            start=aligned_words[0].start,
            end=aligned_words[-1].end,
            score=round(avg_score, 4),
            words=aligned_words,
        )

    except Exception as e:
        logger.debug("CTC alignment failed: %s", e)
        return None


def refine_alignment_results(
    results,  # list[AlignmentResult] — avoid circular import
    audio_path: str | Path,
    min_similarity: float = 0.5,
    min_ctc_score: float = 0.3,
):
    """
    Refine word-DP alignment results using CTC forced alignment.

    For each ayah result with similarity above ``min_similarity``, runs
    CTC alignment on the corresponding audio segment and replaces
    timestamps if the CTC confidence is sufficient.

    Modifies ``results`` in-place and returns the number of refined ayahs.

    Args:
        results: List of AlignmentResult from word-DP
        audio_path: Path to the source audio file
        min_similarity: Only refine ayahs with similarity >= this (default 0.5)
        min_ctc_score: Accept CTC result only if avg score >= this (default 0.3)

    Returns:
        Number of ayahs whose timestamps were refined.
    """
    if not is_available() or not results:
        return 0

    from ..models import AlignmentResult as AR

    refined = 0
    for i, result in enumerate(results):
        if result.similarity_score < min_similarity:
            continue

        # Give a bit of padding for CTC alignment
        padding = 0.5
        start = max(0.0, result.start_time - padding)
        end = result.end_time + padding

        span = align_text_to_audio(
            audio_path=audio_path,
            text=result.ayah.text,
            start_sec=start,
            end_sec=end,
        )

        if span is None or span.score < min_ctc_score:
            continue

        # Replace timestamps with CTC-refined ones
        results[i] = AR(
            ayah=result.ayah,
            start_time=round(span.start, 3),
            end_time=round(span.end, 3),
            transcribed_text=result.transcribed_text,
            similarity_score=result.similarity_score,
            overlap_detected=result.overlap_detected,
        )
        refined += 1

    return refined
