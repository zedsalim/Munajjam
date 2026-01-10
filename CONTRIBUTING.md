# Contributing to Munajjam

Thank you for your interest in contributing to **Munajjam** (مُنَجِّم), a Python library for synchronizing Quran audio recitations with their corresponding ayahs!

This guide will help you understand the codebase architecture and how to make meaningful contributions.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Codebase Structure](#codebase-structure)
4. [Key Concepts](#key-concepts)

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or poetry for dependency management

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/munajjam.git
cd munajjam

# Install dependencies
pip install -e ".[dev]"

# Or with poetry
poetry install --with dev
```

### Understanding the Workflow

The library follows a simple pipeline:

```
Audio File → Transcription → Alignment → Results
```

**Example Usage:**
```python
from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs

# 1. Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("surah_1.wav")

# 2. Load reference ayahs
ayahs = load_surah_ayahs(1)

# 3. Align segments to ayahs
results = align(segments, ayahs)

# 4. Access timing information
for result in results:
    print(f"Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s")
```

---

## Architecture Overview

Munajjam is organized into **5 main layers**:

```
┌─────────────────────────────────────────────┐
│         User Application Layer              │
│  (WhisperTranscriber, Aligner, align())     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│      Core Processing Layer                  │
│  • Alignment Strategies (DP, Greedy, Hybrid)│
│  • Text Utilities (Arabic normalization)    │
│  • Post-processing (zone realignment)       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│      Data & Infrastructure Layer            │
│  • Models (Pydantic data classes)           │
│  • Data Access (Quran reference data)       │
│  • Configuration & Logging                  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│      Transcription Backend Layer            │
│  • WhisperTranscriber                       │
│  • Audio Processing (silence detection)     │
└─────────────────────────────────────────────┘
```

---

## Codebase Structure

```
munajjam/
├── munajjam/              # Main package
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Configuration management (Pydantic)
│   ├── exceptions.py      # Custom exceptions
│   ├── _logging.py        # Logging utilities
│   │
│   ├── core/              # Alignment algorithms (~2,742 LOC)
│   │   ├── aligner.py            # Unified alignment interface ⭐
│   │   ├── dp_core.py            # Dynamic programming algorithm
│   │   ├── aligner_greedy.py     # Fast greedy matching
│   │   ├── hybrid.py             # Hybrid strategy (DP + greedy)
│   │   ├── zone_realigner.py     # Fixes timing drift
│   │   ├── matcher.py            # Text similarity utilities
│   │   ├── arabic.py             # Arabic text normalization
│   │   ├── cascade_recovery.py   # Recovery strategy
│   │   └── overlap.py            # Overlap detection/removal
│   │
│   ├── models/            # Data structures (Pydantic)
│   │   ├── ayah.py       # Quran verse model
│   │   ├── segment.py    # Audio segment with timing
│   │   ├── result.py     # Alignment result
│   │   └── surah.py      # Surah metadata (114 surahs)
│   │
│   ├── transcription/     # Audio processing
│   │   ├── base.py       # Abstract BaseTranscriber interface
│   │   ├── whisper.py    # Tarteel AI Whisper implementation
│   │   └── silence.py    # Silence detection
│   │
│   └── data/              # Reference data
│       ├── quran.py      # Quran text loader
│       └── quran_ayat.csv # Reference data (6,236 ayahs)
│
├── examples/              # Usage examples ⭐
├── tests/                 # Test suite (needs expansion!)
└── pyproject.toml         # Project configuration
```

**⭐ = Great starting points for contributors**

---

## Key Concepts

### 1. Data Models

All data flows through **Pydantic models** for type safety and validation:

**Ayah** - A Quran verse
```python
@dataclass
class Ayah:
    id: int                  # Unique ID (1-6236)
    surah_id: int            # Surah number (1-114)
    ayah_number: int         # Position within surah
    text: str                # Arabic text
```

**Segment** - A transcribed audio segment
```python
@dataclass
class Segment:
    id: str
    surah_id: int
    start: float             # Start time (seconds)
    end: float               # End time (seconds)
    text: str                # Transcribed Arabic text
    type: SegmentType        # AYAH, ISTI3AZA, or BASMALA
    confidence: float        # 0.0-1.0
```

**AlignmentResult** - The final output
```python
@dataclass
class AlignmentResult:
    ayah: Ayah
    start_time: float
    end_time: float
    transcribed_text: str
    similarity_score: float  # Quality metric (0.0-1.0)
    overlap_detected: bool
```

### 2. Alignment Strategies

Munajjam supports **3 alignment strategies**:

#### **Greedy** (Simple & Fast)
- Linear matching from start to end
- Best for simple cases with 1:1 segment-to-ayah mapping
- Fast but may miss optimal alignments

#### **Dynamic Programming (DP)** (Optimal)
- Finds the globally optimal alignment using cost matrix
- Handles complex merging of segments
- Slower but highest quality

#### **Hybrid** (Recommended)
- Starts with DP for high-quality initial alignment
- Falls back to greedy for low-confidence ayahs
- Includes **split-and-restitch** for long ayahs
- Post-processes with **zone realignment**

**Example:**
```python
from munajjam.core import Aligner

# Use hybrid strategy (recommended)
aligner = Aligner(
    strategy="hybrid",
    quality_threshold=0.85,
    fix_drift=True,
    fix_overlaps=True
)
results = aligner.align(segments, ayahs)
```

### 3. Post-Processing Features

#### **Zone Realignment** (`zone_realigner.py`)
- Identifies "problem zones" (3+ consecutive low-confidence ayahs)
- Re-aligns only those zones to fix drift
- Keeps the best result for each ayah
- Critical for long surahs where timing drifts over time

#### **Overlap Removal** (`overlap.py`)
- Detects overlapping ayah timings
- Intelligently merges duplicate segments
- Ensures clean separation between ayahs

#### **Buffer System**
- Extends ayah boundaries into adjacent silence
- Default: 0.3s into preceding/following silence
- Prevents word cutoffs at boundaries

### 4. Text Matching

The library uses multiple techniques to match transcribed text to reference ayahs:

1. **Last Words Check** - Primary matching (adaptive word count: 1-3)
2. **Full Text Similarity** - Secondary verification using normalized text
3. **Coverage Ratio** - Ensures at least 70% of ayah is captured
4. **Arabic Normalization** - Removes diacritics, normalizes characters

**Example:**
```python
from munajjam.core import normalize_arabic, similarity

# Normalize Arabic text
normalized = normalize_arabic("بِسْمِ ٱللَّهِ")
# Result: "بسم الله"

# Compute similarity
score = similarity("بسم الله الرحمن", "بسم الله الرحمن الرحيم")
# Result: ~0.75 (75% similar)
```

---

**Thank you for contributing to Munajjam! 🌙**
