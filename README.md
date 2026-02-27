# Munajjam-Warsh

**A Python library to synchronize Quran ayat with audio recitations — Warsh (6214 ayahs edition).**

Munajjam-Warsh uses AI-powered speech recognition to automatically generate precise timestamps for each ayah in a Quran audio recording.

This repository is a fork of [Munajjam](https://github.com/Itqan-community/munajjam) by the [ITQAN Community](https://community.itqan.dev).

It has been adapted to support the **Warsh ‘an Nafi’ riwayah (6214 ayahs version)** based on the Madinah Mushaf edition published by the [King Fahd Glorious Qur'an Printing Complex](https://qurancomplex.gov.sa/quran-dev/).

The original implementation was built around the Hafs recitation dataset. This fork migrates:

* Quran ayah dataset → Warsh (6214 ayahs structure)
* Alignment references → Warsh text
* Model tuning & normalization → adjusted for Warsh orthography differences

---

## About This Fork

### What Changed

* ✅ Migrated Quran data to **Warsh (6214 ayahs)**
* ✅ Updated ayah segmentation to match Warsh mushaf structure
* ✅ Prepared for Warsh-based recitation alignment

### What Remains from the Original Project

* Whisper-based transcription pipeline
* Alignment engine (Auto / Hybrid / DP / Greedy)
* Drift correction logic
* Phonetic similarity scoring
* Word-level timestamp support

All core alignment logic and architecture remain from the original Munajjam project.

Full credit goes to the original authors and contributors.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/zedsalim/Munajjam-Warsh.git
cd Munajjam-Warsh/munajjam
```

Install the package:

```bash
pip install .
```

For faster transcription with faster-whisper:

```bash
pip install ".[faster-whisper]"
```

For development (editable install):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Download a Warsh recitation sample

```bash
curl -L -o 001.mp3 "<your-warsh-recitation-url>"
```

> **Important:** Audio files must correspond to **Warsh recitations** and be named by surah number (`001.mp3`, `002.mp3`, etc.).

---

### 2. Run the Alignment

```python
from munajjam.transcription import WhisperTranscriber
from munajjam.core import align
from munajjam.data import load_surah_ayahs

# Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("001.mp3")

# Load Warsh ayahs (6214 structure)
ayahs = load_surah_ayahs(1)

# Align to ayahs (uses auto strategy by default)
results = align("001.mp3", segments, ayahs)

# Get timestamps
for result in results:
    print(f"Ayah {result.ayah.ayah_number}: "
          f"{result.start_time:.2f}s - {result.end_time:.2f}s")
```

---

### 3. Example Output

```
Ayah 1: 5.62s - 9.57s
Ayah 2: 10.51s - 14.72s
Ayah 3: 15.45s - 18.53s
Ayah 4: 19.21s - 22.54s
Ayah 5: 23.27s - 28.19s
Ayah 6: 29.00s - 33.07s
Ayah 7: 33.98s - 46.44s
```

---

## Features

* **Whisper Transcription** – Uses faster-whisper as default backend with Quran-tuned models
* **Four Alignment Strategies** – Auto, Hybrid, DP, and Greedy
* **Arabic Text Normalization** – Handles diacritics, hamzas, and Warsh orthographic variations
* **Automatic Drift Correction** – Multi-pass zone realignment for long recordings
* **Quality Metrics** – Confidence scores for each aligned ayah
* **Phonetic Similarity** – Arabic ASR confusion-aware similarity scoring
* **Word-level Precision** – Uses per-word timestamps (when available) to improve drift recovery

---

## Alignment Strategies

The default `auto` strategy works best for most cases. You can override it:

```python
from munajjam.core import Aligner

# Auto (recommended) - picks the best strategy
aligner = Aligner("001.mp3")

# Hybrid - DP with greedy fallback
aligner = Aligner("001.mp3", strategy="hybrid")

# Greedy - fastest, good for clean recordings
aligner = Aligner("001.mp3", strategy="greedy")

# DP - optimal alignment using dynamic programming
aligner = Aligner("001.mp3", strategy="dp")

results = aligner.align(segments, ayahs)
```

---

## Examples

See the [examples](./examples) directory for more usage patterns:

* `01_basic_usage.py` – Simple transcription and alignment
* `02_comparing_strategies.py` – Compare alignment strategies
* `03_advanced_configuration.py` – Custom settings and options
* `04_batch_processing.py` – Process multiple files

---

## Requirements

* Python 3.10+
* PyTorch 2.0+
* FFmpeg (for audio processing)

---

## Community

Original project resources:

* Website: [https://munajjam.itqan.dev](https://munajjam.itqan.dev)
* ITQAN Community: [https://community.itqan.dev](https://community.itqan.dev)

---

## Acknowledgments

This project is based on the original [Munajjam](https://github.com/Itqan-community/munajjam) library by the [ITQAN Community](https://community.itqan.dev).

Special thanks to:

* [Tarteel AI](https://tarteel.ai) for the Quran-specialized Whisper model
* [King Fahd Glorious Qur'an Printing Complex](https://qurancomplex.gov.sa/quran-dev/) for the Warsh Mushaf edition (6214 ayahs)

All original architectural design and alignment logic credit belongs to the original authors.

---

## License

This project remains under the MIT License, following the original Munajjam repository.
