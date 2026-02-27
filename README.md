# Munajjam-Warsh

**A Python library to synchronize Quran ayat with audio recitations — Warsh (6214 ayahs edition).**

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

For faster transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

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

# Align
results = align("001.mp3", segments, ayahs)

for result in results:
    print(f"Ayah {result.ayah.ayah_number}: "
          f"{result.start_time:.2f}s - {result.end_time:.2f}s")
```

---

## Features

* 🎙 Whisper Transcription (faster-whisper backend supported)
* 🧠 AI-powered ayah alignment
* 📖 Warsh (6214 ayahs) dataset support
* 🔍 Multiple alignment strategies (Auto, Hybrid, DP, Greedy)
* 🔄 Automatic drift correction
* 📊 Quality confidence scoring
* 🔤 Arabic text normalization (Warsh-aware)

---

## Requirements

* Python 3.10+
* PyTorch 2.0+
* FFmpeg

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
