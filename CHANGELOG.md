# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0a1] - 2025-01-12

### Added
- Unified `Aligner` class as the main interface for alignment operations
- Hybrid alignment algorithm with automatic fallback strategies
- Zone re-alignment feature for improved accuracy
- Cascade recovery system for error handling
- Support for `faster-whisper` as optional dependency
- Comprehensive test suite (unit and integration tests)
- Type hints throughout the codebase
- Pydantic models for data validation

### Changed
- Complete project restructure into proper Python package
- Refactored alignment processing for better maintainability
- Improved Arabic text normalization
- Enhanced documentation with algorithm descriptions

### Removed
- Deprecated files and legacy scripts
- Old alignment implementations in favor of unified interface

## [1.0.0] - 2024

### Added
- Initial release
- Basic Quran audio alignment functionality
- Whisper-based transcription
- Greedy and dynamic programming alignment strategies

[Unreleased]: https://github.com/abdullahmosaibah/munajjam/compare/v2.0.0a1...HEAD
[2.0.0a1]: https://github.com/abdullahmosaibah/munajjam/compare/v1.0.0...v2.0.0a1
[1.0.0]: https://github.com/abdullahmosaibah/munajjam/releases/tag/v1.0.0
