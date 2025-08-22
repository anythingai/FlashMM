# Windows Installation Guide for FlashMM

## Problem Solved ✅

The project now uses **Poetry only** for dependency management, eliminating the confusion of multiple requirements files.

## Installation (Windows + Python 3.11-3.13)

### One-Command Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install Poetry and dependencies
pip install poetry
poetry install
```

### What This Does

1. **Installs Poetry**: Modern Python dependency management
2. **Windows-compatible dependencies**: Automatically excludes `uvloop` (Unix-only)
3. **Development tools**: Installs testing, linting, and development packages
4. **Project installation**: Installs FlashMM in editable mode

## Key Changes Made

### ✅ Cleaned Up Dependency Management
- **Removed**: `requirements.txt`, `requirements-py313-core.txt`
- **Updated**: [`pyproject.toml`](pyproject.toml) for Windows compatibility
- **Single source**: Poetry manages all dependencies

### ✅ Windows Compatibility Fixed
- **Removed**: `uvloop = "^0.17.0"` (Windows incompatible)
- **Added**: Python 3.13 support
- **Updated**: Package versions for compatibility

### ✅ Organized Dependencies
- **Production**: Core application dependencies
- **Development**: Testing, linting, type checking
- **Testing**: Additional test utilities

## Verify Installation

```bash
# Check installed packages
poetry show

# Run tests (if available)
poetry run pytest

# Start the application
poetry run python -m src.flashmm.main
```

## Benefits of Poetry

1. **Dependency resolution**: Handles conflicts automatically
2. **Virtual environment**: Built-in management
3. **Lock file**: Reproducible builds
4. **Development groups**: Organize dev vs production deps
5. **Modern standard**: Industry best practice

This solution eliminates the multiple requirements files confusion and provides a clean, modern dependency management approach.