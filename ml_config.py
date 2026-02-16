"""
ML/DL Configuration for PDF Analyzer
Contains model settings, paths, and utilities for ML features
"""

import os
import torch
from pathlib import Path

# ======================================================
# DEVICE CONFIGURATION
# ======================================================

def get_device():
    """
    Automatically detect and return the best available device.
    Prioritizes: CUDA GPU > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

DEVICE = get_device()

# ======================================================
# MODEL PATHS AND CACHE
# ======================================================

# Base directory for model cache
MODEL_CACHE_DIR = Path.home() / '.cache' / 'pdf_analyzer_models'
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# EasyOCR settings
EASYOCR_MODEL_DIR = MODEL_CACHE_DIR / 'easyocr'
EASYOCR_LANGUAGES = ['en']
EASYOCR_GPU = DEVICE == 'cuda'

# NER Model settings
NER_MODEL_NAME = "dslim/bert-base-NER"
NER_CACHE_DIR = MODEL_CACHE_DIR / 'transformers'

# ======================================================
# PHASE 3 MODEL SETTINGS
# ======================================================

# LayoutLMv3 (DLA) settings
LAYOUT_MODEL_NAME = "microsoft/layoutlmv3-base"
LAYOUT_CONFIDENCE_THRESHOLD = 0.4
LAYOUT_CATEGORIES = {
    "Diagram": (0.7, 0.7, 1.0),    # Light Blue
    "Schedule": (0.7, 1.0, 0.7),   # Light Green
    "Legend": (1.0, 0.7, 1.0),     # Light Purple
    "Text": (0.9, 0.9, 0.9),       # Light Grey
}

# TableTransformer settings
TABLE_MODEL_NAME = "microsoft/table-transformer-structure-recognition"
TABLE_CONFIDENCE_THRESHOLD = 0.5

# ======================================================
# PROCESSING AND HIGHLIGHTING
# ======================================================

# OCR settings
EASYOCR_BATCH_SIZE = 4
EASYOCR_WORKERS = 1
EASYOCR_CONFIDENCE_THRESHOLD = 0.2

# NER settings
NER_CONFIDENCE_THRESHOLD = 0.5
NER_MAX_LENGTH = 512

# Highlighting colors (Pastel)
MARK_HIGHLIGHT_COLOR = (0.7, 0.85, 1.0)      # Light Blue
MEASUREMENT_HIGHLIGHT_COLOR = (1.0, 0.85, 0.7) # Light Orange

# Performance settings
ENABLE_MODEL_CACHING = True
ENABLE_RESULT_CACHING = False

def get_model_info():
    """Return dictionary with current ML configuration"""
    return {
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cache_dir': str(MODEL_CACHE_DIR),
        'easyocr_gpu': EASYOCR_GPU,
        'ner_model': NER_MODEL_NAME,
        'layout_model': LAYOUT_MODEL_NAME,
        'table_model': TABLE_MODEL_NAME,
    }
