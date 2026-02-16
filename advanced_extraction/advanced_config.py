"""
Advanced Extraction Configuration
Each unique tag gets its own distinct light color.
"""

import re

# ======================================================
# MARK/TAG DETECTION PATTERNS
# ======================================================

MARK_REGEX = re.compile(r'\b[A-Z]{1,4}-\d+[A-Z]?\b', re.IGNORECASE)

EXTENDED_MARK_PATTERNS = [
    re.compile(r'\bAHU-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bFCU-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bVAV-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bEF-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bRTU-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bCH-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bP-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bHWP-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bCWP-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bB-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bCT-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bMAU-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bHRU-?\d+[A-Z]?\b', re.IGNORECASE),
    re.compile(r'\bDX-?\d+[A-Z]?\b', re.IGNORECASE),
]

# Measurement patterns
MEASUREMENT_PATTERNS = [
    r'\d+"\s*Ø\s*/\s*\d+',
    r'\d+"\s*x\s*\d+"\s*/\s*\d+',
    r'\d+"\s*x\s*\d+"',
    r'\d+"\s*Ø',
    r'\d+\s*[Dd][Ii][Aa]\.?',
    r'\d+\s*[Gg][Aa][Ll]',
    r'\d+\s*[Tt][Oo][Nn]',
    r'\d+\s*[Cc][Ff][Mm]',
    r'\d+\s*[Gg][Pp][Mm]',
    r'\d+\s*[Mm][Bb][Hh]',
    r'\d+\s*[Bb][Tt][Uu]',
    r'\d+\s*[Hh][Pp]',
    r'\d+\s*[Kk][Ww]',
]

COMBINED_MEASUREMENT_REGEX = re.compile(
    '|'.join(MEASUREMENT_PATTERNS), re.IGNORECASE
)

# ======================================================
# TAG COLOR PALETTE – 30 distinct light/pastel colors
# Each unique tag value (VAV-1, AHU-2, etc.) gets its own color
# ======================================================

TAG_COLOR_PALETTE = [
    (0.55, 0.82, 1.0),    # Sky Blue
    (0.70, 1.0,  0.70),   # Mint Green
    (1.0,  0.80, 0.55),   # Warm Gold
    (0.85, 0.70, 1.0),    # Lavender
    (1.0,  0.70, 0.70),   # Salmon
    (0.55, 1.0,  0.90),   # Aqua
    (1.0,  0.90, 0.55),   # Light Yellow
    (0.90, 0.65, 0.90),   # Orchid
    (0.65, 0.95, 0.65),   # Pale Green
    (1.0,  0.75, 0.85),   # Pink
    (0.75, 0.90, 1.0),    # Periwinkle
    (0.85, 1.0,  0.65),   # Lime
    (1.0,  0.65, 0.80),   # Hot Pink
    (0.65, 1.0,  0.75),   # Spring Green
    (0.95, 0.80, 1.0),    # Mauve
    (1.0,  0.85, 0.65),   # Peach
    (0.65, 0.85, 0.95),   # Steel Blue
    (0.80, 1.0,  0.80),   # Honeydew
    (1.0,  0.70, 0.55),   # Coral
    (0.70, 0.80, 1.0),    # Cornflower
    (0.90, 1.0,  0.70),   # Pale Lime
    (1.0,  0.60, 0.70),   # Rose
    (0.60, 0.90, 0.85),   # Teal Light
    (0.95, 0.75, 0.60),   # Tan
    (0.75, 0.75, 1.0),    # Slate Blue
    (0.80, 1.0,  0.55),   # Chartreuse
    (1.0,  0.80, 0.80),   # Misty Rose
    (0.55, 0.90, 1.0),    # Deep Sky
    (0.90, 0.90, 0.55),   # Khaki
    (0.80, 0.60, 0.95),   # Medium Purple
]

# Fixed colors for non-tag items
MEASUREMENT_COLOR = (1.0, 0.85, 0.70)  # Light Orange

# ======================================================
# COLOR ASSIGNMENT
# ======================================================

_tag_color_map = {}  # tag_value -> color

def get_color_for_tag(tag_value: str) -> tuple:
    """Assign a unique color to each tag. Same tag always gets same color."""
    key = tag_value.upper()
    if key not in _tag_color_map:
        idx = len(_tag_color_map) % len(TAG_COLOR_PALETTE)
        _tag_color_map[key] = TAG_COLOR_PALETTE[idx]
    return _tag_color_map[key]

def reset_tag_colors():
    """Reset color assignments (call before each new PDF processing)."""
    global _tag_color_map
    _tag_color_map = {}

def get_tag_color_legend() -> dict:
    """Return current tag -> color mapping for display."""
    return dict(_tag_color_map)

# ======================================================
# EXCEL COLUMN PATTERNS
# ======================================================

TAG_COLUMN_PATTERNS = [
    r'(?i)mark', r'(?i)tag', r'(?i)equip', r'(?i)item',
    r'(?i)label', r'(?i)designation', r'(?i)code', r'(?i)symbol',
]
