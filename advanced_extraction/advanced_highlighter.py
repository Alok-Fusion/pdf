"""
Advanced Highlighter - Each tag gets its own unique color.
Supports both digital text AND scanned pages (uses OCR coordinates).
Only highlights on mechanical floor plan pages.
"""

import io
import fitz  # PyMuPDF
from typing import Dict

from .advanced_config import get_color_for_tag, MEASUREMENT_COLOR, reset_tag_colors


def highlight_with_tags(pdf_bytes: bytes, extraction_results: Dict) -> bytes:
    """Highlight on mech floor plan pages. Uses OCR coords for scanned pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    diagram_pages = set(extraction_results.get("diagram_pages", []))
    ocr_pages = extraction_results.get("ocr_pages", {})

    reset_tag_colors()

    # Per-page lookups
    excel_by_page = {}
    for item in extraction_results.get("excel_matches", []):
        excel_by_page.setdefault(item["page"], []).append(item["value"])

    marks_by_page = {}
    for item in extraction_results.get("marks_found", []):
        marks_by_page.setdefault(item["page"], []).append(item["mark"])

    meas_by_page = {}
    for item in extraction_results.get("measurements", []):
        meas_by_page.setdefault(item["page"], []).append(item["value"])

    for page_num in diagram_pages:
        if page_num > len(doc):
            continue
        page = doc[page_num - 1]

        # Check if this is a scanned page with OCR data
        ocr_data = ocr_pages.get(page_num)
        ocr_words = ocr_data["words"] if ocr_data else None

        # 1. Excel values — each tag gets its own color
        for val in excel_by_page.get(page_num, []):
            color = get_color_for_tag(val)
            _smart_highlight(page, val, color, ocr_words)

        # 2. Auto marks
        for mark in marks_by_page.get(page_num, []):
            color = get_color_for_tag(mark)
            _smart_highlight(page, mark, color, ocr_words)

        # 3. Measurements — orange
        for meas in meas_by_page.get(page_num, []):
            _smart_highlight(page, meas, MEASUREMENT_COLOR, ocr_words)

    out = io.BytesIO()
    doc.save(out)
    out.seek(0)
    doc.close()
    return out.getvalue()


def _smart_highlight(page, text: str, color: tuple, ocr_words=None):
    """
    Highlight text on a page.
    For digital pages: uses page.search_for()
    For scanned pages: uses OCR word bounding boxes
    """
    if not text or len(text) < 2:
        return

    # Try digital text search first
    variants = [text, text.replace("-", " "), text.replace("-", "")]
    for v in variants:
        quads = page.search_for(v)
        if quads:
            for q in quads:
                annot = page.add_highlight_annot(q)
                annot.set_colors(stroke=color)
                annot.update()
            return

    # Fallback: OCR word coordinate matching
    if ocr_words:
        text_upper = text.upper()
        for word_info in ocr_words:
            word_text = word_info["text"].upper()
            # Check if the OCR word matches the search term
            if text_upper in word_text or word_text in text_upper:
                bbox = word_info["bbox"]
                if bbox.width > 0 and bbox.height > 0:
                    annot = page.add_highlight_annot(bbox)
                    annot.set_colors(stroke=color)
                    annot.update()
