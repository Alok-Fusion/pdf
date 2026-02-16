"""
Core Extraction Engine - With OCR fallback for scanned pages.
Only highlights on mechanical floor plan pages.
"""

import io
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, Optional, List

from .advanced_config import (
    MARK_REGEX, EXTENDED_MARK_PATTERNS, COMBINED_MEASUREMENT_REGEX,
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\u2014", "-").replace("\u2013", "-").strip()


# ======================================================
# OCR FOR SCANNED PAGES
# ======================================================

def ocr_page(page, dpi=300) -> dict:
    """
    Run OCR on a single PDF page. Returns text and word coordinates.
    Coordinates are scaled back to PDF points (72 DPI).
    """
    pix = page.get_pixmap(dpi=dpi)
    img = Image.fromarray(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    )

    # Get detailed OCR data with coordinates
    tess_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    scale = 72.0 / dpi  # Convert pixel coords to PDF points
    words = []
    text_parts = []

    for j in range(len(tess_data['text'])):
        word = tess_data['text'][j].strip()
        if word:
            text_parts.append(word)
            left = tess_data['left'][j] * scale
            top = tess_data['top'][j] * scale
            width = tess_data['width'][j] * scale
            height = tess_data['height'][j] * scale
            words.append({
                "text": word,
                "bbox": fitz.Rect(left, top, left + width, top + height)
            })

    return {
        "text": " ".join(text_parts),
        "words": words,
        "is_ocr": True
    }


# ======================================================
# STRICT: Mechanical Floor Plan Page Detection
# ======================================================

def is_mechanical_floor_plan(page_text: str) -> tuple:
    upper = page_text.upper()

    has_mechanical = any(kw in upper for kw in [
        'MECHANICAL', 'HVAC', 'M-0', 'M-1', 'M-2', 'M-3', 'M-4',
        'M-5', 'M-6', 'M-7', 'M-8', 'M-9',
    ]) or bool(re.search(r'\bM-?\d+', upper))

    has_floor_plan = any(kw in upper for kw in [
        'FLOOR PLAN', 'PLAN', 'LAYOUT', 'ROOF PLAN',
        'REFLECTED CEILING', 'CEILING PLAN',
        '1ST FLOOR', '2ND FLOOR', '3RD FLOOR', 'FIRST FLOOR',
        'SECOND FLOOR', 'THIRD FLOOR', 'GROUND FLOOR',
        'BASEMENT', 'MEZZANINE', 'PENTHOUSE',
    ])

    is_reject = any(kw in upper for kw in [
        'SCHEDULE', 'SPECIFICATIONS', 'GENERAL NOTES', 'NOTES',
        'TABLE OF CONTENTS', 'INDEX', 'ABBREVIATIONS',
        'COVER SHEET', 'TITLE SHEET', 'LEGEND',
        'RISER DIAGRAM', 'SCHEMATIC', 'FLOW DIAGRAM',
    ])

    if is_reject:
        return False, "Schedule/notes/detail page"
    if has_mechanical and has_floor_plan:
        return True, "Mechanical floor plan"

    is_section = any(kw in upper for kw in [
        'SECTION', 'ELEVATION', 'DETAIL', 'ENLARGED',
    ])
    if is_section and not has_floor_plan:
        return False, "Section/elevation/detail page"

    if has_mechanical and len(page_text.strip()) < 500:
        return True, "Mechanical drawing (low text)"
    if has_mechanical:
        return False, "Mechanical (not a floor plan)"

    return False, "Non-mechanical page"


class AutoExtractor:
    def __init__(self, pdf_bytes: bytes, excel_data: Optional[Dict] = None):
        self.pdf_bytes = pdf_bytes
        self.excel_data = excel_data
        self.extraction_results = {
            "marks_found": [],
            "measurements": [],
            "excel_matches": [],
            "missing_excel_values": [],
            "page_summary": [],
            "diagram_pages": [],
            "skipped_pages": [],
            "ocr_pages": {},  # page_num -> ocr_data (words with coords)
        }

    def extract_all(self, progress_callback=None) -> Dict:
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        excel_values = []
        if self.excel_data:
            for item in self.excel_data.get("all_values", []):
                val = str(item["value"]).strip()
                if val and len(val) >= 2:
                    excel_values.append(val)

        found_excel = set()

        for i, page in enumerate(doc):
            p_num = i + 1
            if progress_callback:
                progress_callback(p_num, total_pages)

            # --- Get text: digital first, OCR fallback ---
            digital_text = page.get_text()
            is_scanned = len(digital_text.strip()) < 50

            if is_scanned:
                # Scanned page → run OCR
                ocr_data = ocr_page(page, dpi=200)  # 200 DPI for speed
                p_text = normalize_text(ocr_data["text"])
                self.extraction_results["ocr_pages"][p_num] = ocr_data
            else:
                p_text = normalize_text(digital_text)

            # --- Check if mechanical floor plan ---
            is_mfp, reason = is_mechanical_floor_plan(p_text)
            if is_scanned and not is_mfp:
                # For scanned pages, be more lenient — check with lower threshold
                # If OCR found very little text, it might still be a drawing
                if len(p_text.strip()) < 100:
                    is_mfp = True
                    reason = "Scanned drawing (OCR sparse)"

            page_marks = []
            page_meas = []
            page_excel = []

            if is_mfp:
                self.extraction_results["diagram_pages"].append(p_num)

                # Auto marks
                seen = set()
                for pattern in [MARK_REGEX] + EXTENDED_MARK_PATTERNS:
                    for m in pattern.findall(p_text):
                        mu = m.strip().upper()
                        if mu not in seen:
                            seen.add(mu)
                            page_marks.append(mu)
                            self.extraction_results["marks_found"].append(
                                {"mark": mu, "page": p_num,
                                 "source": "ocr" if is_scanned else "auto"}
                            )

                # Measurements
                for meas in COMBINED_MEASUREMENT_REGEX.findall(p_text):
                    v = meas.strip()
                    page_meas.append(v)
                    self.extraction_results["measurements"].append(
                        {"value": v, "page": p_num}
                    )

                # Excel values
                if self.excel_data:
                    p_upper = p_text.upper()
                    for ev in excel_values:
                        if ev.upper() in p_upper:
                            found_excel.add(ev.upper())
                            page_excel.append(ev)
                            self.extraction_results["excel_matches"].append({
                                "value": ev, "page": p_num,
                            })
            else:
                self.extraction_results["skipped_pages"].append(
                    {"page": p_num, "reason": reason}
                )

            self.extraction_results["page_summary"].append({
                "Page": p_num,
                "Type": "Mech Floor Plan" if is_mfp else "Skipped",
                "Scanned": "Yes (OCR)" if is_scanned else "No",
                "Reason": reason if not is_mfp else "✅ Mechanical Floor Plan",
                "Marks": len(page_marks),
                "Measurements": len(page_meas),
                "Excel Hits": len(page_excel) if self.excel_data else "N/A",
            })

        doc.close()

        # Dedup
        seen_m = set()
        unique = []
        for m in self.extraction_results["excel_matches"]:
            key = (m["value"], m["page"])
            if key not in seen_m:
                seen_m.add(key)
                unique.append(m)
        self.extraction_results["excel_matches"] = unique

        if self.excel_data:
            self.extraction_results["missing_excel_values"] = sorted(
                [ev for ev in excel_values if ev.upper() not in found_excel]
            )

        return self.extraction_results
