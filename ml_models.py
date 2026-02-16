"""
Machine Learning Models for PDF Analysis
Enhanced OCR, NER for technical specifications, and intelligent extraction
"""

import io
import re
import math
from typing import List, Dict, Set, Tuple, Optional
from functools import lru_cache

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import pandas as pd

# ML/DL imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ml_config import (
    DEVICE, EASYOCR_LANGUAGES, EASYOCR_GPU, EASYOCR_CONFIDENCE_THRESHOLD,
    NER_MODEL_NAME, NER_CONFIDENCE_THRESHOLD, EASYOCR_MODEL_DIR, 
    NER_CACHE_DIR, LAYOUT_MODEL_NAME, LAYOUT_CONFIDENCE_THRESHOLD,
    LAYOUT_CATEGORIES, TABLE_MODEL_NAME, TABLE_CONFIDENCE_THRESHOLD,
    MARK_HIGHLIGHT_COLOR, MEASUREMENT_HIGHLIGHT_COLOR, get_model_info
)

def normalize_text(text: str) -> str:
    """Normalize common PDF dash and space variations."""
    if not text: return ""
    return text.replace("—", "-").replace("–", "-").strip()

# ======================================================
# MODEL INITIALIZATION (Lazy Loading)
# ======================================================

_easyocr_reader = None
_ner_pipeline = None
_layout_pipeline = None
_table_pipeline = None

def get_easyocr_reader():
    """Lazy initialization of EasyOCR reader"""
    global _easyocr_reader
    if not EASYOCR_AVAILABLE: return None
    if _easyocr_reader is None:
        with st.spinner("🔄 Loading EasyOCR..."):
            _easyocr_reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=EASYOCR_GPU, model_storage_directory=str(EASYOCR_MODEL_DIR))
    return _easyocr_reader

def get_ner_pipeline():
    """Lazy initialization of NER pipeline"""
    global _ner_pipeline
    if not TRANSFORMERS_AVAILABLE: return None
    if _ner_pipeline is None:
        with st.spinner("🔄 Loading NER..."):
            device = 0 if DEVICE == 'cuda' else -1
            _ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, tokenizer=NER_MODEL_NAME, device=device, aggregation_strategy="simple")
    return _ner_pipeline

def get_layout_pipeline():
    """Lazy initialization of DLA pipeline"""
    global _layout_pipeline
    if not TRANSFORMERS_AVAILABLE: return None
    if _layout_pipeline is None:
        with st.spinner(f"🔄 Loading Layout Analysis..."):
            device = 0 if DEVICE == 'cuda' else -1
            _layout_pipeline = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
    return _layout_pipeline

def get_table_pipeline():
    """Lazy initialization of Table Recognition pipeline"""
    global _table_pipeline
    if not TRANSFORMERS_AVAILABLE: return None
    if _table_pipeline is None:
        with st.spinner(f"🔄 Loading Table Transformer..."):
            device = 0 if DEVICE == 'cuda' else -1
            _table_pipeline = pipeline("object-detection", model=TABLE_MODEL_NAME, device=device)
    return _table_pipeline

# ======================================================
# PHASE 3 CORE LOGIC
# ======================================================

def detect_document_layout(image_array: np.ndarray) -> List[Dict]:
    """Detect layout regions using ML."""
    try:
        pipeline = get_layout_pipeline()
        if not pipeline: return []
        results = pipeline(Image.fromarray(image_array))
        
        layout_regions = []
        for res in results:
            if res.get('score', 0) >= LAYOUT_CONFIDENCE_THRESHOLD:
                label = res.get('label', '').lower()
                if any(x in label for x in ['drawing', 'diagram', 'chart', 'image']): category = "Diagram"
                elif 'table' in label: category = "Schedule"
                else: category = "Text"
                layout_regions.append({'category': category, 'box': res.get('box'), 'confidence': res.get('score')})
        return layout_regions
    except Exception as e:
        st.warning(f"DLA failed: {str(e)}")
        return []

def is_mechanical_plan(page_text: str, layout_regions: List[Dict]) -> bool:
    """Check if page is a Mechanical Plan diagram."""
    page_text_upper = page_text.upper()
    
    # Strong keywords that almost certainly mean it's a mechanical page
    mech_keywords = [
        'MECHANICAL', 'M-PLANE', 'HVAC', 'DUCT', 'PIPING', 'PLUMBING', 
        'M-0', 'M-1', 'M-2', 'M-3', 'SHEET', 'PLAN', 'ELEVATION', 'SECTION',
        'EQUIPMENT', 'CHILLER', 'PUMP', 'UNIT', 'BOILER', 'FAN'
    ]
    # Common codes or abbreviations in mechanical drawings
    mech_codes = [r'M-\d+', r'MP-\d+', r'HVAC-\d+', r'MD-\d+', r'M\d+', r'MP\d+']
    
    has_mech_text = any(k in page_text_upper for k in mech_keywords) or \
                    any(re.search(code, page_text_upper) for code in mech_codes)
    
    # Check for "Architectural" or other types to avoid false positives
    is_arch = any(k in page_text_upper for k in ['ARCHITECTURAL', 'FLOOR FINISH', 'F-PSET', 'A-1', 'A-2', 'A1', 'A2'])
    
    # Check if DLA found a diagram
    has_diagram = any(r['category'] == "Diagram" for r in layout_regions)
    
    # If it's architectural and we don't see strong mech text, skip
    if is_arch and not any(k in page_text_upper for k in ['HVAC', 'DUCT', 'MECHANICAL']):
        return False
        
    if has_mech_text:
        # If we have mechanical text, we are quite sure.
        # Fallback: proceed even if DLA missed the diagram region (often happens with lines).
        is_schedule = any(k in page_text_upper for k in ['SCHEDULE', 'LIST', 'NOTES']) and not has_diagram
        if is_schedule and 'PLAN' not in page_text_upper:
            return False
        return True
        
    return has_diagram # Last resort: if DLA sees a diagram, maybe it's mech?

def apply_conditional_highlighting(pdf_bytes: bytes, marks: Set[str], measurements_df: pd.DataFrame, ocr_results: Optional[List[Dict]] = None) -> bytes:
    """Apply highlighting ONLY to Mechanical Plan Diagrams."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    highlight_count = 0
    for page_index, page in enumerate(doc):
        # Determine text and words for this page
        if ocr_results and page_index < len(ocr_results):
            text = ocr_results[page_index]['text']
            ocr_words = ocr_results[page_index]['words']
        else:
            text = page.get_text()
            ocr_words = []
        
        # Render for DLA (low DPI for speed)
        pix = page.get_pixmap(dpi=72)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        layout_regions = detect_document_layout(img_array)
        
        is_mech = is_mechanical_plan(text, layout_regions)
        if is_mech:
            st.info(f"📍 Highlighting mechanical diagram on Page {page_index + 1}")
            
            # 1. Marks (Light Blue)
            for mark in marks:
                if not mark: continue
                match_found = False
                
                # Try Digital Search First
                for variant in [mark, mark.replace("-", " "), mark.replace("-", "")]:
                    rects = page.search_for(variant)
                    if rects:
                        for r in rects:
                            annot = page.add_highlight_annot(r)
                            annot.set_colors(stroke=MARK_HIGHLIGHT_COLOR)
                            annot.update()
                            highlight_count += 1
                        match_found = True
                
                # If digital search failed and we have OCR words, try coordinate-based match
                if not match_found and ocr_words:
                    for ow in ocr_words:
                        if mark.upper() in ow['text'].upper():
                            annot = page.add_highlight_annot(ow['bbox'])
                            annot.set_colors(stroke=MARK_HIGHLIGHT_COLOR)
                            annot.update()
                            highlight_count += 1
            
            # 2. Measurements (Light Orange)
            if not measurements_df.empty:
                page_meas = measurements_df[measurements_df['page'] == (page_index + 1)]
                for _, row in page_meas.iterrows():
                    m_raw = row['raw']
                    if not m_raw: continue
                    
                    match_found = False
                    m_variants = {m_raw, m_raw.replace('"', '\"'), m_raw.replace('Ø', 'O'), m_raw.replace(' ', '')}
                    
                    for v in m_variants:
                        rects = page.search_for(v)
                        if rects:
                            for r in rects:
                                annot = page.add_highlight_annot(r)
                                annot.set_colors(stroke=MEASUREMENT_HIGHLIGHT_COLOR)
                                annot.update()
                                highlight_count += 1
                            match_found = True
                            break
                    
                    # OCR Fallback for measurements
                    if not match_found and ocr_words:
                        # For measurements, we often have multiple words. Match if the first word exists.
                        m_first_word = m_raw.split()[0]
                        for ow in ocr_words:
                            if m_first_word in ow['text']:
                                annot = page.add_highlight_annot(ow['bbox'])
                                annot.set_colors(stroke=MEASUREMENT_HIGHLIGHT_COLOR)
                                annot.update()
                                highlight_count += 1
        else:
            reason = "Not a mechanical plan" if not any(k in text.upper() for k in ['MECHANICAL', 'M-', 'MP-']) else "No diagram detected"
            st.write(f"⏩ Page {page_index + 1} skipped: {reason}")
    
    if highlight_count == 0:
        st.warning("⚠️ No highlights were added. Check if 'Highlight Diagram Pages Only' is too strict.")

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()

# ======================================================
# LEGACY WRAPPERS & UTILS
# ======================================================

def enhanced_ocr_pdf(pdf_bytes: bytes, target_dpi=300) -> List[Dict]:
    """Perform OCR on a PDF and return collective text and coordinate metadata."""
    st.info("🖼️ Scanned PDF detected. Running OCR with coordinate tracking...")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ocr_results = []
    
    with st.status("📷 Running OCR and mapping coordinates...") as status:
        for i, page in enumerate(doc):
            st.write(f"Processing Page {i+1}...")
            pix = page.get_pixmap(dpi=target_dpi)
            img = Image.fromarray(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
            
            # Use Tesseract to get detailed data including coordinates
            # config='--psm 1' can help with technical drawings
            tess_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            page_text = []
            page_words = []
            
            # Scale coordinates back to PDF points
            # PDF units are usually 72 DPI. target_dpi is what we rendered at.
            scale = 72.0 / target_dpi
            
            for j in range(len(tess_data['text'])):
                word = tess_data['text'][j].strip()
                if word:
                    page_text.append(word)
                    # Convert pixel [L, T, W, H] to PDF points [X1, Y1, X2, Y2]
                    left = tess_data['left'][j] * scale
                    top = tess_data['top'][j] * scale
                    width = tess_data['width'][j] * scale
                    height = tess_data['height'][j] * scale
                    page_words.append({
                        "text": word,
                        "bbox": [left, top, left + width, top + height]
                    })
            
            ocr_results.append({
                "text": " ".join(page_text),
                "words": page_words
            })
            
        status.update(label="OCR and Mapping Complete!", state="complete")
        
    doc.close()
    return ocr_results

def hybrid_mark_extraction(text: str, use_ml: bool = True) -> Tuple[Set[str], Dict[str, int]]:
    mark_regex = re.compile(r'\b[A-Z]{1,4}-\d+\b', re.IGNORECASE)
    regex_marks = set(m.upper() for m in mark_regex.findall(text))
    ml_marks = set()
    if use_ml and TRANSFORMERS_AVAILABLE:
        ner = get_ner_pipeline()
        if ner:
            try:
                entities = ner(text)
                for ent in entities:
                    if ent['score'] >= NER_CONFIDENCE_THRESHOLD:
                        m = mark_regex.search(ent['word'])
                        if m: ml_marks.add(m.group().upper())
            except: pass
    combined = regex_marks.union(ml_marks)
    return combined, {'regex_only': len(regex_marks - ml_marks), 'ml_only': len(ml_marks - regex_marks), 'both': len(regex_marks & ml_marks), 'total': len(combined)}

def extract_measurements_with_ner(text: str) -> List[str]:
    return re.findall(r'\d+"\s*Ø\s*/\s*\d+|\d+"\s*x\s*\d+"\s*/\s*\d+|\d+"\s*x\s*\d+|\d+"\s*Ø', text, re.I)

def check_ml_availability() -> Dict[str, bool]:
    return {'easyocr': EASYOCR_AVAILABLE, 'transformers': TRANSFORMERS_AVAILABLE, 'device': DEVICE}

def get_ml_stats() -> Dict:
    return {'models_loaded': {'easyocr': _easyocr_reader is not None, 'ner': _ner_pipeline is not None, 'layout': _layout_pipeline is not None, 'table': _table_pipeline is not None}, 'config': get_model_info(), 'availability': check_ml_availability()}
