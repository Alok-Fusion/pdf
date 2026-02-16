import io
import re
import json
import zipfile
import math
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
import cv2
import numpy as np
import traceback


# --------- STANDALONE OCR (no ML dependency) ---------

def ocr_page_with_tesseract(pdf_bytes, page_index, target_dpi=100):
    """OCR a single PDF page using pytesseract. Returns {text, words} with PDF-point coords."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    pix = page.get_pixmap(dpi=target_dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold (CRUCIAL for drawings)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    img = Image.fromarray(thresh)

    doc.close()
    custom_config = r'--oem 3 --psm 6'

    tess_data = pytesseract.image_to_data(
        img,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    scale = 72.0 / target_dpi

    page_text_parts = []
    page_words = []
    for j in range(len(tess_data['text'])):
        word = tess_data['text'][j].strip()
        conf = int(tess_data['conf'][j]) if str(tess_data['conf'][j]).lstrip('-').isdigit() else 0
        if word and conf > 0:
            page_text_parts.append(word)
            left   = tess_data['left'][j]   * scale
            top    = tess_data['top'][j]    * scale
            width  = tess_data['width'][j]  * scale
            height = tess_data['height'][j] * scale
            page_words.append({
                "text": word,
                "bbox": fitz.Rect(left, top, left + width, top + height),
                "conf": conf,
            })

    return {
        "text": " ".join(page_text_parts),
        "words": page_words,
    }

# ML/DL imports
try:
    from ml_models import (
        enhanced_ocr_pdf, hybrid_mark_extraction, 
        extract_measurements_with_ner, check_ml_availability,
        get_ml_stats, detect_document_layout, is_mechanical_plan,
        apply_conditional_highlighting, normalize_text
    )
    ML_AVAILABLE = True
    ML_ERROR = None
except Exception as e:
    ML_AVAILABLE = False
    ML_ERROR = str(e)
    ML_TRACEBACK = traceback.format_exc()

    def normalize_text(text):
        if not text: return ""
        return text.replace("\u2014", "-").replace("\u2013", "-").strip()

    def enhanced_ocr_pdf(pdf_bytes, target_dpi=300):
        """Fallback OCR using pytesseract when ML models are unavailable."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n_pages = len(doc)
        doc.close()
        results = []
        for i in range(n_pages):
            results.append(ocr_page_with_tesseract(pdf_bytes, i, target_dpi))
        return results

    def hybrid_mark_extraction(text, use_ml=False):
        """Fallback: regex-only mark extraction."""
        marks = set(m.upper() for m in MARK_REGEX.findall(normalize_text(text)))
        return marks, {"regex_only": len(marks), "ml_only": 0, "both": 0, "total": len(marks)}

    def check_ml_availability():
        return {"easyocr": False, "transformers": False, "device": "cpu"}

    def get_ml_stats():
        return {}

    def detect_document_layout(img):
        return []

    def is_mechanical_plan(text, regions):
        return True

    def apply_conditional_highlighting(*a, **kw):
        return None

    def extract_measurements_with_ner(text):
        return []

st.set_page_config(layout="wide", page_title="Mechanical PDF Analyzer Pro v4")

# --------- REGEX & UTILS ---------
MARK_REGEX = re.compile(r'\b[A-Z]{1,4}-\d+\b', re.IGNORECASE)
MEAS_PATTERN = r'\d+"\s*Ø\s*/\s*\d+|\d+"\s*x\s*\d+"\s*/\s*\d+|\d+"\s*x\s*\d+"|\d+"\s*Ø'

# --------- 30 UNIQUE TAG COLORS ---------
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
MEASUREMENT_COLOR = (1.0, 0.85, 0.70)  # Orange for measurements

_tag_color_map = {}
def get_color_for_tag(tag_value):
    key = tag_value.upper()
    if key not in _tag_color_map:
        idx = len(_tag_color_map) % len(TAG_COLOR_PALETTE)
        _tag_color_map[key] = TAG_COLOR_PALETTE[idx]
    return _tag_color_map[key]


# --------- EXCEL TAG READER ---------

def load_tags_from_excel(file_content):
    """Read ALL values from Excel, skip last 2 columns."""
    try:
        xls = pd.ExcelFile(file_content)
    except Exception as e:
        return {"error": str(e)}

    all_values = []
    seen = set()
    columns_used = []
    columns_skipped = []

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
            if df.empty:
                continue
            cols = df.columns.tolist()
            if len(cols) > 2:
                use_cols = cols[:-2]
                skip_cols = cols[-2:]
            else:
                use_cols = cols
                skip_cols = []

            if not columns_used:
                columns_used = use_cols
                columns_skipped = skip_cols

            for _, row in df.iterrows():
                for col in use_cols:
                    val = str(row[col]).strip()
                    if not val or val.lower() in ['nan', 'none', '', 'total', 'grand total']:
                        continue
                    if val.isdigit() and len(val) <= 2:
                        continue
                    if len(val) < 2:
                        continue
                    key = val.upper()
                    if key not in seen:
                        seen.add(key)
                        all_values.append(val)
        except:
            continue

    return {
        "all_values": all_values,
        "columns_used": columns_used,
        "columns_skipped": columns_skipped,
        "total_count": len(all_values),
    }


def extract_marks(text):
    return sorted(list(set(m.upper() for m in MARK_REGEX.findall(normalize_text(text)))))


# --------- CORE PROCESSING ---------

def create_export_package(original_pdf, highlighted_pdf, marks_df, measurements_df, legends, file_name, excel_matches_df=None, missing_tags=None):
    """Create ZIP with all results."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(f"output/{file_name.rsplit('.',1)[0]}_highlighted.pdf", highlighted_pdf)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            summary_data = {
                "File Name": [file_name],
                "Total Marks": [len(marks_df)],
                "Total Measurements": [len(measurements_df)],
                "Legends Found": [len(legends) if legends else 0]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            if not marks_df.empty: marks_df.to_excel(writer, sheet_name="Marks", index=False)
            if not measurements_df.empty: measurements_df.to_excel(writer, sheet_name="Measurements", index=False)
            if legends: pd.DataFrame([{"Symbol": k, "Meaning": v} for k, v in legends.items()]).to_excel(writer, sheet_name="Legends", index=False)
            if excel_matches_df is not None and not excel_matches_df.empty:
                excel_matches_df.to_excel(writer, sheet_name="Excel_Matches", index=False)
            if missing_tags:
                pd.DataFrame({"Missing Tag": missing_tags}).to_excel(writer, sheet_name="Missing_Tags", index=False)
        zipf.writestr("data/summary.xlsx", excel_buffer.getvalue())

        comprehensive_json = {
            "file_name": file_name,
            "marks": marks_df.to_dict('records') if not marks_df.empty else [],
            "measurements": measurements_df.to_dict('records') if not measurements_df.empty else [],
            "legends": legends
        }
        zipf.writestr("data/data.json", json.dumps(comprehensive_json, indent=2).encode("utf-8"))

    zip_buffer.seek(0)
    return zip_buffer.getvalue(), excel_buffer.getvalue(), comprehensive_json


# --------- STREAMLIT UI ---------

st.title("🔧 Mechanical PDF Analyzer Pro (v4.0)")
st.markdown("All features from v3 + **Excel Tag Matching** + **Unique Colors Per Tag**")

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_file = st.file_uploader("📄 Upload Mechanical PDF", type=["pdf"])
with col_up2:
    excel_file = st.file_uploader("📊 Upload Excel Takeoff (Optional)", type=["xlsx", "xls"])
    if excel_file:
        st.success("✅ Excel loaded — tags will be cross-referenced")

st.sidebar.header("⚙️ Settings")
run_ocr = st.sidebar.checkbox("Run OCR (Scanned PDFs)", value=True)
use_ml = st.sidebar.checkbox("🧠 Use AI (NER + DLA)", value=ML_AVAILABLE)

if use_ml and ML_AVAILABLE:
    diag_only = st.sidebar.checkbox("📐 Highlight Diagram Pages Only", value=True, help="Only apply highlights to mechanical drawing pages.")
else:
    diag_only = False
    if use_ml and not ML_AVAILABLE:
        st.sidebar.error(f"ML Error: {ML_ERROR}")
        if st.sidebar.button("Show Full Error"):
            st.sidebar.code(ML_TRACEBACK)

show_debug = st.sidebar.checkbox("🪲 Show Debug Info", value=False)

# --- Excel Preview ---
excel_data = None
if excel_file:
    excel_data = load_tags_from_excel(excel_file.read())
    excel_file.seek(0)  # Reset for potential re-read
    if "error" in excel_data:
        st.error(f"Excel Error: {excel_data['error']}")
    else:
        with st.expander(f"📋 Excel Tags: {excel_data['total_count']} values from {len(excel_data['columns_used'])} columns"):
            st.write(f"**Columns used:** {excel_data['columns_used']}")
            st.write(f"**Columns skipped (last 2):** {excel_data['columns_skipped']}")
            st.write(f"**Sample values:** {excel_data['all_values'][:20]}")

if uploaded_file and st.button("🚀 Process PDF", type="primary", use_container_width=True):
    pdf_bytes = uploaded_file.read()
    _tag_color_map.clear()  # Reset colors each run

    try:
        with st.status("🔍 Analyzing Document...", expanded=True) as status:
            st.write("📄 Extracting text and detecting layout...")
            page_texts = []
            page_ocr_words = {}  # page_index -> list of OCR word dicts

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    page_texts.append(p.extract_text() or "")

            # --- Per-page OCR fallback ---
            # Drawing pages typically have < 1500 chars of digital text
            # (just border/title block), while text-heavy pages have 2500+.
            # We OCR pages with little digital text to extract drawing content.
            if run_ocr:
                ocr_count = 0

                doc_check = fitz.open(stream=pdf_bytes, filetype="pdf")

                for i, page in enumerate(doc_check):

                    page_text = page_texts[i] if i < len(page_texts) else ""
                    text_len = len(page_text.strip())

                    # Detect embedded images
                    images = page.get_images(full=True)

                    # Smart OCR trigger
                    should_ocr = (
                        len(images) > 0     # Drawing pages almost always have raster content
                        or text_len < 3000  # Still catch low-text drawing pages
                    )

                    if should_ocr:

                        st.write(f"🖼️ Page {i+1}: image/text heuristic triggered — running OCR...")

                        ocr_result = ocr_page_with_tesseract(
                            pdf_bytes,
                            i,
                            target_dpi=100   # 👈 IMPORTANT improvement
                        )

                        # Keep best text source
                        if len(ocr_result["text"].strip()) > text_len:
                            page_texts[i] = ocr_result["text"]

                        page_ocr_words[i] = ocr_result["words"]
                        ocr_count += 1
                doc_check.close()

                if ocr_count:
                    st.write(f"✅ OCR completed on {ocr_count} page(s).")

            full_text = "\n".join(page_texts)

            if show_debug:
                st.text_area("📝 Extracted Text Preview", full_text[:5000] + "...", height=200)

            st.write("🏷️ Identifying equipment marks...")
            marks_set, stats = hybrid_mark_extraction(full_text, use_ml=use_ml)
            marks_list = sorted(list(marks_set))

            st.write("📏 Extracting measurements...")
            all_meas = []
            for i, p_text in enumerate(page_texts):
                p_meas = re.findall(MEAS_PATTERN, p_text, re.I)
                for m in p_meas:
                    all_meas.append({"raw": m, "page": i + 1})
            measurements_df = pd.DataFrame(all_meas)

            # --- Excel cross-reference ---
            excel_matches = []
            missing_tags = []
            if excel_data and "error" not in excel_data:
                st.write("📊 Cross-referencing Excel tags...")
                found = set()
                for i, p_text in enumerate(page_texts):
                    p_upper = p_text.upper()
                    for ev in excel_data["all_values"]:
                        if ev.upper() in p_upper:
                            found.add(ev.upper())
                            excel_matches.append({"value": ev, "page": i + 1})
                missing_tags = [v for v in excel_data["all_values"] if v.upper() not in found]
                # Dedup
                seen_em = set()
                unique_em = []
                for em in excel_matches:
                    key = (em["value"], em["page"])
                    if key not in seen_em:
                        seen_em.add(key)
                        unique_em.append(em)
                excel_matches = unique_em

            st.write("🎨 Highlighting with unique colors per tag...")

            # --- ALWAYS use per-tag unique color highlighting ---
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_index, page in enumerate(doc):
                p_num = page_index + 1

                # If diag_only, skip non-diagram pages (but never skip OCR'd pages —
                # they're drawing pages that were specifically OCR'd by user request)
                if diag_only and page_index not in page_ocr_words:
                    page_text = page_texts[page_index] if page_index < len(page_texts) else ""
                    upper = page_text.upper()
                    has_mech = any(kw in upper for kw in [
                        'MECHANICAL', 'HVAC',
                    ]) or bool(re.search(r'\bM-?\d+', upper))
                    has_plan = any(kw in upper for kw in [
                        'FLOOR PLAN', 'PLAN', 'LAYOUT', 'ROOF PLAN',
                        'REFLECTED CEILING', 'CEILING PLAN',
                        '1ST FLOOR', '2ND FLOOR', '3RD FLOOR',
                        'FIRST FLOOR', 'SECOND FLOOR', 'THIRD FLOOR',
                        'GROUND FLOOR', 'BASEMENT',
                    ])
                    is_reject = any(kw in upper for kw in [
                        'SCHEDULE', 'SPECIFICATIONS', 'GENERAL NOTES',
                        'TABLE OF CONTENTS', 'COVER SHEET',
                    ])
                    if is_reject or not (has_mech and has_plan):
                        continue

                is_ocr_page = page_index in page_ocr_words
                ocr_words = page_ocr_words.get(page_index, [])

                # ---- Helper: highlight via digital search OR OCR coords ----
                def highlight_tag_on_page(tag_text, color):
                    """Try digital search first; fall back to OCR word boxes."""
                    found = False
                    variants = [tag_text, tag_text.replace("-", " "), tag_text.replace("-", "")]
                    for v in variants:
                        rects = page.search_for(v)
                        if rects:
                            for r in rects:
                                annot = page.add_highlight_annot(r)
                                annot.set_colors(stroke=color)
                                annot.update()
                            found = True
                            break
                    # OCR coordinate fallback
                    if not found and is_ocr_page:
                        tag_upper = tag_text.upper()
                        # Try to find multi-word tags by combining consecutive OCR words
                        tag_parts = tag_upper.split()
                        if len(tag_parts) <= 1:
                            # Single-word tag: match individual OCR words
                            for ow in ocr_words:
                                if tag_upper == ow["text"].upper() or tag_upper in ow["text"].upper():
                                    annot = page.add_highlight_annot(ow["bbox"])
                                    annot.set_colors(stroke=color)
                                    annot.update()
                                    found = True
                        else:
                            # Multi-word: slide a window over OCR words
                            n = len(tag_parts)
                            for wi in range(len(ocr_words) - n + 1):
                                window_text = " ".join(ocr_words[wi + k]["text"].upper() for k in range(n))
                                if tag_upper == window_text or tag_upper in window_text:
                                    combined_rect = ocr_words[wi]["bbox"]
                                    for k in range(1, n):
                                        combined_rect = combined_rect | ocr_words[wi + k]["bbox"]
                                    annot = page.add_highlight_annot(combined_rect)
                                    annot.set_colors(stroke=color)
                                    annot.update()
                                    found = True
                    return found

                # Marks — each gets unique color
                for m in marks_list:
                    color = get_color_for_tag(m)
                    highlight_tag_on_page(m, color)

                # Excel matches on this page — each gets unique color
                if excel_matches:
                    page_em = [em for em in excel_matches if em["page"] == p_num]
                    for em in page_em:
                        color = get_color_for_tag(em["value"])
                        highlight_tag_on_page(em["value"], color)

                # Measurements — orange
                if not measurements_df.empty:
                    pm = measurements_df[measurements_df['page'] == p_num]
                    for _, row in pm.iterrows():
                        highlight_tag_on_page(row['raw'], MEASUREMENT_COLOR)

            buf = io.BytesIO()
            doc.save(buf)
            buf.seek(0)
            highlighted_pdf = buf.getvalue()
            doc.close()

            status.update(label="✅ Analysis Complete!", state="complete")

        # Outputs
        st.balloons()
        marks_df = pd.DataFrame([{"Mark": m, "Color": f"Color #{i+1}"} for i, m in enumerate(marks_list)])
        excel_matches_df = pd.DataFrame(excel_matches) if excel_matches else pd.DataFrame()

        t1, t2, t3, t4 = st.tabs(["📊 Data Preview", "📋 Excel Report", "📄 Highlighted PDF", "⬇️ Export"])

        with t1:
            st.subheader("🏷️ Equipment Marks (Unique Color Each)")
            if not marks_df.empty:
                st.dataframe(marks_df, use_container_width=True, hide_index=True)
            st.subheader("📏 Measurements")
            if not measurements_df.empty:
                st.dataframe(measurements_df, use_container_width=True, hide_index=True)

        with t2:
            if not excel_data or "error" in excel_data:
                st.info("Upload an Excel file to see cross-reference results.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Excel Values", excel_data["total_count"])
                c2.metric("Found in PDF ✅", len(excel_matches))
                c3.metric("Not Found ❌", len(missing_tags))

                if not excel_matches_df.empty:
                    st.subheader("✅ Found in PDF")
                    st.dataframe(excel_matches_df, use_container_width=True, hide_index=True)
                if missing_tags:
                    st.subheader("❌ Not Found in PDF")
                    st.dataframe(pd.DataFrame({"Value": missing_tags}), use_container_width=True, hide_index=True)

        with t3:
            st.success("Highlighted PDF Generated Successfully.")

        with t4:
            zip_data, excel_export, json_data = create_export_package(
                pdf_bytes, highlighted_pdf, marks_df, measurements_df, {},
                uploaded_file.name, excel_matches_df, missing_tags
            )
            col1, col2 = st.columns(2)
            col1.download_button("📂 Download All (ZIP)", zip_data, f"analysis_{uploaded_file.name}.zip", "application/zip")
            col2.download_button("📄 Highlighted PDF", highlighted_pdf, f"highlighted_{uploaded_file.name}", "application/pdf")

            col3, col4 = st.columns(2)
            col3.download_button("📗 Excel Summary", excel_export, "summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            col4.download_button("📋 JSON Data", json.dumps(json_data, indent=2), "data.json", "application/json")
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.code(traceback.format_exc())
