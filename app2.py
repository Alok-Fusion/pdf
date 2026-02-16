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

# ML/DL imports
try:
    from ml_models import (
        enhanced_ocr_pdf, hybrid_mark_extraction, 
        extract_measurements_with_ner, check_ml_availability,
        get_ml_stats, detect_document_layout, is_mechanical_plan,
        apply_conditional_highlighting
    )
    ML_AVAILABLE = True
    ML_ERROR = None
except Exception as e:
    ML_AVAILABLE = False
    ML_ERROR = str(e)
    ML_TRACEBACK = traceback.format_exc()

st.set_page_config(layout="wide", page_title="Mechanical PDF Analyzer Pro")

# --------- REGEX & UTILS ---------
from ml_models import normalize_text
MARK_REGEX = re.compile(r'\b[A-Z]{1,4}-\d+\b', re.IGNORECASE)
MEAS_PATTERN = r'\d+"\s*Ø\s*/\s*\d+|\d+"\s*x\s*\d+"\s*/\s*\d+|\d+"\s*x\s*\d+|\d+"\s*Ø'

def extract_marks(text: str):
    return sorted(list(set(m.upper() for m in MARK_REGEX.findall(normalize_text(text)))))

# --------- CORE PROCESSING ---------

def create_export_package(original_pdf, highlighted_pdf, marks_df, measurements_df, legends, file_name):
    """Create ZIP package with all results."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(f"output/{file_name.rsplit('.',1)[0]}_highlighted.pdf", highlighted_pdf)
        
        # Excel Summary
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Always add a summary sheet to avoid "At least one sheet must be visible" crash
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
        zipf.writestr("data/summary.xlsx", excel_buffer.getvalue())
        
        # JSON Data
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

st.title("🔧 Mechanical PDF Analyzer Pro (v3.0)")
st.markdown("Automated Equipment Mark Detection & Engineering Calculation Engine")

uploaded_file = st.file_uploader("Upload Mechanical PDF", type=["pdf"])

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

if uploaded_file and st.button("🚀 Process PDF", type="primary", use_container_width=True):
    pdf_bytes = uploaded_file.read()
    
    try:
        with st.status("🔍 Analyzing Document...", expanded=True) as status:
            st.write("📄 Extracting text and detecting layout...")
            page_texts = []
            page_ocr_data = None
            
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    page_texts.append(p.extract_text() or "")
            
            full_text_digital = "\n".join(page_texts)
            
            # OCR Fallback
            if len(full_text_digital.strip()) < 50 and run_ocr:
                page_ocr_data = enhanced_ocr_pdf(pdf_bytes)
                page_texts = [p['text'] for p in page_ocr_data]
            
            full_text = "\n".join(page_texts)
            
            if show_debug:
                with st.expander("📝 Extracted Text Preview"):
                    st.text(full_text[:5000] + "...")
                
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
            
            st.write("🎨 Applying conditional highlighting...")
            if diag_only:
                highlighted_pdf = apply_conditional_highlighting(pdf_bytes, marks_set, measurements_df, ocr_results=page_ocr_data)
            else:
                # Optimized fallback highlighting (no diagram restriction)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_index, page in enumerate(doc):
                    # Marks
                    for m in marks_list:
                        for r in page.search_for(m):
                            annot = page.add_highlight_annot(r)
                            annot.set_colors(stroke=(0.7, 0.85, 1.0))
                            annot.update()
                    # Measurements
                    if not measurements_df.empty:
                        pm = measurements_df[measurements_df['page'] == (page_index + 1)]
                        for _, row in pm.iterrows():
                            for r in page.search_for(row['raw']):
                                annot = page.add_highlight_annot(r)
                                annot.set_colors(stroke=(1.0, 0.85, 0.7))
                                annot.update()
                buf = io.BytesIO()
                doc.save(buf)
                buf.seek(0)
                highlighted_pdf = buf.getvalue()
                doc.close()
            
            status.update(label="✅ Analysis Complete!", state="complete")

        # Outputs
        st.balloons()
        t1, t2, t3 = st.tabs(["📊 Data Preview", "📄 Highlighted PDF", "⬇️ Export"])
        
        with t1:
            st.subheader("Equipment Marks")
            marks_df = pd.DataFrame([{"Mark": m} for m in marks_list])
            st.dataframe(marks_df, width="stretch")
            st.subheader("Measurements")
            st.dataframe(measurements_df, width="stretch")
            
        with t2:
            st.success("Highlighted PDF Generated Successfully.")
            
        with t3:
            zip_data, excel_data, json_data = create_export_package(pdf_bytes, highlighted_pdf, marks_df, measurements_df, {}, uploaded_file.name)
            
            col1, col2 = st.columns(2)
            col1.download_button("📂 Download All (ZIP)", zip_data, f"analysis_{uploaded_file.name}.zip", "application/zip")
            col2.download_button("📄 Highlighted PDF", highlighted_pdf, f"highlighted_{uploaded_file.name}", "application/pdf")
            
            col3, col4 = st.columns(2)
            col3.download_button("📗 Excel Summary", excel_data, "summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            col4.download_button("📋 JSON Data", json.dumps(json_data, indent=2), "data.json", "application/json")
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.code(traceback.format_exc())




