"""
Advanced Automated Document Extraction App
Supports scanned PDFs via OCR. Each tag = unique color.
Only highlights mechanical floor plan pages.
"""

import streamlit as st
import pandas as pd
import io
import zipfile
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_extraction import AutoExtractor, load_tags_from_excel, highlight_with_tags

st.set_page_config(layout="wide", page_title="Advanced Document Extraction")


def main():
    st.title("🚀 Advanced Document Extraction")
    st.markdown("Supports **scanned PDFs** (OCR) • Each tag = unique color • Mechanical floor plans only")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    with col2:
        excel_file = st.file_uploader("📊 Upload Excel (Optional)", type=["xlsx", "xls"])
        if excel_file:
            st.success("✅ Excel loaded")
        else:
            st.info("ℹ️ PDF-only mode")

    if not pdf_file:
        st.stop()

    excel_data = None
    if excel_file:
        excel_bytes = excel_file.read()
        excel_data = load_tags_from_excel(excel_bytes)
        if "error" in excel_data:
            st.error(f"Excel Error: {excel_data['error']}")
            st.stop()
        with st.expander(f"📋 Excel: {excel_data['total_count']} values from {len(excel_data['columns_used'])} columns"):
            st.write(f"**Columns used:** {excel_data['columns_used']}")
            st.write(f"**Columns skipped:** {excel_data['columns_skipped']}")

    if st.button("🚀 Process PDF", type="primary", use_container_width=True):
        pdf_bytes = pdf_file.read()

        # Progress bar for OCR
        progress_bar = st.progress(0, text="Starting extraction...")
        status_text = st.empty()

        def update_progress(page, total):
            pct = page / total
            progress_bar.progress(pct, text=f"Processing page {page}/{total}...")
            status_text.text(f"Page {page}/{total} — {'OCR running...' if True else 'reading text...'}")

        extractor = AutoExtractor(pdf_bytes, excel_data)
        results = extractor.extract_all(progress_callback=update_progress)

        progress_bar.progress(1.0, text="Highlighting...")
        highlighted_pdf = highlight_with_tags(pdf_bytes, results)

        progress_bar.empty()
        status_text.empty()
        st.balloons()

        # Count OCR pages
        ocr_count = len(results.get("ocr_pages", {}))

        # ===== RESULTS =====
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Page Summary", "🏷️ Marks & Measurements", "📋 Excel Report", "📥 Export"
        ])

        with tab1:
            st.subheader("📊 Page-by-Page Summary")
            summary_df = pd.DataFrame(results["page_summary"])
            st.dataframe(
                summary_df.style.apply(
                    lambda row: ['background-color: #e6f3ff' if row['Type'] == 'Mech Floor Plan'
                                 else 'background-color: #f5f5f5'] * len(row),
                    axis=1
                ),
                use_container_width=True, hide_index=True
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Floor Plans", len(results["diagram_pages"]))
            c2.metric("Skipped", len(results["skipped_pages"]))
            c3.metric("OCR Pages", ocr_count)
            c4.metric("Total Marks", len(results["marks_found"]))
            c5.metric("Measurements", len(results["measurements"]))

        with tab2:
            if results["marks_found"]:
                st.subheader("🏷️ Equipment Marks")
                st.dataframe(pd.DataFrame(results["marks_found"]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No marks found.")

            if results["measurements"]:
                st.subheader("📏 Measurements")
                st.dataframe(pd.DataFrame(results["measurements"]),
                             use_container_width=True, hide_index=True)

        with tab3:
            if not excel_data:
                st.info("Upload an Excel file to see cross-reference results.")
            else:
                matched = len(results["excel_matches"])
                missing = len(results["missing_excel_values"])
                total = excel_data["total_count"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Excel Values", total)
                c2.metric("Found in PDF ✅", matched)
                c3.metric("Not Found ❌", missing)

                if results["excel_matches"]:
                    st.subheader("✅ Found on Floor Plans")
                    st.dataframe(pd.DataFrame(results["excel_matches"]),
                                 use_container_width=True, hide_index=True)

                if results["missing_excel_values"]:
                    st.subheader("❌ Not Found")
                    st.dataframe(
                        pd.DataFrame({"Value": results["missing_excel_values"]}),
                        use_container_width=True, hide_index=True
                    )

        with tab4:
            st.subheader("📥 Download Results")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"highlighted_{pdf_file.name}", highlighted_pdf)
                xl_buf = io.BytesIO()
                # Need to remove non-serializable ocr_pages for Excel
                export_results = {k: v for k, v in results.items() if k != "ocr_pages"}
                with pd.ExcelWriter(xl_buf, engine="openpyxl") as w:
                    pd.DataFrame(export_results["page_summary"]).to_excel(
                        w, sheet_name="Page_Summary", index=False)
                    if export_results["marks_found"]:
                        pd.DataFrame(export_results["marks_found"]).to_excel(
                            w, sheet_name="Marks", index=False)
                    if export_results["measurements"]:
                        pd.DataFrame(export_results["measurements"]).to_excel(
                            w, sheet_name="Measurements", index=False)
                    if excel_data and export_results["excel_matches"]:
                        pd.DataFrame(export_results["excel_matches"]).to_excel(
                            w, sheet_name="Excel_Found", index=False)
                    if excel_data and export_results["missing_excel_values"]:
                        pd.DataFrame({"Value": export_results["missing_excel_values"]}).to_excel(
                            w, sheet_name="Not_Found", index=False)
                zf.writestr("extraction_report.xlsx", xl_buf.getvalue())
                zf.writestr("raw_data.json", json.dumps(export_results, indent=2, default=str))

            c1, c2 = st.columns(2)
            c1.download_button("📦 Download ZIP", zip_buf.getvalue(),
                               f"analysis_{pdf_file.name}.zip", "application/zip")
            c2.download_button("📄 Highlighted PDF", highlighted_pdf,
                               f"highlighted_{pdf_file.name}", "application/pdf")


if __name__ == "__main__":
    main()
