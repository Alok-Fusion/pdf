"""Quick diagnostic using fitz (fast) to check text per page."""
import fitz

for fname in ["mechanical (2).pdf", "mechanical (3).pdf"]:
    print(f"\n=== {fname} ===")
    try:
        doc = fitz.open(fname)
        print(f"  Total pages: {len(doc)}")
        for i, page in enumerate(doc):
            txt = page.get_text().strip()
            preview = txt[:80].replace("\n", " ") if txt else "(empty)"
            print(f"  Page {i+1}: {len(txt)} chars | {preview}")
        doc.close()
    except Exception as e:
        print(f"  ERROR: {e}")
