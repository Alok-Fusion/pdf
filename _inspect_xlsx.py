import pandas as pd
import sys

files = [
    r"Takeoff_SUNO Nursing and Health Building.xlsx",
    r"Takeoff-Health Partners Lakeview -19oct205.xlsx"
]

for f in files:
    print(f"\n{'='*80}")
    print(f"FILE: {f}")
    print('='*80)
    try:
        xl = pd.ExcelFile(f)
        print(f"Sheet names: {xl.sheet_names}")
        for sname in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sname)
            print(f"\n--- Sheet: {sname} ---")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            print(df.head(20).to_string())
            print()
    except Exception as e:
        print(f"Error: {e}")

sys.stdout.flush()
