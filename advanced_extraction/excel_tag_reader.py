"""
Excel Tag Reader - Reads ALL columns except last 2.
Tracks which column each value came from (for color-coded highlighting).
"""

import pandas as pd
import re
from typing import Dict


def load_tags_from_excel(file_content) -> Dict:
    """
    Load ALL values from Excel, skip last 2 columns.
    Each value tracks its source column for color-coded highlighting.
    """
    try:
        xls = pd.ExcelFile(file_content)
    except Exception as e:
        return {"error": f"Failed to open Excel: {str(e)}"}

    all_values = []      # List of {value, column}
    seen_values = set()
    columns_used = []
    columns_skipped = []
    sheet_info = []

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

            sheet_count = 0
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
                    if key not in seen_values:
                        seen_values.add(key)
                        all_values.append({"value": val, "column": col})
                        sheet_count += 1

            sheet_info.append({"sheet": sheet_name, "values": sheet_count})

        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {e}")

    return {
        "all_values": all_values,   # [{value, column}, ...]
        "columns_used": columns_used,
        "columns_skipped": columns_skipped,
        "total_count": len(all_values),
        "sheet_info": sheet_info,
    }
