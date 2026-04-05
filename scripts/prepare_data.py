"""Convert company_1000_data.xlsx → data/companies.csv with standardised column names."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "company_1000_data.xlsx"
CSV_PATH = ROOT / "data" / "companies.csv"


def main() -> None:
    if not XLSX_PATH.exists():
        print(f"ERROR: {XLSX_PATH} not found")
        sys.exit(1)

    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    print(f"Loaded {len(df)} rows from xlsx")
    print(f"Original columns: {list(df.columns)}")

    column_map = {
        "Consolidated ID": "id",
        "Company Name": "company_name",
        "Country": "country",
        "Long Offering": "long_offering",
    }
    df = df.rename(columns=column_map)
    df = df[["id", "company_name", "country", "long_offering"]]

    # Convert id to string (it may be float from xlsx)
    df["id"] = df["id"].astype(int).astype(str)

    # Drop rows with missing long_offering
    before = len(df)
    df = df.dropna(subset=["long_offering"])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing long_offering")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
