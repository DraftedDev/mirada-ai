import sys
import pandas as pd
from datetime import datetime, timedelta

def generate_csv(start_date_str, end_date_str, output_file="out.csv"):
    tickers = ["NVDA", "AMD", "TSM", "AVGO", "INTC", "QQQ"]

    # Parse input dates
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        sys.exit(1)

    rows = []
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + timedelta(days=62)  # ~2 months
        for ticker in tickers:
            rows.append([ticker, current_start.strftime("%d-%m-%Y"), current_end.strftime("%d-%m-%Y")])
        current_start += timedelta(days=10)

    df = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    df.to_csv(output_file, index=False)
    print(f"CSV file generated: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen-nvda-fetch.py <start_date> <end_date>")
        print("Example: py gen-nvda-fetch.py 2022-01-01 2025-12-31")
        sys.exit(1)

    generate_csv(sys.argv[1], sys.argv[2])
