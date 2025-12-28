import pandas as pd
import sys
import os

def split_csv(input_file, p1, p2):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load the original data
    # If your CSV doesn't have headers, we assign them;
    # if it does, pandas will use them automatically.
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- TRANSFORMATION LOGIC ---
    # Group by start and end dates to find tickers belonging to the same window
    grouped = df.groupby(['start', 'end'])['ticker'].apply(list).reset_index()

    processed_data = []
    for _, row in grouped.iterrows():
        tickers = row['ticker']
        if tickers:
            primary = tickers[0]
            others = ";".join(tickers[1:])
            processed_data.append({
                'ticker': primary,
                'start': row['start'],
                'end': row['end'],
                'others': others
            })

    new_df = pd.DataFrame(processed_data)
    # ----------------------------

    total_rows = len(new_df)
    split_idx = int(total_rows * (p1 / 100))

    # Split the processed dataframe
    df_1 = new_df.iloc[:split_idx]
    df_2 = new_df.iloc[split_idx:]

    output_1 = f'split_{p1}.csv'
    output_2 = f'split_{p2}.csv'

    # Save to CSV with the requested headers: ticker,start,end,others
    df_1.to_csv(output_1, index=False)
    df_2.to_csv(output_2, index=False)

    print(f"Successfully processed and split {total_rows} grouped time-windows:")
    print(f" - {output_1}: {len(df_1)} rows ({p1}%)")
    print(f" - {output_2}: {len(df_2)} rows ({p2}%)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: py split-nvda-data.py <percent1> <percent2> <input>")
        print("Example: py split-nvda-data.py 80 20 fetch.csv")
    else:
        try:
            val1 = int(sys.argv[1])
            val2 = int(sys.argv[2])
            inp = sys.argv[3]
            split_csv(inp, val1, val2)
        except ValueError:
            print("Error: Please provide integer values for percentages.")
