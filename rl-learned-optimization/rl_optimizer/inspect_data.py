
import pandas as pd
import os

files = [
    "pretrained/ant/returns.csv"
]

for f in files:
    print(f"--- Inspecting {f} ---")
    if not os.path.exists(f):
        print("File not found.")
        continue
        
    try:
        df = pd.read_csv(f)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
        print("-----------------------")
    except Exception as e:
        print(f"Error reading {f}: {e}")
