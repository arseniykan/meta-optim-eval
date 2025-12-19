
import pandas as pd
import os

f = "/home/spl_arseniy/optim4rl/logs/lopt_rl_ant/5/result_Test.feather"
print(f"--- Inspecting {f} ---")
if os.path.exists(f):
    df = pd.read_feather(f)
    print(f"Shape: {df.shape}")
    print(df)
else:
    print("File not found.")
