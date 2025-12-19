
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from configs import all_configs  # Import configs to get frame scaling

# Project Styling
sns.set()
plt.style.use("seaborn-v0_8-colorblind")

tex_fonts = {
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (6, 4),
}
plt.rcParams.update(tex_fonts)

def smooth(data, span=200):
    """Exponential moving average smoothing."""
    if len(data) < span:
        return data
    return data.ewm(span=span).mean()

def get_frames_per_update(env_name):
    # Normalize env name to match config keys
    key = env_name.lower()
    if key not in all_configs:
        print(f"Warning: {key} not found in configs. Defaulting to 1 frame/update.")
        return 1
    
    cfg = all_configs[key]
    return cfg["NUM_ENVS"] * cfg["NUM_STEPS"]

def plot_comparison(envs, output_dir="comparison_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to search
    optimizer_sources = {
        "OPEN": "pretrained",
        "Adam": "baselines/{env}/adam",
        "RMSProp": "baselines/{env}/rmsprop",
    }
    
    colors = {
        "OPEN": "tab:blue",
        "Adam": "tab:orange",
        "RMSProp": "tab:green"
    }

    for env in envs:
        print(f"Processing {env}...")
        plt.figure()
        plt.title(f"Comparison on {env}")
        
        found_data = False
        frames_per_update = get_frames_per_update(env)
        print(f"  Frames per update for {env}: {frames_per_update}")
        
        for opt_name, path_template in optimizer_sources.items():
            if opt_name == "OPEN":
                # OPEN saves results directly in pretrained/env/returns.csv
                # Check for discrete env path
                path = f"pretrained/{env}/returns.csv"
                if not os.path.exists(path):
                     # Try alternative style
                     path = f"pretrained/{env}_OPEN/returns.csv" 
            else:
                path = path_template.format(env=env) + "/returns.csv"
            
            if not os.path.exists(path):
                print(f"  Skipping {opt_name}: File not found at {path}")
                continue
            
            # Load Data
            try:
                df = pd.read_csv(path)
                
                # Determine data orientation
                # We want [Steps, Runs]
                # Heuristic: If dim0 > dim1 (Longer rows) and dim1 < 100 (Runs), assume [Steps, Runs]
                # OPEN files are often [Runs, Steps] (e.g. 16x2341)
                # Baseline files from my script are [Steps, Runs] (e.g. 2441x18)
                
                if df.shape[0] < df.shape[1] and df.shape[0] < 50: 
                    # Likely Rows=Runs (OPEN style) -> Transpose to [Steps, Runs]
                    data = df.values.T 
                    # Drop unnamed if first col is index-like? Usually csv read handles it unless index=False wasn't used.
                    # OPEN csv often has index column if written by pandas default.
                    if "Unnamed: 0" in df.columns:
                         data = df.drop(columns=["Unnamed: 0"]).values.T
                else:
                    # Likely Rows=Steps (Baseline Style)
                     data = df.values
                     if "Unnamed: 0" in df.columns: # Should check logic
                         data = df.drop(columns=["Unnamed: 0"]).values
                
                found_data = True
                
                # Steps (X-axis)
                steps = np.arange(data.shape[0])
                frames = steps * frames_per_update
                
                # Calculate Stats
                # Mean across runs (axis 1)
                mean = data.mean(axis=1)
                std = data.std(axis=1) / np.sqrt(data.shape[1]) # Standard Error
                
                # Smooth
                mean_smooth = pd.Series(mean).ewm(span=50).mean()
                std_smooth = pd.Series(std).ewm(span=50).mean()
                
                color = colors.get(opt_name, None)
                
                line = plt.plot(frames, mean_smooth, label=opt_name, linewidth=1.5, color=color)
                plt.fill_between(frames, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.2, color=color)
                
            except Exception as e:
                print(f"  Error reading {opt_name}: {e}")

        if found_data:
            plt.xlabel("Frames")
            plt.ylabel("Return")
            plt.xlim(0, 1e7) # Limit to 10M frames
            plt.legend()
            plt.tight_layout()
            
            out_file = f"{output_dir}/{env}_comparison.png"
            plt.savefig(out_file, dpi=300)
            print(f"Saved plot to {out_file}")
            plt.close()
        else:
            print(f"No data found for {env}, skipping plot.")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=True, help="List of environments to plot")
    parser.add_argument("--out", type=str, default="comparison_plots")
    args = parser.parse_args()
    
    plot_comparison(args.envs, args.out)
