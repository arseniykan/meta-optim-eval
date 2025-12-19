
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Styling
sns.set(style="darkgrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6)
})

def smoothed_curve(data, window=20):
    return pd.Series(data).rolling(window, min_periods=1).mean()

def load_data():
    data = {}
    
    # 1. Baselines (Steps x Runs) -> Transpose to (Runs x Steps) for easier processing
    # Step size = 20480
    baseline_step_size = 20480
    
    for opt in ['adam', 'rmsprop']:
        path = f"baselines/ant/{opt}/returns.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # DF is (Steps, Runs) e.g., (2441, 18)
            # We want to aggregate statistics across runs
            
            # X axis
            x = np.arange(len(df)) * baseline_step_size
            
            # Mean and Std
            mean = df.mean(axis=1)
            std = df.std(axis=1) / np.sqrt(df.shape[1]) # SE
            
            data[f"Baseline ({opt.capitalize()})"] = {
                'x': x,
                'mean': smoothed_curve(mean),
                'std': smoothed_curve(std)
            }
            print(f"Loaded {opt}: {df.shape}")
            
    # 2. Pretrained (Runs x Steps)
    # Step size approx 21350 or 20480? Let's assume 20480 to match or check frames.
    # User said 50M steps.
    pretrained_path = "pretrained/ant/returns.csv"
    if os.path.exists(pretrained_path):
        df = pd.read_csv(pretrained_path)
        # Drop unnamed if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        # Shape is (Runs, Steps) e.g., (18, 2342)
        # We need to mean across dim 0
        mean = df.mean(axis=0)
        std = df.std(axis=0) / np.sqrt(df.shape[0])
        
        # X axis
        # Assuming same frequency
        x = np.arange(len(mean)) * baseline_step_size
        
        data["OPEN (Pretrained)"] = {
            'x': x,
            'mean': smoothed_curve(mean),
            'std': smoothed_curve(std)
        }
        print(f"Loaded Pretrained: {df.shape}")


    # 3. Optim4RL (Feather)
    lopt_path = "/home/spl_arseniy/optim4rl/logs/lopt_rl_ant/5/result_Test.feather"
    if os.path.exists(lopt_path):
        df = pd.read_feather(lopt_path)
        # Clip to 50M
        df = df[df['Step'] <= 5e7]
        
        # Interpolate for smoothness
        from scipy.interpolate import PchipInterpolator # Monotonic spline
        
        steps = df['Step'].values
        returns = df['Return'].values
        
        # Create dense x for smooth curve
        x_new = np.linspace(steps.min(), steps.max(), 500)
        interpolator = PchipInterpolator(steps, returns)
        y_new = interpolator(x_new)
        
        data["Optim4RL (Lopt-5)"] = {
            'x': x_new,
            'mean': y_new,
            'std': None, # Single run
            'raw_x': steps, # Keep raw points for scatter plot if desired
            'raw_y': returns
        }
        print(f"Loaded Optim4RL: {len(df)} points (Interpolated to 500)")


    return data

def plot_all(data, output_file="ant_final_comparison.png"):
    plt.figure()
    
    colors = {
        "Baseline (Adam)": "tab:orange",
        "Baseline (Rmsprop)": "tab:green",
        "OPEN (Pretrained)": "tab:blue",
        "Optim4RL (Lopt-5)": "tab:red"
    }

    for name, d in data.items():
        x = d['x']
        y = d['mean']
        
        color = colors.get(name, None)
        line, = plt.plot(x, y, label=name, color=color, linewidth=2)
        
        if d['std'] is not None:
            plt.fill_between(x, y - d['std'], y + d['std'], alpha=0.2, color=line.get_color())


    plt.title("Ant Environment: Optimizer Comparison")
    plt.xlabel("Frames")
    plt.ylabel("Return")
    plt.xlim(0, 1e7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    data = load_data()
    plot_all(data)
