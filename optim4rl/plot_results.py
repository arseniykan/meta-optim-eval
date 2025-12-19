
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_feather(file_paths):
    plt.figure(figsize=(10, 6))
    
    # Use distinct colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            continue

        try:
            # Load the feather file
            df = pd.read_feather(file_path)
            
            # Create a nice label
            # logs/ppo_humanoid/2/result_Test.feather -> ppo_humanoid/2
            parts = file_path.split('/')
            if len(parts) >= 3:
                label = f"{parts[-3]}/{parts[-2]}"
            else:
                label = file_path
            
            # Check standard columns
            if 'Step' in df.columns and 'Return' in df.columns:
                color = colors[i % len(colors)]
                plt.plot(df['Step'], df['Return'], label=label, color=color, alpha=0.9, linewidth=2)
            else:
                print(f"Could not find 'Step' and 'Return' columns to plot in {file_path}")
                
        except Exception as e:
            print(f"Error reading feather file {file_path}: {e}")

    plt.xlabel('Training Steps')
    plt.ylabel('Episode Return')
    plt.title('Performance Comparison: Humanoid')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = "comparison_plot.png"
    plt.savefig(output_file)
    print(f"Comparison plot saved to: {output_file}")
    #plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python plot_results.py <file1.feather> [file2.feather] ...")
    else:
        plot_feather(sys.argv[1:])
