import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot SNGP results from CSV')
parser.add_argument('csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--output_dir', type=str, default='./plots', 
                    help='Directory to save plots (default: ./plots)')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV data
df = pd.read_csv(args.csv_path)

# Calculate rank as percentage of dataset size
df['rank_percentage'] = (df['rank'] / df['num_examples']) * 100

# Get unique datasets and their sizes
datasets = df['dataset'].dropna().unique()

# Create plots for each dataset
for dataset_name in datasets:
    dataset_df = df[df['dataset'] == dataset_name]
    dataset_sizes = sorted(dataset_df['num_examples'].unique())
    
    # Determine number of subplots needed
    n_sizes = len(dataset_sizes)
    n_cols = min(3, n_sizes)  # Max 3 columns
    n_rows = int(np.ceil(n_sizes / n_cols))
    
    # Create figure with subplots - one row for log likelihood, one for MSE
    fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 10))
    if n_sizes == 1:
        axes = axes.reshape(2, 1)
    elif n_cols == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'{dataset_name} Dataset - SNGP Performance vs Rank Percentage', 
                 fontsize=16, fontweight='bold')
    
    for idx, size in enumerate(dataset_sizes):
        col_idx = idx % n_cols
        size_df = dataset_df[dataset_df['num_examples'] == size].sort_values('rank_percentage')
        
        # Plot test log likelihood
        axes[0, col_idx].plot(size_df['rank_percentage'], 
                              size_df['test_log_likelihood'], 
                              'o-', linewidth=2, markersize=8)
        axes[0, col_idx].set_xlabel('Rank (% of Dataset Size)', fontsize=11)
        axes[0, col_idx].set_ylabel('Test Log Likelihood', fontsize=11)
        axes[0, col_idx].set_title(f'N = {int(size):,}', fontsize=12, fontweight='bold')
        axes[0, col_idx].grid(True, alpha=0.3)
        
        # Plot test MSE
        axes[1, col_idx].plot(size_df['rank_percentage'], 
                              size_df['test_mse'], 
                              'o-', color='orangered', linewidth=2, markersize=8)
        axes[1, col_idx].set_xlabel('Rank (% of Dataset Size)', fontsize=11)
        axes[1, col_idx].set_ylabel('Test MSE', fontsize=11)
        axes[1, col_idx].grid(True, alpha=0.3)
    
    # Hide any extra subplots
    for idx in range(n_sizes, n_rows * n_cols):
        col_idx = idx % n_cols
        axes[0, col_idx].set_visible(False)
        axes[1, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'sngp_{dataset_name.lower()}_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot for {dataset_name} to {output_path}")
    plt.close()

# Also create a combined comparison plot showing all dataset sizes together
for dataset_name in datasets:
    dataset_df = df[df['dataset'] == dataset_name]
    dataset_sizes = sorted(dataset_df['num_examples'].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{dataset_name} Dataset - All Sizes Comparison', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dataset_sizes)))
    
    for idx, size in enumerate(dataset_sizes):
        size_df = dataset_df[dataset_df['num_examples'] == size].sort_values('rank_percentage')
        
        # Plot test log likelihood
        axes[0].plot(size_df['rank_percentage'], 
                     size_df['test_log_likelihood'], 
                     'o-', color=colors[idx], linewidth=2, 
                     markersize=6, label=f'N = {int(size):,}')
        
        # Plot test MSE
        axes[1].plot(size_df['rank_percentage'], 
                     size_df['test_mse'], 
                     'o-', color=colors[idx], linewidth=2, 
                     markersize=6, label=f'N = {int(size):,}')
    
    axes[0].set_xlabel('Rank (% of Dataset Size)', fontsize=12)
    axes[0].set_ylabel('Test Log Likelihood', fontsize=12)
    axes[0].set_title('Test Log Likelihood vs Rank %', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Rank (% of Dataset Size)', fontsize=12)
    axes[1].set_ylabel('Test MSE', fontsize=12)
    axes[1].set_title('Test MSE vs Rank %', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'sngp_{dataset_name.lower()}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot for {dataset_name} to {output_path}")
    plt.close()

print("\nAll plots generated successfully!")