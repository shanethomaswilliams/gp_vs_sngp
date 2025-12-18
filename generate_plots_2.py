import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot RFF + Laplace and GP results comparison from CSV files')
parser.add_argument('sngp_csv_path', type=str, help='Path to the RFF + Laplace CSV file')
parser.add_argument('gp_csv_path', type=str, help='Path to the GP CSV file')
parser.add_argument('--output_dir', type=str, default='./plots', 
                    help='Directory to save plots (default: ./plots)')
parser.add_argument('--ll_zoom_min', type=float, default=0,
                    help='Minimum y-axis value for zoomed log likelihood plots (default: -5.0)')
parser.add_argument('--ll_zoom_max', type=float, default=0.9,
                    help='Maximum y-axis value for zoomed log likelihood plots (default: 1.0)')
parser.add_argument('--mse_zoom_min', type=float, default=0.0,
                    help='Minimum y-axis value for zoomed MSE plots (auto if not specified)')
parser.add_argument('--mse_zoom_max', type=float, default=0.2,
                    help='Maximum y-axis value for zoomed MSE plots (auto if not specified)')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV data
df_sngp = pd.read_csv(args.sngp_csv_path)
df_gp = pd.read_csv(args.gp_csv_path)

# Add method label to each dataframe
df_sngp['method'] = 'RFF + Laplace'
df_gp['method'] = 'GP'

# Combine dataframes
df_combined = pd.concat([df_sngp, df_gp], ignore_index=True)

# Calculate rank as percentage of dataset size (only for RFF + Laplace which has rank)
df_combined['rank_percentage'] = (df_combined['rank'] / df_combined['num_examples']) * 100

# Get unique datasets
datasets = df_combined['dataset'].dropna().unique()

# Create combined comparison plots for each dataset
for dataset_name in datasets:
    dataset_df = df_combined[df_combined['dataset'] == dataset_name]
    
    # Get unique sizes that exist in both methods
    sngp_sizes = set(dataset_df[dataset_df['method'] == 'RFF + Laplace']['num_examples'].unique())
    gp_sizes = set(dataset_df[dataset_df['method'] == 'GP']['num_examples'].unique())
    common_sizes = sorted(sngp_sizes.intersection(gp_sizes))
    
    if len(common_sizes) == 0:
        print(f"Warning: No common dataset sizes found for {dataset_name}, skipping...")
        continue
    
    # Generate color gradients
    # Red gradient for GP (darker to lighter red)
    red_colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(common_sizes)))
    # Blue gradient for RFF + Laplace (darker to lighter blue)
    blue_colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(common_sizes)))
    
    # Create separate figures for log likelihood and MSE
    for metric, metric_name, y_label in [('test_log_likelihood', 'Log_Likelihood', 'Test Log Likelihood'),
                                          ('test_mse', 'MSE', 'Test MSE')]:
        # Regular plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if dataset_name == 'CrazySin':
            fig.suptitle(f'High Variance Sine Dataset - GP vs RFF + Laplace {metric_name}', 
                     fontsize=16, fontweight='bold')
        elif dataset_name == 'Sin':
            fig.suptitle(f'Sine Dataset - GP vs RFF + Laplace {metric_name}', 
                     fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'{dataset_name} Dataset - GP vs RFF + Laplace {metric_name}', 
                     fontsize=16, fontweight='bold')
        
        # Store handles for legend grouping
        rff_handles = []
        gp_handles = []
        
        for idx, size in enumerate(common_sizes):
            # Get RFF + Laplace data for this size
            sngp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                       (dataset_df['method'] == 'RFF + Laplace')].sort_values('rank_percentage')
            
            # Get GP data for this size (single point)
            gp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                     (dataset_df['method'] == 'GP')]
            
            if len(sngp_size_df) > 0:
                # Plot RFF + Laplace (solid line)
                line, = ax.plot(sngp_size_df['rank_percentage'], 
                            sngp_size_df[metric], 
                            'o-', color=blue_colors[idx], linewidth=2, 
                            markersize=6, label=f'RFF + Laplace N={int(size):,}')
                rff_handles.append(line)
            
            if len(gp_size_df) > 0:
                gp_val = gp_size_df[metric].values[0]
                
                if len(sngp_size_df) > 0:
                    x_range = [sngp_size_df['rank_percentage'].min(), 
                              sngp_size_df['rank_percentage'].max()]
                    
                    # Plot GP as horizontal dotted line
                    line, = ax.plot(x_range, [gp_val, gp_val], 
                                '--', color=red_colors[idx], linewidth=2.5, 
                                label=f'GP N={int(size):,}')
                    ax.plot([x_range[0]], [gp_val], 'D', color=red_colors[idx], 
                                markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)
                    gp_handles.append(line)
        
        ax.set_xlabel('Rank (% of Dataset Size)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Create grouped legend: all RFF + Laplace, then all GP
        all_handles = rff_handles + gp_handles
        ax.legend(handles=all_handles, loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f'gp_vs_rff_{dataset_name.lower()}_{metric_name.lower()}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} comparison plot for {dataset_name} to {output_path}")
        plt.close()
        
        # Create ZOOMED version
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if dataset_name == 'CrazySin':
            fig.suptitle(f'High Variance Sine Dataset - GP vs RFF + Laplace {metric_name}  (ZOOMED)', 
                     fontsize=16, fontweight='bold')
        elif dataset_name == 'Sin':
            fig.suptitle(f'Sine Dataset - GP vs RFF + Laplace {metric_name}  (ZOOMED)', 
                     fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'{dataset_name} Dataset - GP vs RFF + Laplace {metric_name} (ZOOMED)', 
                     fontsize=16, fontweight='bold')
        
        # Store handles for legend grouping
        rff_handles = []
        gp_handles = []
        
        for idx, size in enumerate(common_sizes):
            # Get RFF + Laplace data for this size
            sngp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                       (dataset_df['method'] == 'RFF + Laplace')].sort_values('rank_percentage')
            
            # Get GP data for this size
            gp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                     (dataset_df['method'] == 'GP')]
            
            if len(sngp_size_df) > 0:
                line, = ax.plot(sngp_size_df['rank_percentage'], 
                            sngp_size_df[metric], 
                            'o-', color=blue_colors[idx], linewidth=2, 
                            markersize=6, label=f'RFF + Laplace N={int(size):,}')
                rff_handles.append(line)
            
            if len(gp_size_df) > 0:
                gp_val = gp_size_df[metric].values[0]
                
                if len(sngp_size_df) > 0:
                    x_range = [sngp_size_df['rank_percentage'].min(), 
                              sngp_size_df['rank_percentage'].max()]
                    
                    line, = ax.plot(x_range, [gp_val, gp_val], 
                                '--', color=red_colors[idx], linewidth=2.5, 
                                label=f'GP N={int(size):,}')
                    ax.plot([x_range[0]], [gp_val], 'D', color=red_colors[idx], 
                                markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)
                    gp_handles.append(line)
        
        ax.set_xlabel('Rank (% of Dataset Size)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Set zoom limits
        if metric == 'test_log_likelihood':
            ax.set_ylim(args.ll_zoom_min, args.ll_zoom_max)
        else:  # MSE
            if args.mse_zoom_min is not None and args.mse_zoom_max is not None:
                ax.set_ylim(args.mse_zoom_min, args.mse_zoom_max)
            else:
                # Auto-zoom: use data range with 10% padding
                all_vals = dataset_df[metric].dropna()
                if len(all_vals) > 0:
                    val_min, val_max = all_vals.min(), all_vals.max()
                    val_range = val_max - val_min
                    ax.set_ylim(val_min - 0.1*val_range, val_max + 0.1*val_range)
        
        # Create grouped legend
        all_handles = rff_handles + gp_handles
        ax.legend(handles=all_handles, loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the zoomed figure
        output_path = output_dir / f'gp_vs_rff_{dataset_name.lower()}_{metric_name.lower()}_comparison_ZOOMED.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ZOOMED {metric_name} comparison plot for {dataset_name} to {output_path}")
        plt.close()

# Create individual subplots for each dataset size
for dataset_name in datasets:
    dataset_df = df_combined[df_combined['dataset'] == dataset_name]
    
    # Get unique sizes that exist in both methods
    sngp_sizes = set(dataset_df[dataset_df['method'] == 'RFF + Laplace']['num_examples'].unique())
    gp_sizes = set(dataset_df[dataset_df['method'] == 'GP']['num_examples'].unique())
    common_sizes = sorted(sngp_sizes.intersection(gp_sizes))
    
    if len(common_sizes) == 0:
        continue
    
    # Determine number of subplots needed
    n_sizes = len(common_sizes)
    n_cols = min(3, n_sizes)  # Max 3 columns
    n_rows = int(np.ceil(n_sizes / n_cols))
    
    # Create separate figures for log likelihood and MSE
    for metric, metric_name, y_label in [('test_log_likelihood', 'Log_Likelihood', 'Test Log Likelihood'),
                                          ('test_mse', 'MSE', 'Test MSE')]:
        # Regular plot
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
        if n_sizes == 1:
            axes = np.array([axes])
        
        fig.suptitle(f'{dataset_name} Dataset - GP vs RFF + Laplace {metric_name} by Size', 
                     fontsize=16, fontweight='bold')
        
        for idx, size in enumerate(common_sizes):
            col_idx = idx % n_cols
            
            # Get RFF + Laplace data
            sngp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                       (dataset_df['method'] == 'RFF + Laplace')].sort_values('rank_percentage')
            
            # Get GP data
            gp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                     (dataset_df['method'] == 'GP')]
            
            # Plot RFF + Laplace
            if len(sngp_size_df) > 0:
                axes[col_idx].plot(sngp_size_df['rank_percentage'], 
                                      sngp_size_df[metric], 
                                      'o-', color='steelblue', linewidth=2, markersize=8,
                                      label='RFF + Laplace')
                
                # Plot GP as horizontal line
                if len(gp_size_df) > 0:
                    gp_val = gp_size_df[metric].values[0]
                    x_range = [sngp_size_df['rank_percentage'].min(), 
                              sngp_size_df['rank_percentage'].max()]
                    axes[col_idx].plot(x_range, [gp_val, gp_val], 
                                         '--', color='firebrick', linewidth=2.5, label='GP')
                    axes[col_idx].plot([x_range[0]], [gp_val], 'D', color='firebrick', 
                                         markersize=10, markeredgecolor='darkred', markeredgewidth=1.5)
            
            axes[col_idx].set_xlabel('Rank (% of Dataset Size)', fontsize=11)
            axes[col_idx].set_ylabel(y_label, fontsize=11)
            axes[col_idx].set_title(f'N = {int(size):,}', fontsize=12, fontweight='bold')
            axes[col_idx].legend(loc='best', fontsize=10)
            axes[col_idx].grid(True, alpha=0.3)
        
        # Hide any extra subplots
        for idx in range(n_sizes, n_cols):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f'gp_vs_rff_{dataset_name.lower()}_{metric_name.lower()}_by_size.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} size comparison plot for {dataset_name} to {output_path}")
        plt.close()
        
        # Create ZOOMED version
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
        if n_sizes == 1:
            axes = np.array([axes])
        
        fig.suptitle(f'{dataset_name} Dataset - GP vs RFF + Laplace {metric_name} by Size (ZOOMED)', 
                     fontsize=16, fontweight='bold')
        
        for idx, size in enumerate(common_sizes):
            col_idx = idx % n_cols
            
            # Get RFF + Laplace data
            sngp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                       (dataset_df['method'] == 'RFF + Laplace')].sort_values('rank_percentage')
            
            # Get GP data
            gp_size_df = dataset_df[(dataset_df['num_examples'] == size) & 
                                     (dataset_df['method'] == 'GP')]
            
            # Plot RFF + Laplace
            if len(sngp_size_df) > 0:
                axes[col_idx].plot(sngp_size_df['rank_percentage'], 
                                      sngp_size_df[metric], 
                                      'o-', color='steelblue', linewidth=2, markersize=8,
                                      label='RFF + Laplace')
                
                # Plot GP as horizontal line
                if len(gp_size_df) > 0:
                    gp_val = gp_size_df[metric].values[0]
                    x_range = [sngp_size_df['rank_percentage'].min(), 
                              sngp_size_df['rank_percentage'].max()]
                    axes[col_idx].plot(x_range, [gp_val, gp_val], 
                                         '--', color='firebrick', linewidth=2.5, label='GP')
                    axes[col_idx].plot([x_range[0]], [gp_val], 'D', color='firebrick', 
                                         markersize=10, markeredgecolor='darkred', markeredgewidth=1.5)
            
            axes[col_idx].set_xlabel('Rank (% of Dataset Size)', fontsize=11)
            axes[col_idx].set_ylabel(y_label, fontsize=11)
            axes[col_idx].set_title(f'N = {int(size):,}', fontsize=12, fontweight='bold')
            
            # Set zoom limits
            if metric == 'test_log_likelihood':
                axes[col_idx].set_ylim(args.ll_zoom_min, args.ll_zoom_max)
            else:  # MSE
                if args.mse_zoom_min is not None and args.mse_zoom_max is not None:
                    axes[col_idx].set_ylim(args.mse_zoom_min, args.mse_zoom_max)
                else:
                    # Auto-zoom for this specific size
                    size_vals = dataset_df[dataset_df['num_examples'] == size][metric].dropna()
                    if len(size_vals) > 0:
                        val_min, val_max = size_vals.min(), size_vals.max()
                        val_range = val_max - val_min
                        axes[col_idx].set_ylim(val_min - 0.1*val_range, val_max + 0.1*val_range)
            
            axes[col_idx].legend(loc='best', fontsize=10)
            axes[col_idx].grid(True, alpha=0.3)
        
        # Hide any extra subplots
        for idx in range(n_sizes, n_cols):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save the zoomed figure
        output_path = output_dir / f'gp_vs_rff_{dataset_name.lower()}_{metric_name.lower()}_by_size_ZOOMED.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ZOOMED {metric_name} size comparison plot for {dataset_name} to {output_path}")
        plt.close()

print("\nAll plots generated successfully!")