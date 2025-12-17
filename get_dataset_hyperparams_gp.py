import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def parse_directory_name(dirname):
    """
    Parse directory name in format {DatasetName}_{N}_{Seed}
    Returns (dataset_name, N, seed) or None if doesn't match
    """
    parts = dirname.split('_')
    if len(parts) < 3:
        return None
    
    try:
        # Last part is seed, second to last is N
        seed = int(parts[-1])
        N = int(parts[-2])
        # Everything else is dataset name
        dataset_name = '_'.join(parts[:-2])
        return (dataset_name, N, seed)
    except ValueError:
        return None


def load_hyperparams(filepath):
    """Load hyperparameters from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return {
            'lengthscale': data.get('lengthscale'),
            'outputscale': data.get('outputscale'),
            'noise': data.get('noise')
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {filepath}: {e}")
        return None


def compute_statistics(values):
    """Compute mean, median, and std for a list of values"""
    if not values:
        return {'mean': None, 'median': None, 'std': None}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr))
    }


def analyze_hyperparameters(base_path, target_N=1500):
    """
    Analyze hyperparameters from GP experiments
    
    Args:
        base_path: Path to the directory containing experiment folders
        target_N: Number of examples to filter for (default: 1500)
    
    Returns:
        Dictionary with statistics for each dataset
    """
    base_path = Path(base_path)
    
    # Store hyperparameters grouped by dataset
    dataset_hyperparams = defaultdict(lambda: {
        'lengthscale': [],
        'outputscale': [],
        'noise': []
    })
    
    # Scan all directories
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        # Parse directory name
        parsed = parse_directory_name(item.name)
        if parsed is None:
            continue
        
        dataset_name, N, seed = parsed
        
        # Filter for target N
        if N != target_N:
            continue
        
        # Load hyperparameters
        hyperparams_path = item / 'hyperparams.json'
        hyperparams = load_hyperparams(hyperparams_path)
        
        if hyperparams is None:
            continue
        
        # Store values
        for key in ['lengthscale', 'outputscale', 'noise']:
            if hyperparams[key] is not None:
                dataset_hyperparams[dataset_name][key].append(hyperparams[key])
    
    # Compute statistics for each dataset
    results = {}
    for dataset_name, hyperparams in dataset_hyperparams.items():
        results[dataset_name] = {
            'lengthscale': compute_statistics(hyperparams['lengthscale']),
            'outputscale': compute_statistics(hyperparams['outputscale']),
            'noise': compute_statistics(hyperparams['noise']),
            'num_seeds': len(hyperparams['lengthscale'])
        }
    
    return results


def print_results(results):
    """Pretty print the results"""
    print("\n" + "="*80)
    print("HYPERPARAMETER STATISTICS ACROSS SEEDS")
    print("="*80)
    
    for dataset_name, stats in sorted(results.items()):
        print(f"\n{dataset_name.upper()}")
        print(f"  Number of seeds: {stats['num_seeds']}")
        print("-" * 80)
        
        for param in ['lengthscale', 'outputscale', 'noise']:
            param_stats = stats[param]
            print(f"  {param.capitalize()}:")
            print(f"    Mean:   {param_stats['mean']:.6f}")
            print(f"    Median: {param_stats['median']:.6f}")
            print(f"    Std:    {param_stats['std']:.6f}")
        print()


def save_results_json(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_path}")


def save_results_csv(results, output_path):
    """Save results to CSV file"""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Dataset', 'Num_Seeds',
            'Lengthscale_Mean', 'Lengthscale_Median', 'Lengthscale_Std',
            'Outputscale_Mean', 'Outputscale_Median', 'Outputscale_Std',
            'Noise_Mean', 'Noise_Median', 'Noise_Std'
        ])
        
        # Data rows
        for dataset_name, stats in sorted(results.items()):
            writer.writerow([
                dataset_name,
                stats['num_seeds'],
                stats['lengthscale']['mean'],
                stats['lengthscale']['median'],
                stats['lengthscale']['std'],
                stats['outputscale']['mean'],
                stats['outputscale']['median'],
                stats['outputscale']['std'],
                stats['noise']['mean'],
                stats['noise']['median'],
                stats['noise']['std']
            ])
    
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze GP hyperparameters across seeds')
    parser.add_argument('path', type=str, help='Path to directory containing experiment folders')
    parser.add_argument('--N', type=int, default=1500, help='Number of examples to filter for (default: 1500)')
    parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    parser.add_argument('--output-csv', type=str, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Analyze hyperparameters
    results = analyze_hyperparameters(args.path, target_N=args.N)
    
    if not results:
        print(f"No results found for N={args.N} in {args.path}")
    else:
        # Print results
        print_results(results)
        
        # Save to files if requested
        if args.output_json:
            save_results_json(results, args.output_json)
        
        if args.output_csv:
            save_results_csv(results, args.output_csv)