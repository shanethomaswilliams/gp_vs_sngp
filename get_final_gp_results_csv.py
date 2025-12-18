import os
import json
import pandas as pd
import re
from pathlib import Path
import argparse


def parse_folder_name(folder_name):
    """
    Parse folder name to extract hyperparameters.
    Format: {Dataset}_{Num_Examples}_{SEED}
    
    Returns dict with: dataset, num_examples, seed
    """
    # Pattern: {string}_{int}_{int}
    pattern = r'([A-Za-z]+)_(\d+)_(\d+)'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'dataset': match.group(1),
            'num_examples': int(match.group(2)),
            'seed': int(match.group(3))
        }
    return None


def collect_gp_results(target_dir):
    """
    Recursively collect all test_results.json files from GP folders.
    
    Args:
        target_dir: Root directory to search
        
    Returns:
        List of dicts containing all results and metadata
    """
    results = []
    
    # Recursively search for test_results.json files
    for root, dirs, files in os.walk(target_dir):
        if 'test_results.json' in files:
            folder_name = os.path.basename(root)
            
            # Try to parse as GP folder format
            metadata = parse_folder_name(folder_name)
            
            if metadata:
                # Load the test results
                json_path = os.path.join(root, 'test_results.json')
                try:
                    with open(json_path, 'r') as f:
                        test_results = json.load(f)
                    
                    # Combine metadata and test results
                    row = {
                        **metadata,
                        'test_log_likelihood': test_results.get('test_log_likelihood'),
                        'test_mse': test_results.get('test_mse'),
                        'test_rmse': test_results.get('test_rmse'),
                        'folder_path': root
                    }
                    results.append(row)
                    
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect GP test results into a CSV")
    parser.add_argument("target_dir", type=str, help="Root directory to search for results")
    parser.add_argument("--output", type=str, default="final_result_gp.csv", 
                        help="Output CSV filename (default: final_result_gp.csv)")
    
    args = parser.parse_args()
    
    print(f"Searching for GP results in: {args.target_dir}")
    
    # Collect all results
    results = collect_gp_results(args.target_dir)
    
    if not results:
        print("No GP results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by dataset, num_examples, seed for readability
    df = df.sort_values(['dataset', 'num_examples', 'seed'])
    
    # Reorder columns for clarity
    column_order = [
        'dataset', 'num_examples', 'seed',
        'test_log_likelihood', 'test_mse', 'test_rmse', 'folder_path'
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\nFound {len(results)} GP results")
    print(f"Results saved to: {args.output}")
    print(f"\nSummary by dataset:")
    print(df.groupby('dataset').size())
    print(f"\nSummary by num_examples:")
    print(df.groupby('num_examples').size())
    print(f"\nPreview:")
    print(df.head(10))


if __name__ == "__main__":
    main()