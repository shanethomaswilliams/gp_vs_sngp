import os
import json
import pandas as pd
import re
from pathlib import Path
import argparse


def parse_folder_name(folder_name):
    """
    Parse folder name to extract hyperparameters.
    Format: SNGP_R{RANK}_LS{lengthscale}_OS{outputscale}_{NUM_EXAMPLES}{DATASET}
    
    Returns dict with: rank, lengthscale, outputscale, num_examples, dataset
    """
    # Pattern: SNGP_R{int}_LS{float}_OS{float}_{int}{string}
    pattern = r'SNGP_R(\d+)_LS([\d.]+)_OS([\d.]+)_(\d+)([A-Za-z]+)'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'rank': int(match.group(1)),
            'lengthscale': float(match.group(2)),
            'outputscale': float(match.group(3)),
            'num_examples': int(match.group(4)),
            'dataset': match.group(5)
        }
    return None


def collect_sngp_results(target_dir):
    """
    Recursively collect all test_results.json files from SNGP folders.
    
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
            
            # Check if this is an SNGP folder
            if folder_name.startswith('SNGP_R'):
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
    parser = argparse.ArgumentParser(description="Collect SNGP test results into a CSV")
    parser.add_argument("target_dir", type=str, help="Root directory to search for results")
    parser.add_argument("--output", type=str, default="sngp_final_results.csv", 
                        help="Output CSV filename (default: sngp_final_results.csv)")
    
    args = parser.parse_args()
    
    print(f"Searching for SNGP results in: {args.target_dir}")
    
    # Collect all results
    results = collect_sngp_results(args.target_dir)
    
    if not results:
        print("No SNGP results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by dataset, num_examples, rank for readability
    df = df.sort_values(['dataset', 'num_examples', 'rank'])
    
    # Reorder columns for clarity
    column_order = [
        'dataset', 'num_examples', 'rank', 'lengthscale', 'outputscale',
        'test_log_likelihood', 'test_mse', 'test_rmse', 'folder_path'
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\nFound {len(results)} SNGP results")
    print(f"Results saved to: {args.output}")
    print(f"\nSummary by dataset:")
    print(df.groupby('dataset').size())
    print(f"\nPreview:")
    print(df.head(10))


if __name__ == "__main__":
    main()