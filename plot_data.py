#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from pathlib import Path

def get_label_mapping():
    """Map sweep numbers to readable labels"""
    return {
        0: "2 layers, 1 head",
        1: "3 layers, 1 head", 
        2: "4 layers, 1 head",
        3: "5 layers, 1 head",
        4: "2 layers, 2 heads",
        5: "3 layers, 2 heads",
        6: "4 layers, 2 heads", 
        7: "5 layers, 2 heads"
    }

def main():
    # Load extracted data
    data_file = "/home/michael-lutz/iluvattnshun/extracted_data.json"
    
    if not Path(data_file).exists():
        print(f"Error: Data file {data_file} not found. Run extract_data.py first.")
        return
    
    with open(data_file, 'r') as f:
        run_data = json.load(f)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    label_mapping = get_label_mapping()
    
    # Sort run_data by layers first, then by heads for consistent legend ordering
    def get_sort_key(run):
        return (run['num_layers'], run['num_heads'])
    
    sorted_runs = sorted(run_data, key=get_sort_key)
    
    # Group by configuration and plot
    for run in sorted_runs:
        sweep_num = run['sweep_num']
        accuracies = run['accuracies']
        num_heads = run['num_heads']
        
        if accuracies:  # Only plot if we have data
            depths = sorted([int(k) for k in accuracies.keys()])
            acc_values = [accuracies[str(d)] for d in depths]
            
            label = label_mapping.get(sweep_num, f"sweep_{sweep_num}")
            
            # Use dotted lines for 1 head configurations
            linestyle = '--' if num_heads == 1 else '-'
            plt.plot(depths, acc_values, label=label, linewidth=2, linestyle=linestyle)
    
    plt.xlabel('Variable Chain Depth (number of hops)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy per Dependency Depth for Different Model Configurations', fontsize=14)
    plt.xlim(0, 49)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = '/home/michael-lutz/iluvattnshun/accuracy_per_depth.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    main()