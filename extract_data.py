#!/usr/bin/env python3

import os
import re
import yaml
import json
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def extract_config_from_yaml(yaml_path):
    """Extract num_layers and num_heads from YAML config file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['num_layers'], config['num_heads']

def extract_accuracy_from_tensorboard(log_dir):
    """Extract accuracy per depth from TensorBoard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get all scalar tags
    tags = event_acc.Tags()['scalars']
    
    # Find accuracy per depth tags
    accuracy_tags = [tag for tag in tags if tag.startswith('val/acc_per_depth/')]
    
    accuracies = {}
    for tag in accuracy_tags:
        # Extract depth from tag (e.g., 'val/acc_per_depth/0' -> 0)
        depth = int(tag.split('/')[-1])
        
        # Get the scalar data
        scalar_events = event_acc.Scalars(tag)
        if scalar_events:
            # Get the final accuracy value
            final_accuracy = scalar_events[-1].value
            accuracies[depth] = final_accuracy
    
    return accuracies

def get_sweep_mapping():
    """Map sweep numbers to (layers, heads) configuration"""
    return {
        0: (2, 1),  # sweep 0 -> 2 layers, 1 head
        1: (3, 1),  # sweep 1 -> 3 layers, 1 head  
        2: (4, 1),  # sweep 2 -> 4 layers, 1 head
        3: (5, 1),  # sweep 3 -> 5 layers, 1 head
        4: (2, 2),  # sweep 4 -> 2 layers, 2 heads
        5: (3, 2),  # sweep 5 -> 3 layers, 2 heads
        6: (4, 2),  # sweep 6 -> 4 layers, 2 heads
        7: (5, 2),  # sweep 7 -> 5 layers, 2 heads
    }

def main():
    logs_dir = Path("/home/michael-lutz/iluvattnshun/logs/var_rename")
    
    # Find runs 43-50
    run_data = []
    
    for run_num in range(43, 51):
        run_dirs = list(logs_dir.glob(f"run_{run_num}_sweep_*"))
        if not run_dirs:
            print(f"Warning: No directory found for run {run_num}")
            continue
            
        run_dir = run_dirs[0]  # Take first match
        
        # Extract sweep number from directory name
        sweep_match = re.search(r'sweep_(\d+)', run_dir.name)
        if not sweep_match:
            print(f"Warning: Could not extract sweep number from {run_dir.name}")
            continue
        sweep_num = int(sweep_match.group(1))
        
        # Get config from YAML
        yaml_path = run_dir / f"{run_dir.name}.yaml"
        if not yaml_path.exists():
            print(f"Warning: YAML file not found for {run_dir.name}")
            continue
            
        try:
            num_layers, num_heads = extract_config_from_yaml(yaml_path)
            
            # Extract accuracy from TensorBoard logs
            accuracies = extract_accuracy_from_tensorboard(str(run_dir))
            
            run_data.append({
                'run_num': run_num,
                'sweep_num': sweep_num,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'accuracies': accuracies
            })
            
            print(f"Processed run {run_num} (sweep {sweep_num}): {num_layers} layers, {num_heads} heads")
            
        except Exception as e:
            print(f"Error processing run {run_num}: {e}")
            continue
    
    # Save extracted data to JSON file
    output_file = "/home/michael-lutz/iluvattnshun/extracted_data.json"
    with open(output_file, 'w') as f:
        json.dump(run_data, f, indent=2)
    
    print(f"\nData extracted and saved to: {output_file}")
    print(f"Total runs processed: {len(run_data)}")

if __name__ == "__main__":
    main()