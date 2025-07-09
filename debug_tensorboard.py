#!/usr/bin/env python3

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def debug_tensorboard_logs():
    """Debug what tags are available in TensorBoard logs"""
    logs_dir = Path("/home/michael-lutz/iluvattnshun/logs/var_rename")
    
    # Check one run to see what tags are available
    run_dir = logs_dir / "run_43_sweep_4"
    
    print(f"Checking TensorBoard logs in: {run_dir}")
    
    event_acc = EventAccumulator(str(run_dir))
    event_acc.Reload()
    
    # Get all available tags
    tags = event_acc.Tags()
    
    print("\nAvailable tag types:")
    for tag_type, tag_list in tags.items():
        if hasattr(tag_list, '__len__'):
            print(f"  {tag_type}: {len(tag_list)} tags")
            if tag_list:
                print(f"    Examples: {tag_list[:5]}")
        else:
            print(f"  {tag_type}: {tag_list}")
    
    # Focus on scalar tags
    scalar_tags = tags.get('scalars', [])
    print(f"\nAll scalar tags ({len(scalar_tags)}):")
    for tag in sorted(scalar_tags):
        print(f"  {tag}")
    
    # Look for accuracy-related tags
    accuracy_tags = [tag for tag in scalar_tags if 'accuracy' in tag.lower()]
    print(f"\nAccuracy-related tags ({len(accuracy_tags)}):")
    for tag in accuracy_tags:
        print(f"  {tag}")

if __name__ == "__main__":
    debug_tensorboard_logs()