#!/bin/bash

# Launches a grid of sweeps on different GPUs. Usage (from the root directory):
# chmod +x examples/launch_var_rename_sweep.sh
# ./examples/launch_var_rename_sweep.sh

# CONFIGURATION
n=8  # number of grid cells
GPUs=(0 1 2 3 4 5 6 7)  # List of available GPUs

# define your sweep parameters

# testing with 4 heads, 1-4 layers
sweeps[0]="--num_layers=1 --num_heads=4" # ablating away one hop shortcut (this should fail)
sweeps[1]="--num_layers=2 --num_heads=4" # 16 if 4^2, or 8 if only matters at front
sweeps[2]="--num_layers=3 --num_heads=4" # 32 if ..., 16 otherwise
sweeps[3]="--num_layers=4 --num_heads=4 --learning_rate=1e-3" # new learning rate with grad clipping
# testing with 8 layers, 1-4 heads
sweeps[4]="--num_layers=4 --num_heads=1" # should be able to hit 8
sweeps[5]="--num_layers=4 --num_heads=2" # maybe 16?? (depends on if can consistently use heads later down)
sweeps[6]="--num_layers=4 --num_heads=3" # maybe 24??
sweeps[7]="--num_layers=3 --num_heads=2" # to confirm the relationship...

SESSION_NAME="var_rename_sweep"

# kill existing session if needed
tmux kill-session -t $SESSION_NAME 2>/dev/null

# calculate grid dimensions (rows and cols)
cols=$(awk "BEGIN { print int(sqrt($n)) }")
rows=$(( (n + cols - 1) / cols ))  # ceil(n / cols)

# create new session
tmux new-session -d -s $SESSION_NAME -x "$(tput cols)" -y "$(tput lines)"
tmux rename-window -t $SESSION_NAME "sweeps"

# first pane is already created (pane 0)
pane_ids=()
pane_ids+=("0")

# create grid: split rows first
for ((i = 1; i < rows; i++)); do
    tmux split-window -v -t $SESSION_NAME:0.0
    tmux select-layout -t $SESSION_NAME tiled
done

# for each row, split horizontally into columns
for ((i = 0; i < rows; i++)); do
    for ((j = 1; j < cols; j++)); do
        tmux select-pane -t $SESSION_NAME:0.$((i * cols))
        tmux split-window -h -t $SESSION_NAME
        tmux select-layout -t $SESSION_NAME tiled
    done
done

# assign sweeps to panes
for ((i = 0; i < n; i++)); do
    gpu_index=$((i % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    cmd="CUDA_VISIBLE_DEVICES=$gpu python -m examples.var_rename ${sweeps[$i]} --run_name=sweep_${i}"
    tmux send-keys -t $SESSION_NAME:0.$i "$cmd" C-m
done

# attach to session
tmux attach -t $SESSION_NAME
