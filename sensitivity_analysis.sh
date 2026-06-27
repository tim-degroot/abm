#!/bin/bash
# =====================================================
# sensitivity.sh
# Runs the sensitivity analysis on the HPC cluster.
# Before running: uv run -m code.sensitivity --generate
# Usage: sbatch sensitivity_analysis.sh
# After running: uv run -m code.sensitivity --aggregate
# =====================================================
#SBATCH --job-name=abm-sa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=50:00:00
#SBATCH --output=sensitivity_%A_%a.out
#SBATCH --array=0-9%2

export PATH="$HOME/.local/bin:$PATH"
cd $HOME/abm
uv run -m code.sensitivity --evaluate --model-seed $SLURM_ARRAY_TASK_ID