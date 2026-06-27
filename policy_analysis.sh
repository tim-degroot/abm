#!/bin/bash
# =====================================================
# policy_analysis.sh
# Runs the policy analysis pipeline on the HPC cluster.
# Usage: sbatch policy_analysis.sh
# =====================================================
#SBATCH --job-name=abm-policy-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=policy_analysis_%j.out

export PATH="$HOME/.local/bin:$PATH"

cd $HOME/abm
uv run -m code.policy_analysis