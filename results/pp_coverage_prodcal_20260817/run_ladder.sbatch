#!/bin/bash
#SBATCH --job-name=prodcal_ladder
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=prodcal_ladder_%j.out
#SBATCH --error=prodcal_ladder_%j.err

# Prodcal calibration ladder (prereg fe72d52b..7d0b3f5f, AMENDMENT-1).
# One node, 26 workers = one worker per registered invocation (cluster-skill
# gotcha 7: match the reservation to the actual work count). Pure-CPU numpy
# harness — no GPU, no galaxy catalogue. Seeds live inside run_ladder.py
# (paired-seed scheme, base 20270818); no SLURM seed derivation on purpose:
# the registered scheme must not be layered over.
# Walltime: measured 23.5 s/realization for the longest cell => ~2.4 h
# uncontended; 8 h covers gotcha-6 contention margin. Idempotent: run_ladder
# skips cells whose JSON already exists, so a resubmit resumes.

set -euo pipefail
cd "$HOME/darksiren-emri"
source cluster/modules.sh
cd results/pp_coverage_prodcal_20260817
mkdir -p cells
../../.venv/bin/python -u run_ladder.py --workers 26
