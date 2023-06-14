#!/bin/bash

#SBATCH --job-name=ArrayJobScript_RF
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --output=ArrayJobScript_RF_%A.out
#SBATCH --mem=16GB

# Exit immediately if a command fails
set -e

# Print job information
echo "Starting Random Forest job with task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname -s)"
echo

# Read the parameters from a file into an array
mapfile -t params < params_RF.txt

# Extract the parameter for the current task ID
param="${params[$SLURM_ARRAY_TASK_ID]}"

# Print the parameter and command to be run
echo "Using parameter: $param"
echo "Running command: python RF/RF.py $param"
echo

# Run the Python script
python RF/RF.py $param

echo "Random Forest job completed successfully"


                                              