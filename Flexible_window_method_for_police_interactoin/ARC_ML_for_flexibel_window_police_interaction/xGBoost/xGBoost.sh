#!/bin/bash

#SBATCH --job-name=ArrayJobScript_xGBoost
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH --output=ArrayJobScript_xGBoost_%A.out
#SBATCH --mem=16GB

# Print job information
echo "xGBoost file with task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname -s)"
echo

# Read parameters from file
index=0
while read line; do
    LINEARRAY[$index]="$line"
    index=$(($index+1))
done < params_xGBoost.txt

# Print selected parameter
echo "Selected parameter: ${LINEARRAY[$SLURM_ARRAY_TASK_ID]}"

# Run the Python program
echo "Starting main program Python code for xGBoost"
python xGBoost/xGBoost.py ${LINEARRAY[$SLURM_ARRAY_TASK_ID]}
echo "Ending SLURM script for xGBoost training"
