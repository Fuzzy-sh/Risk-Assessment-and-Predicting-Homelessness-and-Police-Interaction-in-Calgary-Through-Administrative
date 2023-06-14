#!/bin/bash

echo "Starting Preprocessing from top level Fuzzy script"
echo "Today is $(date)"
echo "---------------------------------------------------"

# send the parameters and write them down to the params_xGBoost.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 

echo "Creating the params for xGBoost"

max_depth=( 3 5 6 10 15 20 )
learning_rate=( 0.01 0.1 0.2 0.3 0.4)
subsample=( 0.7 0.8 0.9 )
colsample_bytree=( 0.4 0.5 0.6 0.7 0.8 0.9 )
colsample_bylevel=( 0.4 0.5 0.6 0.7 0.8 0.9 )
n_estimators=( 100 500 1000 )

echo "max_depth parameters are: ${max_depth[@]}"
echo "learning_rate parameters are: ${learning_rate[@]}"
echo "subsample parameters are: ${subsample[@]}"
echo "colsample_bytree parameters are: ${colsample_bytree[@]}"
echo "colsample_bylevel parameters are: ${colsample_bylevel[@]}"
echo "n_estimators parameters are: ${n_estimators[@]}"
echo "---------------------------------------------------"

jobCounter=0

echo "starting point for the job counter for xGBoost is: $jobCounter"

rm params_xGBoost.txt

for max_depth in "${max_depth[@]}" ; do
  for learning_rate in "${learning_rate[@]}" ; do
    for subsample in "${subsample[@]}" ; do 
      for colsample_bytree in "${colsample_bytree[@]}" ; do
        for colsample_bylevel in "${colsample_bylevel[@]}" ; do
          for n_estimators in "${n_estimators[@]}" ; do
            echo ${max_depth} ${learning_rate} ${subsample} ${colsample_bytree} ${colsample_bylevel} ${n_estimators} ${jobCounter} >> params_xGBoost.txt
            jobCounter=$((jobCounter+1))
          done
        done
      done
    done
  done
done

echo "Number of the parameters created in params_xGBoost.txt are: $jobCounter"
echo "---------------------------------------------------"

# Submit jobs

step=100
start=0
echo "The step for submitting the jobs is: $step"
echo "Job counter at the start of: $start"
declare -a jobID


for (( COUNTER=$start; COUNTER<=$jobCounter; COUNTER+=step )); do
  # Wait until the previous batch of jobs has finished
  
  while true;
  do
    # Get the number of jobs currently running or queued
    num_jobs=$(squeue -u faezehsadat.shahidi -h -t pending,running -r | wc -l)
    
    # If the number of jobs is less than 4000, break out of the loop and submit the next batch
    if (( $num_jobs < 3899 ))
    then
      break
    else
      # If the number of jobs is 4000 or more, wait for 10 minutes before checking again
      echo "Waiting for previous batch of jobs to finish $num_jobs jobs"
      sleep 600
    fi
  done
  
  # Submit the next batch of jobs
  end=$((COUNTER+step-1))
  jobID[$COUNTER]=$(sbatch --array=$COUNTER-$end xGBoost/xGBoost.sh)
  
  echo "Job counter at the end of $end"
  echo "Submitted by job id number: ${jobID[COUNTER]}"

done

echo "---------------------------------------------------"
echo "End of xGBoost"
echo "---------------------------------------------------"