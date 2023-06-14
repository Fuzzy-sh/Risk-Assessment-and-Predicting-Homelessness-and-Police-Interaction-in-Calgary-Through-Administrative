#!/bin/bash

# Print informative messages about the current execution stage
echo "Creating the params for Random Forest"
echo "---------------------------------------------------"

# Define the parameter values to explore
bootstrap=( 'True' )
max_depth=( 10 20 30 40 50 60 70 80 90 100 )
max_features=( 'sqrt' 'log2')
min_samples_leaf=( 1 2 4 8 )
min_samples_split=( 2 5 10 )
n_estimators=( 400 600 800 1000 1200 1400 1600 1800 2000 )
criterion=( 'gini' 'entropy')

# Print the parameters to explore
echo "bootstrap parameters are: ${bootstrap[*]}"
echo "max_depth parameters are: ${max_depth[*]}"
echo "max_features parameters are: ${max_features[*]}"
echo "min_samples_leaf parameters are: ${min_samples_leaf[*]}"
echo "min_samples_split parameters are: ${min_samples_split[*]}"
echo "n_estimators parameters are: ${n_estimators[*]}"
echo "criterion parameters are: ${criterion[*]}"

echo "---------------------------------------------------"
echo "Creating the parameter combinations"
echo "---------------------------------------------------"

# Generate all parameter combinations and save them to a file
jobCounter=0
echo "Starting point for the job counter is: $jobCounter"
rm params_RF.txt
    
for bootstrap in "${bootstrap[@]}" ; do
    for max_depth in "${max_depth[@]}" ; do
        for max_features in "${max_features[@]}" ; do 
            for min_samples_leaf in "${min_samples_leaf[@]}" ; do
                for min_samples_split in "${min_samples_split[@]}" ; do
                    for n_estimators in "${n_estimators[@]}" ; do
                        for criterion in "${criterion[@]}" ; do
                            echo ${bootstrap} ${max_depth} ${max_features} ${min_samples_leaf} ${min_samples_split} ${n_estimators} ${criterion} ${jobCounter} >> params_RF.txt
                            jobCounter=$((jobCounter+1))
                        done
                    done
                done
            done
        done
    done
done

echo "Number of parameter combinations created in params_RF.txt: $jobCounter"
echo "---------------------------------------------------"

# Submit jobs to the computing cluster
echo "Submitting the jobs to the computing cluster"
echo "---------------------------------------------------"

# Define the batch size and start point for job submission

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
  jobID[$COUNTER]=$(sbatch --array=$COUNTER-$end RF/RF.sh)
  
  echo "Job counter at the end of $end"
  echo "Submitted by job id number: ${jobID[COUNTER]}"

done

echo "---------------------------------------------------"
echo "End of Random Forest"
echo "---------------------------------------------------"

