export PROJECT_DIR=/users/mlepori/scratch/Compositional_Subnetworks
for file in configs/CompositionalSubnetworks/RN50_Inside_Count/Experiment_Configs/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/scratch/Compositional_Subnetworks/CVR/configs/CompositionalSubnetworks/RN50_Inside_Count/Experiment_Configs/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/CVR/run.script
done
