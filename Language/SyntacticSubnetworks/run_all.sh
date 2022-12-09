export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/
for file in configs/CompositionalSubnetworks/SV_Agreement_Singular_Hyperparameter_Sweep/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/SyntacticSubnetworks/configs/CompositionalSubnetworks/SV_Agreement_Singular_Hyperparameter_Sweep/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/Language/SyntacticSubnetworks/run.script
done

