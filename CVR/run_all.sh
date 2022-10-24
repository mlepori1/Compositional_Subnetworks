export PROJECT_DIR=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks
for file in configs/CompositionalSubnetworks/RN18_Param_Search/Inside_Subnetwork/Train/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/CVR/configs/CompositionalSubnetworks/RN18_Param_Search/Inside_Subnetwork/Train/${JOBNAME}.yaml
    sbatch -J CS-$JOBNAME $PROJECT_DIR/CVR/run.script
done

