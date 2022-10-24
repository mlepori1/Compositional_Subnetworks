export PROJECT_DIR=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks
for file in configs/CompositionalSubnetworks/RN18_Train_Composition/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/CVR/configs/CompositionalSubnetworks/RN18_Train_Composition/${JOBNAME}.yaml
    sbatch -J CS-$JOBNAME $PROJECT_DIR/CVR/run.script
done

