export PROJECT_DIR=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks
for file in configs/CompositionalSubnetworks/RN50_Param_Search/Contact_Subnetwork/Test/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/CVR/configs/CompositionalSubnetworks/RN50_Param_Search/Contact_Subnetwork/Test/${JOBNAME}.yaml

    sbatch -J CS-$JOBNAME $PROJECT_DIR/CVR/test.script
done

