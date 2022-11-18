export PROJECT_DIR=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks
for file in configs/CompositionalSubnetworks/RN50_Contact_Inside/test/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/CVR/configs/CompositionalSubnetworks/RN50_Contact_Inside/test/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/CVR/run.script
done

