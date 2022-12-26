export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision
for file in configs/Count_Contact/ViT12/Model_Hyperparameters/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/CVR/configs/Count_Contact/ViT12/Model_Hyperparameters/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/CVR/run.script
done

