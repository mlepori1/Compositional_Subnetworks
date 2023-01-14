export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision
export CONFIG_DIR=configs/Inside_Contact/ViT12/Model_Hyperparameters
for file in ${CONFIG_DIR}/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/CVR/${CONFIG_DIR}/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/CVR/run.script
done

