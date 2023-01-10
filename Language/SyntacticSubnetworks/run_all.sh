export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language
export CONFIG_DIR=configs/Anaphora_Singular/BERT_Small/Mask_Training/Random_Configs
for file in ${CONFIG_DIR}/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/SyntacticSubnetworks/${CONFIG_DIR}/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/SyntacticSubnetworks/run.script
done

