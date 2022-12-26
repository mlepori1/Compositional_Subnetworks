export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language
for file in configs/SV_Agr_Singular/BERT_Small_LM/Model_Hyperparameters/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/SyntacticSubnetworks/configs/SV_Agr_Singular/BERT_Small_LM/Model_Hyperparameters/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/SyntacticSubnetworks/run.script
done

