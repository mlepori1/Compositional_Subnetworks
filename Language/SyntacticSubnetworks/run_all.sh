export PROJECT_DIR=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/
for file in configs/CompositionalSubnetworks/BERT_SVAgr_Sing/Test_Configs/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/SyntacticSubnetworks/configs/CompositionalSubnetworks/BERT_SVAgr_Sing/Test_Configs/${JOBNAME}.yaml

 
    sbatch -J CS-$JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/Language/SyntacticSubnetworks/run.script
done

