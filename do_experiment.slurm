#!/usr/bin/env bash
#SBATCH --job-name=SNGP_STAT_PATT    # Job name
#SBATCH --output=/Users/shanewilliams/GradSchool/Fall2025/Statistical_Pattern_Recognition/final-project/gp_vs_sngp/results/SNGP_FINAL_RESULTS/logs/%x_%j.out         # Output file (%j is job ID, %x is job name)
#SBATCH --error=/Users/shanewilliams/GradSchool/Fall2025/Statistical_Pattern_Recognition/final-project/gp_vs_sngp/results/SNGP_FINAL_RESULTS/logs/%x_%j.err          # Error file (%j is job ID, %x is job name)
#SBATCH --time=03:00:00            # Time limit (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=128G                 # Memory per node
#SBATCH --partition=batch


# Load any necessary modules for your environment
module load cuda/11.2
source activate Echo_MIL

modelID=$(printf "SNGP_R%d_LS%.2f_OS%.2f_%d%s" \
    "$rank" "$lengthscale" "$outputscale" "$num_example" "$dataset")
eval_dir="${savePath}/SNGP/${modelID}"
mkdir -p "${eval_dir}/logs"
echo "eval dir: "
echo $eval_dir

JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}
LOG_DIR="/Users/shanewilliams/GradSchool/Fall2025/Statistical_Pattern_Recognition/final-project/gp_vs_sngp/results/SNGP_FINAL_RESULTS/logs"
OUTPUT_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
ERROR_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err"

echo "${eval_dir}/logs/logs.out"
echo $OUTPUT_FILE
echo $ERROR_FILE

# Create symbollic links to redirect output files to train_dir
python /Users/shanewilliams/GradSchool/Fall2025/Statistical_Pattern_Recognition/final-project/gp_vs_sngp/main.py \
    --modelName $model_name\
    --rank $rank\
    --lenScale $lengthscale\
    --outScale $outputscale\
    --noise $noise\
    --lr $lr\
    --n_epochs $n_epochs\
    --dataset $dataset\
    --num_examples $num_example\
    --tr_ratio $tr_ratio\
    --seed $seed\
    --savePath $savePath\
    --learn_hyperparams $learn_hyperparams\


PYTHON_EXIT_CODE=$?


# Copy log files to output directory
if [ -d "${eval_dir}" ]; then
    mkdir -p "${eval_dir}/logs"
    
    cp "${OUTPUT_FILE}" "${eval_dir}/logs/logs.out"
    cp "${ERROR_FILE}" "${eval_dir}/logs/logs.err"
    
    # Delete original files
    # rm -f "${OUTPUT_FILE}"
    # rm -f "${ERROR_FILE}"
fi


conda deactivate

    
    