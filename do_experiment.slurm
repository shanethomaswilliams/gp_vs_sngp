#!/usr/bin/env bash
#SBATCH --job-name=GP_VS_SNGP      # Job name
#SBATCH --output=/cluster/tufts/hugheslab/swilli26/stat-patt-final/gp_vs_sngp/results/test/logs/%x_%j.out         # Output file (%j is job ID, %x is job name)
#SBATCH --error=/cluster/tufts/hugheslab/swilli26/stat-patt-final/gp_vs_sngp/results/test/logs/%x_%j.err          # Error file (%j is job ID, %x is job name)
#SBATCH --time=01:00:00            # Time limit (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8         # CPU cores
#SBATCH --mem=128G                 # Memory per node
#SBATCH --partition=batch


# Load any necessary modules for your environment
module load cuda/11.2
source activate Echo_MIL

eval_dir="${savePath}/GP/${dataset}_${num_example}_${seed}"
JOB_ID=${SLURM_JOB_ID}
JOB_NAME=${SLURM_JOB_NAME}
LOG_DIR="/cluster/tufts/hugheslab/swilli26/stat-patt-final/gp_vs_sngp/results/test/logs"
OUTPUT_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
ERROR_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err"

# Create symbollic links to redirect output files to train_dir
python /cluster/tufts/hugheslab/swilli26/stat-patt-final/gp_vs_sngp/main.py \
    --modelName $model_name\
    --rank $rank\
    --lenScale $lengthscale\
    --outScale $outputscale\
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

    echo "${eval_dir}/logs"
    echo $OUTPUT_FILE
    echo $ERROR_FILE
    
    cp "${OUTPUT_FILE}" "${eval_dir}/logs/logs.out"
    cp "${ERROR_FILE}" "${eval_dir}/logs/logs.err"
    
    # Delete original files
    rm -f "${OUTPUT_FILE}"
    rm -f "${ERROR_FILE}"
fi


conda deactivate

    
    