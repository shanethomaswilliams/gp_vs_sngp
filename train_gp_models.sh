#!/bin/bash
#
# Comprehensive Hyperparameter Search for Abstention Model
# Usage: bash comprehensive_abstention_search.sh ACTION_NAME
#
# This script systematically explores the hyperparameter space to find
# optimal settings for uncertainty quantification and abstention behavior

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi


########################### FIXED CONFIGURATION ################################
export model_name="GP"
export rank=0
export tr_ratio=0.99
export hyperparam_lr=0.01
export savePath="/cluster/tufts/hugheslab/swilli26/stat-patt-final/gp_vs_sngp/results/GP_FINAL_RESULTS"
export learn_hyperparams="False"
export lr=0.0005
export n_epochs=10000


########################### HYPERPARAMETER SEARCH SPACE ################################
declare -a datasets=("Sin" "CrazySin" "Friedman")
# declare -a datasets=("Friedman")
declare -a num_examples=($(seq 5000 5000 75000))
# declare -a num_examples=(1000)
declare -a seeds=(1001)


########################### EXPERIMENT EXECUTION ################################
total_experiments=0
for dataset in "${datasets[@]}"; do
    for num_example in "${num_examples[@]}"; do
        for seed in "${seeds[@]}"; do
            ((total_experiments++))
        done
    done
done

echo "Total experiments to run: $total_experiments"
echo "Estimated time with early stopping: $((total_experiments * 15 / 60)) hours"

# Run the comprehensive search
experiment_count=0
for dataset in "${datasets[@]}"; do
    for num_example in "${num_examples[@]}"; do
        for seed in "${seeds[@]}"; do
            ((experiment_count++))
        
            export dataset=$dataset
            export num_example=$num_example
            export eval_dir="$savePath/GP/$dataset_$num_example"
            export seed=$seed
            case "$dataset" in
                Sin)
                    export lengthscale=1.226761
                    export outputscale=1.805912
                    export noise=0.100047
                    ;;
                CrazySin)
                    export lengthscale=0.276328
                    export outputscale=2.735109
                    export noise=0.099198
                    ;;
                Friedman)
                    export lengthscale=1.154271
                    export outputscale=26.312668
                    export noise=0.137135
                    ;;
            esac
            
            echo "[$experiment_count/$total_experiments] Running: Gaussian Process on Dataset: $dataset, N = $num_example"
            
            if [[ $ACTION_NAME == 'submit' ]]; then
                sbatch < ./do_experiment.slurm
                
            elif [[ $ACTION_NAME == 'run_here' ]]; then
                bash ./do_experiment.slurm
            
            elif [[ $ACTION_NAME == 'list' ]]; then
                echo "Would run: Gaussian Process on Dataset: $dataset, N = $num_example"
            fi
        done
    done
done