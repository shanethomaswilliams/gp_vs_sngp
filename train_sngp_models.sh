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
export model_name="SNGP"
export rank=0
export tr_ratio=0.99
export hyperparam_lr=0.01
export savePath="/Users/shanewilliams/GradSchool/Fall2025/Statistical_Pattern_Recognition/final-project/gp_vs_sngp/results/SNGP_FINAL_RESULTS"
export learn_hyperparams="False"
export lr=0.00000005
export n_epochs=50000


########################### HYPERPARAMETER SEARCH SPACE ################################
declare -a datasets=("Friedman")
# declare -a datasets=("Friedman")
# declare -a num_examples=($(seq 5000 5000 75000))
# declare -a num_examples=(1000 5000 10000 25000)
declare -a num_examples=(1000)
# declare -a num_examples=(50000)
# declare -a rank_percent=(1 5 15 25 50 75 100)
declare -a rank_percent=(500 75 50 25 15 5 1)

declare -a seeds=(1001)


########################### EXPERIMENT EXECUTION ################################
total_experiments=0
for dataset in "${datasets[@]}"; do
    for num_example in "${num_examples[@]}"; do
        for seed in "${seeds[@]}"; do
            for rank_per in "${rank_percent[@]}"; do
                ((total_experiments++))
            done
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
            for rank_per in "${rank_percent[@]}"; do
                ((experiment_count++))
            
                export dataset=$dataset
                export num_example=$num_example
                export eval_dir="$savePath/GP/$dataset_$num_example"
                export seed=$seed
                export rank=$(( (num_example * rank_per + 99) / 100 ))
                case "$dataset" in
                    Sin)
                        export lengthscale=1.25
                        export outputscale=1.75
                        export noise=0.1
                        ;;
                    CrazySin)
                        export lengthscale=0.25
                        export outputscale=2.75
                        export noise=0.1
                        ;;
                    Friedman)
                        export lengthscale=1.25
                        export outputscale=10.0
                        export noise=0.125
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
done