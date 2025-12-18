print("Starting imports...", flush=True)
import argparse
import torch
import os
from sngp import train, model
from sngp.viz_utils import save_sngp_fig
from gaussian_process.true_gp import FullGPCholesky, train_gp, save_fig, evaluate_gp
from dataset.makeDataLoaders import load_data_test, load_data_train
import json
import numpy as np
import scipy.stats

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser(description = "Arguments for running experiment")

    #Add model specific arguments
    parser.add_argument("--modelName", type=str, default="GP", help="Name of model to train, SNGP or GP")
    parser.add_argument("--rank", type=int, default=1000, help = "Rank to run for SNGP, not need for GP")
    parser.add_argument("--lenScale", type=float, default=1)
    parser.add_argument("--outScale", type=float, default = 1)
    parser.add_argument("--noise", type=float, default=0.1)

    #Training arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training SNGP")
    parser.add_argument("--n_epochs", type=int, default=10_000, help="Number of epochs to train SNGP")
    parser.add_argument("--lr", type=float, default=0.000005, help="Learning rate for SNGP optimizer")

    #Add arguments for dataset
    parser.add_argument("--dataset", default="Sin", choices=["Sin", "CrazySin", "Friedman"], 
                        help="name of dataset to run")
    parser.add_argument("--num_examples", type=int, default=2_000, help="Number of examples to use for training")
    parser.add_argument("--tr_ratio", type = float, default= 0.7, help="Percent of data to use as training")
    parser.add_argument("--learn_hyperparams", type=str2bool, default=False, help="Whether or nt you learn hyperparemeters")
    parser.add_argument("--n_epochs_gp_train", type=int, default=10_000, help="Number of epochs to train GP hyperparams")

    #Other Useful Args
    parser.add_argument("--seed", type = int, default=42)
    parser.add_argument("--savePath", default="./results")

    return parser


def score(y_N, mean_N, var_N, noise):
    ''' Compute the average log probability of provided dataset under multivariate Gaussian
    
    Computes: log p(t | X) where t ~ N(mean, Cov)
    
    Formula:
        log p(t) = -1/2 * (t - μ)^T Σ^{-1} (t - μ) - 1/2 log|Σ| - N/2 log(2π)

    Args
    ----
    y_N : 1D tensor, shape (N,)
        True outputs
    mean_N : 1D tensor, shape (N,)
        Predicted means
    cov_NN : 2D tensor, shape (N, N)
        Covariance matrix of the predictive distribution

    Returns
    -------
    avg_log_proba : float
        Average log probability per data point
    '''
    y_np = y_N.detach().cpu().numpy()
    N = y_np.shape[0]
    
    # Add jitter for numerical stability
    # jitter = 1e-3
    var_N = var_N + noise**2
    mean_np = mean_N.detach().cpu().numpy()
    var_np = var_N.detach().cpu().numpy()
    std_np = np.sqrt(var_np)

    total_log_proba = scipy.stats.norm.logpdf(y_np, mean_np, std_np)
    return np.sum(total_log_proba) / N

def score_mse_rmse(y_N, mean_N):
    ''' Compute the MSE and RMSE between true and predicted means
    
    Args
    ----
    y_N : 1D tensor, shape (N,)
        True outputs
    mean_N : 1D tensor, shape (N,)
        Predicted means

    Returns
    -------
    mse : float
        Mean Squared Error
    rmse : float
        Root Mean Squared Error
    '''
    residual_N = y_N - mean_N
    mse = torch.mean(residual_N ** 2).item()
    rmse = torch.sqrt(torch.mean(residual_N ** 2)).item()
    return mse, rmse

def score_total(t_N, mean_N, cov, noise=0.0):
    ''' Compute the total log probability under multivariate Gaussian
    
    Formula:
        log p(t) = -1/2 * (t - μ)^T Σ^{-1} (t - μ) - 1/2 log|Σ| - N/2 log(2π)
    
    Args
    ----
    t_N : 1D tensor, shape (N,)
        True outputs
    mean_N : 1D tensor, shape (N,)
        Predicted mean
    cov_NN : 2D tensor, shape (N, N)
        Covariance matrix

    Returns
    -------
    total_log_proba : float
        Total log probability
    '''
    N = t_N.shape[0]
    residual_N = t_N - mean_N
    
    # Add jitter for numerical stability
    jitter = 1e-6
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-4  # larger than 1.9e-5
    cov_stable = cov + (noise**2 + jitter) * torch.eye(N, device=cov.device, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov_stable)
    
    try:
        # Cholesky decomposition: Σ = L L^T
        L = torch.linalg.cholesky(cov_stable)
        
        # Solve L α = residual for α
        alpha = torch.linalg.solve_triangular(L, residual_N.unsqueeze(1), upper=False).squeeze()
        
        # Mahalanobis distance: (t - μ)^T Σ^{-1} (t - μ) = α^T α
        mahalanobis = alpha @ alpha
        
        # Log determinant: log|Σ| = 2 * sum(log(diag(L)))
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        
    except RuntimeError:
        # Fallback: eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_stable)
        eigenvalues_clamped = eigenvalues.clamp(min=1e-8)
        
        # Σ^{-1} = V @ diag(1/λ) @ V^T
        cov_inv = eigenvectors @ torch.diag(1.0 / eigenvalues_clamped) @ eigenvectors.T
        mahalanobis = residual_N @ cov_inv @ residual_N
        log_det = torch.sum(torch.log(eigenvalues_clamped))
    
    # Compute total log-likelihood
    log_2pi = torch.log(torch.tensor(2 * torch.pi))
    total_log_proba = -0.5 * (mahalanobis + log_det + N * log_2pi)
    
    # Return as Python float
    return total_log_proba.item() / N


if __name__ == '__main__':
    #Get arguments
    print("Parsing arguments...", flush=True)
    parser = get_args_parser()
    args = parser.parse_args()
    print(f"Arguments parsed! Running {args.modelName} on {args.dataset} with N={args.num_examples}", flush=True)
    print("EXPERIMENT SUMMARY", flush=True)
    print("===================", flush=True)
    print("  - Model Training for SNGP and GP on Regression Tasks", flush=True)
    print(f"  - modelName: {args.modelName}", flush=True)
    print(f"  - dataset: {args.dataset}", flush=True)
    print(f"  - num_examples: {args.num_examples}", flush=True)
    print(f"  - seed: {args.seed}", flush=True)
    print(f"  - rank (for SNGP): {args.rank}", flush=True)
    print(f"  - lengthscale: {args.lenScale}", flush=True)
    print(f"  - outputscale: {args.outScale}", flush=True)
    print(f"  - noise: {args.noise}", flush=True)
    print("===================", flush=True)

    # if args.modelName == "GP":
    #     torch.set_default_dtype(torch.float64)

    #Check if cuda is avaliable
    print("Checking CUDA availability...", flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(device))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    #Load Datasets
    print("Making Datasets...", flush=True)
    GP_train, GP_test, SNGP_train, SNGP_val, SNGP_test, norm_stats = load_data_train("./dataset/Data/Noisy/"+args.dataset,
                                                             args.num_examples,
                                                             train_percentage= args.tr_ratio,
                                                             random_state=args.seed)
    # GP_test, SNGP_test = load_data_test("./dataset/Data/Noisy/"+args.dataset,
    #                                      random_state=args.seed)
    # GP_test_clean, SNGP_test_clean = load_data_test("./dataset/Data/Clean/"+args.dataset,
    #                                      random_state=args.seed)
    
    for x, y in SNGP_val:
        F = x.shape[1]



    #Make Correct Model and Train
    print("Making Model...", flush=True)
    if args.modelName == "SNGP":
        print("Training SNGP Model...", flush=True)

        saveFolder = args.savePath
        modelID = "SNGP_R%d_LS%.2f_OS%.2f_%d%s" %(args.rank,
                                                  args.lenScale,
                                                  args.outScale,
                                                  args.num_examples,
                                                  args.dataset)
        os.makedirs(saveFolder + f"/SNGP/{modelID}", exist_ok=True)

        #Make SNGP Model
        sngp_model = model.RFFGP_Reg(in_features=F,
                                     out_features=1,
                                     rank = args.rank,
                                     lengthscale= args.lenScale,
                                     outputscale= args.outScale).to(device)
        
        #Trained SNGP Model
        #Note while playing around I found it is incredibly senstive to learning rate
        # I.E. 0.001 diverges but 0.0005 converges so be careful with it
        sngp_model, info = train.train_model(sngp_model, device, 
                                                SNGP_train, SNGP_val, l2pen_mag=1.0,
                                                n_epochs=args.n_epochs, lr=args.lr, do_early_stopping=True)
        torch.save(info, saveFolder + f"/SNGP/{modelID}/training_info.pt")
        if args.rank < 25_000:
            model_path = saveFolder + f"/SNGP/{modelID}/model.pt"
            torch.save({
                'model_state_dict': sngp_model.state_dict(),
                'rank': args.rank,
                'lengthscale': args.lenScale,
                'outputscale': args.outScale,
                'in_features': F,
                'out_features': 1
            }, model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Rank {args.rank} >= 25,000, skipping model save")

        print("Evaluating SNGP Model...", flush=True)
        print("Computing covariance matrix...", flush=True)
        sngp_model.update_precision_from_loader(SNGP_train, device=device)
        print("Inverting covariance matrix...", flush=True)
        covariance = sngp_model.invert_covariance(device=device)

        print(f"Covariance shape BEFORE: {covariance.shape}")

        # It should be (R, R), not (1, R, R) or something weird
        if covariance.dim() == 3:
            covariance = covariance.squeeze(0) 


        for X_test, y_test in GP_test:
            print("Scoring on test data...", flush=True)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            mean, var = sngp_model.get_mean_variance(X_test, covariance)
            print("Computing test scores...", flush=True)

            test_ll = score(y_test, mean, var, noise=0.1)
            # test_ll_2 = score_total(y_test, mean, cov, noise=0.1)
            print("TEST LOG LIKELIHOOD SCORE: ", test_ll)
            # print("TEST LOG LIKELIHOOD SCORE: ", test_ll_2)
            mse, rmse = score_mse_rmse(y_test, mean)
            print("TEST MSE: ", mse)
            print("TEST RMSE: ", rmse)

            json_file = saveFolder + f"/SNGP/{modelID}/test_results.json"
            with open(json_file, "w") as json_file:
                results = {
                    "test_log_likelihood": test_ll,
                    "test_mse": mse,
                    "test_rmse": rmse
                }
                json.dump(results, json_file, indent=4)

            if args.dataset == "Sin" or args.dataset == "CrazySin":
                fig_path = saveFolder + f"/SNGP/{modelID}/test_fig.png"
                save_sngp_fig(sngp_model, X_test, y_test, covariance, args.dataset, fig_path, seed=42)
    elif args.modelName == "GP":
        print("Training Gaussian Process...", flush=True)

        saveFolder = args.savePath
        modelID = "GP_%d%s" %(args.num_examples,
                              args.dataset)
        os.makedirs(saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}", exist_ok=True)

        gp = FullGPCholesky(lengthscale= args.lenScale, outputscale= args.outScale, noise= args.noise, learn_hyperparams=args.learn_hyperparams)
        for X_train, y_train in GP_train:
            if args.learn_hyperparams:
                info = train_gp(gp, X_train, y_train, n_iterations=args.n_epochs_gp_train, lr=0.01)
                hyperparam_dict = {
                    "lengthscale": gp.kernel.lengthscale.item(),
                    "outputscale": gp.kernel.outputscale.item(),
                    "noise": gp.noise.item(),
                    "marginal_likelihood_loss_history": info 
                }

                print("Learned Hyperparams: ")
                print(f"Lengthscale: {hyperparam_dict['lengthscale']}")
                print(f"Outputscale: {hyperparam_dict['outputscale']}")
                print(f"Noise: {hyperparam_dict['noise']}")

                # Save to JSON
                json_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/hyperparams.json"
                with open(json_path, "w") as f:
                    json.dump(hyperparam_dict, f, indent=4)
            print("Fitting to data...", flush=True)
            gp.fit(X_train, y_train)
            
            if args.num_examples < 25_000:
                model_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/model.pt"
                torch.save({
                    'model_state_dict': gp.state_dict(),
                    'X_train': X_train,
                    'y_train': y_train,
                    'num_examples': args.num_examples,
                    'dataset': args.dataset
                }, model_path)
                print(f"Model saved to {model_path}")
            else:
                print(f"Num examples {args.num_examples} >= 25,000, skipping model save")

        for X_test, y_test in GP_test:
            results = evaluate_gp(gp, X_test, y_test)
            # print(results)
            mean, var, cov = gp.predict(X_test)

            test_ll = score(y_test, mean, var, gp.noise)
            # test_ll_2 = score_total(y_test, mean, cov, gp.noise)
            print("TEST LOG LIKELIHOOD SCORE: ", test_ll)
            # print("TEST LOG LIKELIHOOD SCORE: ", test_ll_2)
            mse, rmse = score_mse_rmse(y_test, mean)
            print("TEST MSE: ", mse)
            print("TEST RMSE: ", rmse)

            json_file = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/test_results.json"
            with open(json_file, "w") as json_file:
                results = {
                    "test_log_likelihood": test_ll,
                    "test_mse": mse,
                    "test_rmse": rmse
                }
                json.dump(results, json_file, indent=4)

            if args.dataset == "Sin" or args.dataset == "CrazySin":
                fig_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/test_fig.png"
                save_fig(gp, X_test, y_test, args.dataset, fig_path)