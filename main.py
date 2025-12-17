print("Starting imports...", flush=True)
import argparse
import torch
import os
from sngp import train, model
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
    parser.add_argument("--modelName", type=str, default="SNGP", help="Name of model to train, SNGP or GP")
    parser.add_argument("--rank", type=int, default=1000, help = "Rank to run for SNGP, not need for GP")
    parser.add_argument("--lenScale", type=float, default=0.25)
    parser.add_argument("--outScale", type=float, default = 1.0)

    #Add arguments for dataset
    parser.add_argument("--dataset", default="Sin", choices=["Sin", "CrazySin", "Friedman"], 
                        help="name of dataset to run")
    parser.add_argument("--num_examples", type=int, default=1_000, help="Number of examples to use for training")
    parser.add_argument("--tr_ratio", type = float, default= 0.7, help="Percent of data to use as training")
    parser.add_argument("--learn_hyperparams", type=str2bool, default=True, help="Whether or nt you learn hyperparemeters")

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

    total_log_proba = scipy.stats.norm.logpdf(
        y_N, mean_np, var_np)
    return np.sum(total_log_proba) / N

def score_total(t_N, mean_N, cov_NN):
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
    cov_stable = cov + (gp.noise**2 + jitter) * torch.eye(n_test, device=cov.device, dtype=cov.dtype)
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

    #Check if cuda is avaliable
    print("Checking CUDA availability...", flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")


    #Load Datasets
    print("Making Datasets...", flush=True)
    GP_train, SNGP_train, SNGP_val = load_data_train("./dataset/Data/Noisy/"+args.dataset,
                                                             args.num_examples,
                                                             train_percentage= args.tr_ratio,
                                                             random_state=args.seed)
    GP_test, SNGP_test = load_data_test("./dataset/Data/Noisy/"+args.dataset,
                                         random_state=args.seed)
    GP_test_clean, SNGP_test_clean = load_data_test("./dataset/Data/Clean/"+args.dataset,
                                         random_state=args.seed)
    
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
        os.makedirs(saveFolder + "/SNGP/info", exist_ok=True)

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
                                                n_epochs=1_000, lr=0.0005, do_early_stopping=True)
        torch.save(info, saveFolder +"/SNGP/info/" + modelID)

        sngp_model.update_precision_from_loader(SNGP_train)
        covariance = sngp_model.invert_covariance()

        for (X_test, y_test), (X_clean, y_clean) in zip(GP_test, GP_test_clean):
            mean, cov = sngp_model.get_mean_covaraince(X_test, covariance)

            print(mean.shape)
            print(cov.shape)

            test_ll = score(y_test, mean, cov)
            test_ll_2 = score_total(y_test, mean, cov)
            print("CP2 TEST LOG LIKELIHOOD SCORE: ", test_ll)
            print("TEST LOG LIKELIHOOD SCORE: ", test_ll_2)

            # if args.dataset == "Sin" or args.dataset == "CrazySin":
            #     fig_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/test_fig.png"
            #     save_fig(gp, X_test, args.dataset, fig_path)
    elif args.modelName == "GP":
        print("Training Gaussian Process...", flush=True)

        saveFolder = args.savePath
        modelID = "GP_%d%s" %(args.num_examples,
                              args.dataset)
        os.makedirs(saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}", exist_ok=True)

        gp = FullGPCholesky(lengthscale=1.0, outputscale=1.0, noise=0.2, learn_hyperparams=True)
        for X_train, y_train in GP_train:
            if args.learn_hyperparams:
                info = train_gp(gp, X_train, y_train, n_iterations=2_500, lr=0.01)
                hyperparam_dict = {
                    "lengthscale": gp.kernel.lengthscale.item(),
                    "outputscale": gp.kernel.outputscale.item(),
                    "noise": gp.noise.item(),
                    "marginal_likelihood_loss_history": info 
                }

                print("Learned Hyperparams: ", hyperparam_dict)

                # Save to JSON
                json_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/hyperparams.json"
                with open(json_path, "w") as f:
                    json.dump(hyperparam_dict, f, indent=4)

            gp.fit(X_train, y_train)
            # if args.num_examples < 10_000:
                # torch.save(info, saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/" + modelID)

        for (X_test, y_test), (X_clean, y_clean) in zip(GP_test, GP_test_clean):
            # results = evaluate_gp(gp, X_test, y_test)
            # print(results)
            mean, var, cov = gp.predict(X_test)

            print(mean.shape)
            print(cov.shape)

            print("\n" + "="*60)
            print("COVARIANCE DIAGNOSTICS")
            print("="*60)

            # Check if matrix is symmetric (it should be!)
            asymmetry = torch.max(torch.abs(cov - cov.T))
            print(f"Matrix asymmetry (should be ~0): {asymmetry.item():.2e}")

            # Check eigenvalues
            eigenvalues = torch.linalg.eigvalsh(cov)
            print(f"\nEigenvalue range: [{eigenvalues.min().item():.2e}, {eigenvalues.max().item():.2e}]")
            print(f"Number negative: {(eigenvalues < 0).sum().item()} / {len(eigenvalues)}")
            print(f"Most negative: {eigenvalues.min().item():.2e}")
            print(f"Smallest positive: {eigenvalues[eigenvalues > 0].min().item():.2e}")

            # Check condition number
            cond_num = eigenvalues.max() / eigenvalues[eigenvalues > 0].min()
            print(f"Condition number: {cond_num.item():.2e}")

            print(f"\nTest set size: {len(X_test)}")
            print(f"Hyperparameters:")
            print(f"  lengthscale: {gp.kernel.lengthscale.item():.4f}")
            print(f"  outputscale: {gp.kernel.outputscale.item():.4f}")
            print(f"  noise: {gp.noise.item():.4f}")
            print("="*60)

            test_ll = score(y_test, mean, var, gp.noise)
            test_ll_2 = score_total(y_test, mean, cov)
            print("CP2 TEST LOG LIKELIHOOD SCORE: ", test_ll)
            print("TEST LOG LIKELIHOOD SCORE: ", test_ll_2)

            json_file = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/test_results.json"
            with open(json_file, "w") as json_file:
                json.dump(results, json_file, indent=4)

            if args.dataset == "Sin" or args.dataset == "CrazySin":
                fig_path = saveFolder + f"/GP/{args.dataset}_{args.num_examples}_{args.seed}/test_fig.png"
                save_fig(gp, X_test, args.dataset, fig_path)



    

    
    
