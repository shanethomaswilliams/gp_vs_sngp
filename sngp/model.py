import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.nn.utils as nn_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
from scipy.stats import norm


class RFFGP_Reg(torch.nn.Module):
    def __init__(self, in_features, out_features, learnable_lengthscale=False, learnable_outputscale=False, 
                 lengthscale=0.1, outputscale=1.0, rank=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.learnable_lengthscale = learnable_lengthscale
        self.learnable_outputscale = learnable_outputscale
        if self.learnable_lengthscale:
            self.lengthscale_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(lengthscale))))
        else:
            self.lengthscale_param = torch.log(torch.expm1(torch.tensor(lengthscale)))
        if self.learnable_outputscale:
            self.outputscale_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(outputscale))))
        else:
            self.outputscale_param = torch.log(torch.expm1(torch.tensor(outputscale)))
        self.rank = rank
        #Out_Features X Rank X Rank
        self.precision_mat = torch.eye(self.rank, device='cpu').unsqueeze(0).repeat(self.out_features, 1, 1)
        self.register_buffer('feature_weight', torch.randn(self.rank, self.in_features))
        self.register_buffer('feature_bias', 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)

    def featurize(self, h):
        features = torch.nn.functional.linear(h, (1/self.lengthscale) * self.feature_weight, self.feature_bias)
        return self.outputscale * (2/self.rank)**0.5 * torch.cos(features)
    
    def forward(self, h):
        features = self.featurize(h)
        mean = self.linear(features)
        return mean
    

    def predict(self, h, covariance, num_samples=100):
        # CALCULAING FEATURES
        features_NR = self.featurize(h)

        ## AGAIN, IN OUR CASE THIS IS EQUIVALENT TO THE ENTIRE TRAINING DATASET
        batch_size = features_NR.shape[0]

        mean_pred_N = self.linear(features_NR)
        var_pred_N = []
        ## CREATING OUTLINE FOR VARIANCE TO DO MONTE CARLO SAMPLING
        for n in range(batch_size):
            features_R1 = features_NR[n]
            var_pred = features_R1.T @ covariance @ features_R1
            var_pred_N.append(var_pred)

        norm = torch.distributions.normal.Normal(loc=mean_pred_N, scale=torch.sqrt(var_pred_N[0]))

        #Compute samples from out Norm
        samples_NB = norm.sample(sample_shape=(num_samples,))

        # MCsamples_N = np.mean(samples_NB, axis=1)

        return samples_NB, mean_pred_N, var_pred_N         

    def update_precision_from_loader(self, data_loader, device='cpu'):
        self.eval()
        cov_inv_RR = self.precision_mat.to(device)

        with torch.no_grad():  # ← Add this! No gradients needed
            for X, _ in data_loader:
                X = X.to(device)
                
                # Featurize entire batch
                features_NR = self.featurize(X)  # (N, R)
                
                # Vectorized: Φᵀ @ Φ = sum of all outer products
                cov_inv_RR += features_NR.T @ features_NR  # (R, R)
        
        # Add identity at end
        cov_inv_RR += torch.eye(self.rank, device=device)
        
        # Update precision
        self.precision_mat = cov_inv_RR

    def get_mean_variance(self, X_test, covariance, noise=0.0, device='cpu'):
        features_NR = self.featurize(X_test)  # Should be (N, R)
        
        # Debug: Check shapes
        print(f"features_NR shape: {features_NR.shape}")
        print(f"covariance shape: {covariance.shape}")
        
        # Ensure features_NR is 2D: (N, R)
        if features_NR.dim() == 1:
            features_NR = features_NR.unsqueeze(0)  # (1, R) if only one sample
        
        mean_N = self.linear(features_NR)  # (N, 1)
        mean_N = mean_N.squeeze(-1)  # (N,)
        
        # Vectorized variance computation
        # var[i] = features[i] @ cov @ features[i].T
        var_N = torch.sum((features_NR @ covariance) * features_NR, dim=1)  # (N,)
        
        if noise > 0:
            var_N = var_N + noise**2
    
        print(f"Returning mean shape: {mean_N.shape}, var shape: {var_N.shape}")
        
        return mean_N, var_N

    def invert_covariance(self, device='cpu'):
        covariance_RR = torch.inverse(self.precision_mat)
        return covariance_RR.clone().detach().to(device)
    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.lengthscale_param)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.outputscale_param)
