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

        rank = self.rank

        # with torch.nograd():
        for X,_ in data_loader:
            X = X.to(device)

            #Just need features for regression case
            features_NR = self.featurize(X)
            batch_size = features_NR.shape[0]

            for i in range(batch_size):
                ## RE-ORIENT DIMENSIONS OF FEATURES TO CREATE PHI FROM SNGP PAPER
                phi_R1 = features_NR[i].unsqueeze(1)
                outer_product_RR = phi_R1 @ phi_R1.t()

                cov_inv_RR += outer_product_RR

        #Add identity at end according to formula
        cov_inv_RR += np.eye(rank)

        ## UPDATE COVARIANCE VARIABLE
        self.precision_mat = cov_inv_RR

    def get_mean_variance(self, X_test, covariance, noise=0.0, device='cpu'):
        features_NR = self.featurize(X_test).squeeze()
        N = features_NR.shape[0]

        mean_N = self.linear(features_NR)
        var_N = []
        for n in range(N):
            phi_R = features_NR[n]  # (R,)
            var = phi_R.T @ covariance @ phi_R  # scalar
            if noise > 0:
                var = var + noise**2
            var_N.append(var)

        var_N = torch.stack(var_N)  # (N,)

        mean_N = mean_N.squeeze()
        var_N = var_N.squeeze()
        return mean_N, var_N

    def invert_covariance(self, device='cpu'):
        covariance_RR = torch.inverse(self.precision_mat)
        return torch.tensor(covariance_RR, dtype=torch.float32).to(device)
    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.lengthscale_param)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.outputscale_param)
