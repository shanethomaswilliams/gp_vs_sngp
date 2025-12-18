import sys
import resource
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
torch.set_default_dtype(torch.float32) 

# =============================================================================
# FULL EXPLICIT GP REGRESSION - NO TRICKS
# =============================================================================


class RBFKernel:
    def __init__(self, lengthscale=1.0, outputscale=1.0):
        self.lengthscale = lengthscale
        self.outputscale = outputscale
    
    def __call__(self, X1, X2):
        dist_sq = torch.cdist(X1, X2, p=2).pow(2)
        return self.outputscale**2 * torch.exp(-dist_sq / (2 * self.lengthscale**2))


class RBFKernelLearnable(nn.Module):
    """RBF kernel with learnable hyperparameters."""
    
    def __init__(self, lengthscale=1.0, outputscale=1.0):
        super().__init__()
        # Store as log-parameters for positivity constraint
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale).log())
        self.log_outputscale = nn.Parameter(torch.tensor(outputscale).log())
    
    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()
    
    @property
    def outputscale(self):
        return self.log_outputscale.exp()
    
    def forward(self, X1, X2):
        dist_sq = torch.cdist(X1, X2, p=2).pow(2)
        return self.outputscale**2 * torch.exp(-dist_sq / (2 * self.lengthscale**2))



class FullGP(nn.Module):
    """
    Full GP regression with explicit matrix inverse.
    
    Posterior formulas:
        μ* = K_X*X @ K̂_XX^{-1} @ y
        Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        
    where K̂_XX = K_XX + σ²I
    """
    
    def __init__(self, lengthscale=1.0, outputscale=1.0, noise=0.1, learn_hyperparams=False):
        if learn_hyperparams:
            self.kernel = RBFKernelLearnable(lengthscale, outputscale)
        else:
            self.kernel = RBFKernel(lengthscale, outputscale)
        self.noise = noise
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        N = len(X_train)
        
        # K_XX: kernel matrix between training points, shape (N, N)
        K_XX = self.kernel(X_train, X_train)
        
        # K̂_XX = K_XX + σ²I: noisy kernel matrix
        K_XX_noisy = K_XX + self.noise**2 * torch.eye(N)
        
        # Explicit inverse: K̂_XX^{-1}
        self.K_XX_noisy_inv = torch.linalg.inv(K_XX_noisy)

        
    def predict(self, X_test):
        """
        Full closed-form prediction.
        
        μ* = K_X*X @ K̂_XX^{-1} @ y
        Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        """
        # K_X*X: kernel between test and train, shape (M, N)
        K_X_star_X = self.kernel(X_test, self.X_train)
        
        # K_XX*: kernel between train and test, shape (N, M)
        K_X_X_star = self.kernel(self.X_train, X_test)
        
        # K_X*X*: kernel between test and test, shape (M, M)
        K_X_star_X_star = self.kernel(X_test, X_test)
        
        # =================================================================
        # POSTERIOR MEAN
        # μ* = K_X*X @ K̂_XX^{-1} @ y
        # =================================================================
        mean = K_X_star_X @ self.K_XX_noisy_inv @ self.y_train
        
        # =================================================================
        # POSTERIOR COVARIANCE (full matrix!)
        # Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        # =================================================================
        cov = K_X_star_X_star - K_X_star_X @ self.K_XX_noisy_inv @ K_X_X_star
        
        # Variance is diagonal of covariance
        var = torch.diag(cov)
        
        return mean, var, cov

class FullGPCholesky(nn.Module):
    """
    Full GP regression using Cholesky decomposition.
    
    Posterior formulas (mathematically equivalent to explicit inverse):
        μ* = K_X*X @ K̂_XX^{-1} @ y
        Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        
    where K̂_XX = K_XX + σ²I
    
    Implementation via Cholesky: K̂_XX = L L^T
        α = K̂_XX^{-1} @ y  (solved via L L^T α = y)
        μ* = K_X*X @ α
        v = L^{-1} @ K_XX*  (solved via L v = K_XX*)
        Σ* = K_X*X* - v^T @ v
    """
    
    def __init__(self, lengthscale=1.0, outputscale=1.0, noise=0.1, learn_hyperparams=False):
        super().__init__()
        if learn_hyperparams:
            self.kernel = RBFKernelLearnable(lengthscale, outputscale)
        else:
            self.kernel = RBFKernel(lengthscale, outputscale)
        self.log_noise = nn.Parameter(torch.tensor(noise).log())

    @property
    def noise(self):
        return self.log_noise.exp()
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        N = len(X_train)
        
        # K_XX: kernel matrix between training points, shape (N, N)
        K_XX = self.kernel(X_train, X_train)
        
        # K̂_XX = K_XX + σ²I: noisy kernel matrix
        jitter = 1e-3  # Increase to 1e-3 if this still fails
    
        # K̂_XX = K_XX + σ²I + jitter*I
        K_XX_noisy = K_XX + (self.noise**2 + jitter) * torch.eye(N, dtype=K_XX.dtype, device=K_XX.device)
        del K_XX
        
        # Cholesky decomposition: K̂_XX = L L^T
        # L is lower triangular, shape (N, N)
        self.L = torch.linalg.cholesky(K_XX_noisy)
        del K_XX_noisy  # Free memory immediately
        
        # Precompute α = K̂_XX^{-1} @ y by solving L L^T α = y
        # This is EXACTLY: K_XX_noisy_inv @ y_train
        self.alpha = torch.cholesky_solve(y_train.unsqueeze(1), self.L).squeeze()
        
    def predict(self, X_test):
        """
        Full closed-form prediction using Cholesky decomposition.
        
        Mathematically IDENTICAL to:
            μ* = K_X*X @ K̂_XX^{-1} @ y
            Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        """
        # K_X*X: kernel between test and train, shape (M, N)
        K_X_star_X = self.kernel(X_test, self.X_train)
        
        # K_XX*: kernel between train and test, shape (N, M)
        K_X_X_star = self.kernel(self.X_train, X_test)
        
        # K_X*X*: kernel between test and test, shape (M, M)
        K_X_star_X_star = self.kernel(X_test, X_test)
        
        # =================================================================
        # POSTERIOR MEAN
        # μ* = K_X*X @ α  (where α = K̂_XX^{-1} @ y, precomputed in fit)
        # EXACTLY EQUIVALENT TO: K_X*X @ K̂_XX^{-1} @ y
        # =================================================================
        mean = K_X_star_X @ self.alpha
        
        # =================================================================
        # POSTERIOR COVARIANCE (full matrix!)
        # Σ* = K_X*X* - v^T @ v  (where v = L^{-1} @ K_XX*)
        # EXACTLY EQUIVALENT TO: K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        # =================================================================
        # Solve L v = K_XX* for v using forward substitution
        # v has shape (N, M)
        v = torch.linalg.solve_triangular(self.L, K_X_X_star, upper=False)
        
        # Compute covariance: Σ* = K_X*X* - v^T @ v
        # Mathematical proof that this is equivalent:
        #   v^T @ v = (L^{-1} @ K_XX*)^T @ (L^{-1} @ K_XX*)
        #           = K_XX*^T @ L^{-T} @ L^{-1} @ K_XX*
        #           = K_XX*^T @ (L L^T)^{-1} @ K_XX*
        #           = K_X*X @ K̂_XX^{-1} @ K_XX*  ✓
        cov = K_X_star_X_star - v.T @ v
        
        # Variance is diagonal of covariance
        var = torch.diag(cov)
        
        return mean, var, cov
    
    def marginal_log_likelihood(self):
        """
        Compute marginal log-likelihood (evidence):
            log p(y|X,θ) = -1/2 y^T K̂^{-1} y - 1/2 log|K̂| - n/2 log(2π)
        
        Using Cholesky decomposition:
            - y^T K̂^{-1} y = y^T α (already computed)
            - log|K̂| = 2 * sum(log(diag(L)))
        """
        if self.L is None or self.alpha is None:
            raise RuntimeError("Must call fit() before computing log likelihood")
        
        n = len(self.y_train)
        
        # Term 1: -1/2 y^T K̂^{-1} y = -1/2 y^T α
        data_fit = -0.5 * (self.y_train @ self.alpha)
        
        # Term 2: -1/2 log|K̂| = -sum(log(diag(L)))
        # Since K̂ = L L^T, we have |K̂| = |L|^2, so log|K̂| = 2*log|L|
        # For triangular matrix: log|L| = sum(log(diag(L)))
        log_det = -torch.sum(torch.log(torch.diag(self.L)))
        
        # Term 3: -n/2 log(2π)
        const = -0.5 * n * torch.log(torch.tensor(2 * torch.pi))
        
        return data_fit + log_det + const
    
    def negative_marginal_log_likelihood(self):
        """Negative MLL for minimization."""
        return -self.marginal_log_likelihood()
        



def train_gp(gp, X_train, y_train, n_iterations=100, lr=0.1, verbose=True):
    """Train GP with gradient clipping and hyperparameter constraints"""
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
    losses = []
    
    pbar = tqdm(range(n_iterations), desc="Training GP", disable=not verbose)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for i in pbar:
        try:
            optimizer.zero_grad()
            
            # Fit GP with current hyperparameters
            gp.fit(X_train, y_train)
            
            # Compute negative marginal log-likelihood
            loss = gp.negative_marginal_log_likelihood()
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"\nNon-finite loss at iter {i}, stopping")
                break
            
            # Backprop
            loss.backward()
            
            # CRITICAL: Clip gradients
            torch.nn.utils.clip_grad_norm_(gp.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # CRITICAL: Constrain hyperparameters
            with torch.no_grad():
                # Lengthscale: [0.01, 50] (relaxed for high-D)
                gp.kernel.log_lengthscale.clamp_(
                    torch.log(torch.tensor(0.01)),
                    torch.log(torch.tensor(100.0))
                )
                # Outputscale: [0.01, 10]
                gp.kernel.log_outputscale.clamp_(
                    torch.log(torch.tensor(0.01)),
                    torch.log(torch.tensor(100.0))
                )
                # Noise: [0.001, 1.0]
                gp.log_noise.clamp_(
                    torch.log(torch.tensor(0.001)),
                    torch.log(torch.tensor(1.0))
                )
            
            losses.append(loss.item())
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 200:
                    print(f"\nEarly stop at iter {i}")
                    break
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lengthscale': f'{gp.kernel.lengthscale.item():.3f}',
                'outputscale': f'{gp.kernel.outputscale.item():.3f}',
                'noise': f'{gp.noise.item():.3f}'
            })
            
        except torch._C._LinAlgError:
            print(f"\nCholesky failed at iter {i}, stopping")
            break
    
    return losses

def evaluate_gp(gp, X_test, y_test):
    """
    Evaluate GP performance on test set.
    
    Args:
        gp: Fitted GP model (FullGP or FullGPCholesky)
        X_test: Test inputs, shape (M, D)
        y_test: True test outputs, shape (M,)
    
    Returns:
        dict with keys:
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'test_ll': Average test log-likelihood (per point)
            - 'total_test_ll': Total test log-likelihood
    """
    # Get predictions
    mean, var, cov = gp.predict(X_test)
    
    # =========================================================================
    # MSE and RMSE (pointwise metrics)
    # =========================================================================
    squared_errors = (y_test - mean) ** 2
    mse = squared_errors.mean().item()
    rmse = torch.sqrt(squared_errors.mean()).item()

    # =========================================================================
    # Test Log-Likelihood (accounts for uncertainty)
    # =========================================================================
    # log p(y_test | X_test, D_train) where y_test ~ N(μ*, Σ*)
    # 
    # log p(y) = -1/2 * (y - μ)^T Σ^{-1} (y - μ) - 1/2 log|Σ| - n/2 log(2π)
    # =========================================================================
    
    n_test = len(y_test)
    residual = y_test - mean  # shape (M,)
    
    # For numerical stability, use Cholesky decomposition of Σ*
    # Add small jitter for numerical stability
    jitter = 1e-6
    cov_stable = cov + jitter * torch.eye(n_test)
    
    try:
        L_test = torch.linalg.cholesky(cov_stable)
        
        # Solve L_test @ α = residual for α
        alpha = torch.linalg.solve_triangular(L_test, residual.unsqueeze(1), upper=False).squeeze()
        
        # Mahalanobis distance: (y - μ)^T Σ^{-1} (y - μ) = α^T α
        mahalanobis = alpha @ alpha
        
        # Log determinant: log|Σ| = 2 * sum(log(diag(L)))
        log_det = 2 * torch.sum(torch.log(torch.diag(L_test)))
        
    except RuntimeError as e:
        print(f"Warning: Cholesky failed, using eigendecomposition fallback")
        # Fallback: eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_stable)
        eigenvalues_clamped = eigenvalues.clamp(min=1e-8)
        
        # Σ^{-1} = V @ diag(1/λ) @ V^T
        cov_inv = eigenvectors @ torch.diag(1.0 / eigenvalues_clamped) @ eigenvectors.T
        mahalanobis = residual @ cov_inv @ residual
        log_det = torch.sum(torch.log(eigenvalues_clamped))
    
    # Compute log-likelihood
    log_2pi = torch.log(torch.tensor(2 * torch.pi))
    total_test_ll = -0.5 * (mahalanobis + log_det + n_test * log_2pi)
    avg_test_ll = total_test_ll / n_test
    
    return {
        'mse': mse,
        'rmse': rmse,
        'test_ll': avg_test_ll.item(),  # per-point log-likelihood
        'total_test_ll': total_test_ll.item()
    }

def getSin(X_data, noise=0.0):
    """Generate clean (noise=0) or noisy function values"""
    if noise > 0:
        return np.sin(2 * X_data) - np.cos(X_data) + noise * np.random.randn(len(X_data))
    else:
        return np.sin(2 * X_data) - np.cos(X_data)

def getCrazySin(X_data, noise=0.0):
    """Generate clean (noise=0) or noisy function values"""
    if noise > 0:
        return np.sin(10 * X_data) - np.cos(7 * X_data) + noise * np.random.randn(len(X_data))
    else:
        return np.sin(10 * X_data) - np.cos(7 * X_data)

def save_fig(gp, X_test, y_test, dataset, fig_path, seed=42):
    np.random.seed(seed)

    # ---- Sample 100 random points from test set for plotting ----
    n_plot_points = min(100, len(X_test))  # Don't sample more than available
    plot_indices = np.random.choice(len(X_test), n_plot_points, replace=False)
    X_plot = X_test[plot_indices]
    y_plot = y_test[plot_indices]
    
    # ---- Determine extended range ----
    x_test_1d = X_test.squeeze(-1)
    x_min = x_test_1d.min().item()
    x_max = x_test_1d.max().item()
    x_range = x_max - x_min
    
    # Extend by 10% on each side
    x_extended_min = x_min - 0.2*x_range
    x_extended_max = x_max + 0.2*x_range
    
    # ---- Generate additional noisy test points in extended regions ----
    # Sample ~50 points in each extended region
    n_extra_left = 50
    n_extra_right = 50
    
    x_extra_left = np.random.uniform(x_extended_min, x_min, n_extra_left)
    x_extra_right = np.random.uniform(x_max, x_extended_max, n_extra_right)
    x_extra = np.concatenate([x_extra_left, x_extra_right])
    
    # Generate noisy y values for these extra points
    if dataset == "Sin":
        y_extra = getSin(x_extra, noise=0.1)
    elif dataset == "CrazySin":
        y_extra = getCrazySin(x_extra, noise=0.1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Convert to torch and combine with original test set
    X_extra = torch.tensor(x_extra, dtype=torch.float32).unsqueeze(-1)
    X_test_extended = torch.cat([X_test, X_extra], dim=0)
    
    # ---- Get predictions on extended test set ----
    mean, var, cov = gp.predict(X_test_extended)
    var = var + gp.noise**2
    std = torch.sqrt(var.clamp(min=1e-6))
    
    # Sort everything for plotting
    x_extended_1d = X_test_extended.squeeze(-1)
    idx_extended = torch.argsort(x_extended_1d)
    x_extended_sorted = x_extended_1d[idx_extended]
    mean_sorted = mean[idx_extended]
    std_sorted = std[idx_extended]
    
    # ---- Generate clean data for true function (dense for smooth line) ----
    x_clean = np.linspace(x_extended_min, x_extended_max, 500)
    
    if dataset == "Sin":
        y_clean = getSin(x_clean, noise=0.0)
    elif dataset == "CrazySin":
        y_clean = getCrazySin(x_clean, noise=0.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # ---- Plotting ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Confidence band (now extends to cover clean function range)
    ax.fill_between(
        x_extended_sorted.detach().numpy(),
        (mean_sorted - 2*std_sorted).detach().numpy(),
        (mean_sorted + 2*std_sorted).detach().numpy(),
        alpha=0.3, color='blue', label='95% confidence (±2σ)'
    )
    
    # Posterior mean
    ax.plot(x_extended_sorted.detach().numpy(), mean_sorted.detach().numpy(), 
            'b-', linewidth=2, label='Posterior mean μ*')
    
    # True function (dotted line)
    ax.plot(x_clean, y_clean, 
            'k--', linewidth=1, label='True function')
    
    # Sampled test points (red circles)
    ax.scatter(X_plot.squeeze().detach().numpy(), 
               y_plot.squeeze().detach().numpy(), 
               c='red', s=30, zorder=5, alpha=0.6, label='Test data (sample)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('GP Regression: Full Closed Form', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_extended_sorted.min().item(), x_extended_sorted.max().item())
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.close()

# =============================================================================
# CREATE DATASET
# =============================================================================
'''
torch.manual_seed(40)

# True function
def true_function(x):
    return torch.sin(2 * x) - torch.cos(x)

for n in [10, 50, 500, 1_000, 1_500]:
    N_train = n
    X_train = torch.rand(N_train, 1) * 6 - 3
    X_train = X_train.sort(dim=0).values
    y_train = true_function(X_train).squeeze() + 0.2 * torch.randn(N_train)

    # Test data: dense grid
    X_test = torch.linspace(-4, 4, 200).unsqueeze(1)
    y_true = true_function(X_test).squeeze()


    memory_gb = (N_train ** 2) * 4 / (1024 ** 3)
    print(f"K_XX alone: {memory_gb:.1f} GB") 

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f"Virtual memory limit: soft={soft}, hard={hard}")

    # Try to increase (may require sudo)
    # -1 means unlimited
    try:
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
        print("Increased to hard limit")
    except:
        print("Cannot increase limit")

    # Check available RAM
    import os
    if sys.platform == 'darwin':  # macOS
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        print(f"Total RAM: {mem_bytes / (1024**3):.1f} GB")

    # =============================================================================
    # FIT AND PREDICT
    # =============================================================================

    gp = FullGPCholesky(lengthscale=1.0, outputscale=1.0, noise=0.2, learn_hyperparams=True)
    losses = train_gp(gp, X_train, y_train, n_iterations=10_000, lr=0.01)
    gp.fit(X_train, y_train)
    
    mean, var, cov = gp.predict(X_test)
    std = torch.sqrt(var.clamp(min=1e-6))

    # =============================================================================
    # PRINT THE MATRICES
    # =============================================================================

    print("=" * 60)
    print("FULL GP REGRESSION - EXPLICIT MATRICES")
    print("=" * 60)

    print(f"\nTraining points: {N_train}")
    print(f"Test points: {len(X_test)}")

    print(f"\nK_XX shape (train-train kernel): {gp.kernel(X_train, X_train).shape}")
    # print(f"K̂_XX^{{-1}} shape (inverse): {gp.K_XX_noisy_inv.shape}")

    K_X_star_X = gp.kernel(X_test, X_train)
    # print(f"K_X*X shape (test-train kernel): {K_X_star_X.shape}")

    K_X_star_X_star = gp.kernel(X_test, X_test)
    # print(f"K_X*X* shape (test-test kernel): {K_X_star_X_star.shape}")

    print(f"\nPosterior mean shape: {mean.shape}")
    print(f"Posterior covariance shape: {cov.shape}")
    print(f"Posterior variance shape: {var.shape}")

    # Check eigenvalues of covariance
    eigenvalues = torch.linalg.eigvalsh(cov)
    print(f"\nCovariance eigenvalues: min={eigenvalues.min().item():.6f}, max={eigenvalues.max().item():.6f}")
    print(f"Negative eigenvalues: {(eigenvalues < 0).sum().item()} (numerical noise)")

    print("LEARNED HYPERPARAMETERS:")
    print(f"  - Lengthscale: {gp.kernel.lengthscale.item():.3f}")
    print(f"  - Outputscale: {gp.kernel.outputscale.item():.3f}")


    # =============================================================================
    # PLOT 1: GP REGRESSION
    # =============================================================================

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    ax1 = axes
    ax1.fill_between(
        X_test.squeeze().detach().numpy(),
        (mean - 2*std).detach().numpy(),
        (mean + 2*std).detach().numpy(),
        alpha=0.3, color='blue', label='95% confidence (±2σ)'
    )
    ax1.plot(X_test.squeeze().detach().numpy(), mean.detach().numpy(), 'b-', linewidth=2, label='Posterior mean μ*')
    ax1.plot(X_test.squeeze().detach().numpy(), y_true.detach().numpy(), 'k--', linewidth=1, label='True function')
    ax1.scatter(X_train.squeeze().detach().numpy(), y_train.detach().numpy(), c='red', s=100, zorder=5, label='Training data')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('GP Regression: Full Closed Form', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig('full_gp_regression.png', dpi=600)
    # plt.show()


    # =============================================================================
    # SAMPLE FROM FULL POSTERIOR (with eigendecomposition for numerical stability)
    # =============================================================================

    print("\n" + "=" * 60)
    print("SAMPLING FROM FULL POSTERIOR")
    print("=" * 60)

    # The covariance matrix can have tiny negative eigenvalues due to numerical error.
    # Fix: eigendecomposition, clamp negative eigenvalues to zero, reconstruct.

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    print(f"Raw eigenvalues range: [{eigenvalues.min().item():.2e}, {eigenvalues.max().item():.2e}]")

    # Clamp negative eigenvalues to small positive value
    eigenvalues_fixed = eigenvalues.clamp(min=1e-6)

    # Reconstruct: Σ = V @ diag(λ) @ V^T
    # For sampling: L = V @ diag(sqrt(λ))
    L_cov = eigenvectors @ torch.diag(torch.sqrt(eigenvalues_fixed))

    # Draw samples: f = μ + L @ ε, where ε ~ N(0, I)
    n_samples = 10
    eps = torch.randn(len(X_test), n_samples)
    f_samples = mean.unsqueeze(1) + L_cov @ eps

    # Plot samples
    plt.figure(figsize=(10, 6))
    plt.fill_between(
        X_test.squeeze().detach().numpy(),
        (mean - 2*std).detach().numpy(),
        (mean + 2*std).detach().numpy(),
        alpha=0.2, color='blue', label='95% CI'
    )
    for i in range(n_samples):
        plt.plot(X_test.squeeze().detach().numpy(), f_samples[:, i].detach().numpy(), alpha=0.7, linewidth=1)
    plt.plot(X_test.squeeze().detach().numpy(), mean.detach().numpy(), 'b-', linewidth=2, label='Mean')
    plt.scatter(X_train.squeeze().detach().numpy(), y_train.detach().numpy(), c='red', s=100, zorder=5, label='Training data')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Samples from Full Posterior f* ~ N(μ*, Σ*)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    plt.savefig('full_gp_samples.png', dpi=150)
    # plt.show()

    print("\nKey observations:")
    print("- Samples are CORRELATED (smooth, not jagged)")
    print("- They pass through/near training points")
    print("- They diverge far from training data (uncertainty increases)")
    '''