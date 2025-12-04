import sys
import resource
import torch
import matplotlib.pyplot as plt

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


class FullGP:
    """
    Full GP regression with explicit matrix inverse.
    
    Posterior formulas:
        μ* = K_X*X @ K̂_XX^{-1} @ y
        Σ* = K_X*X* - K_X*X @ K̂_XX^{-1} @ K_XX*
        
    where K̂_XX = K_XX + σ²I
    """
    
    def __init__(self, lengthscale=1.0, outputscale=1.0, noise=0.1):
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


# =============================================================================
# CREATE DATASET
# =============================================================================

torch.manual_seed(40)

# True function
def true_function(x):
    return torch.sin(2 * x) - torch.cos(x)

for n in [20]:
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

    gp = FullGP(lengthscale=1.0, outputscale=1.0, noise=0.2)
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
    print(f"K̂_XX^{{-1}} shape (inverse): {gp.K_XX_noisy_inv.shape}")

    K_X_star_X = gp.kernel(X_test, X_train)
    print(f"K_X*X shape (test-train kernel): {K_X_star_X.shape}")

    K_X_star_X_star = gp.kernel(X_test, X_test)
    print(f"K_X*X* shape (test-test kernel): {K_X_star_X_star.shape}")

    print(f"\nPosterior mean shape: {mean.shape}")
    print(f"Posterior covariance shape: {cov.shape}")
    print(f"Posterior variance shape: {var.shape}")

    # Check eigenvalues of covariance
    eigenvalues = torch.linalg.eigvalsh(cov)
    print(f"\nCovariance eigenvalues: min={eigenvalues.min().item():.6f}, max={eigenvalues.max().item():.6f}")
    print(f"Negative eigenvalues: {(eigenvalues < 0).sum().item()} (numerical noise)")


    # =============================================================================
    # PLOT 1: GP REGRESSION
    # =============================================================================

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    ax1 = axes
    ax1.fill_between(
        X_test.squeeze().numpy(),
        (mean - 2*std).numpy(),
        (mean + 2*std).numpy(),
        alpha=0.3, color='blue', label='95% confidence (±2σ)'
    )
    ax1.plot(X_test.squeeze().numpy(), mean.numpy(), 'b-', linewidth=2, label='Posterior mean μ*')
    ax1.plot(X_test.squeeze().numpy(), y_true.numpy(), 'k--', linewidth=1, label='True function')
    ax1.scatter(X_train.squeeze().numpy(), y_train.numpy(), c='red', s=100, zorder=5, label='Training data')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('GP Regression: Full Closed Form', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig('full_gp_regression.png', dpi=600)
    plt.show()


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
        X_test.squeeze().numpy(),
        (mean - 2*std).numpy(),
        (mean + 2*std).numpy(),
        alpha=0.2, color='blue', label='95% CI'
    )
    for i in range(n_samples):
        plt.plot(X_test.squeeze().numpy(), f_samples[:, i].numpy(), alpha=0.7, linewidth=1)
    plt.plot(X_test.squeeze().numpy(), mean.numpy(), 'b-', linewidth=2, label='Mean')
    plt.scatter(X_train.squeeze().numpy(), y_train.numpy(), c='red', s=100, zorder=5, label='Training data')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Samples from Full Posterior f* ~ N(μ*, Σ*)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    plt.savefig('full_gp_samples.png', dpi=150)
    plt.show()

    print("\nKey observations:")
    print("- Samples are CORRELATED (smooth, not jagged)")
    print("- They pass through/near training points")
    print("- They diverge far from training data (uncertainty increases)")