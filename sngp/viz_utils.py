import matplotlib.pyplot as plt # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from tqdm import tqdm
import math

def display_data(dataloader):
    X = []
    y = []
    for x_batch, y_batch in dataloader:
        X.append(x_batch)
        y.append(y_batch)
    
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    
    X = X.numpy()
    y = y.numpy()

    fig, axs = plt.subplots(figsize=(4, 3), sharex=True, sharey='row')
    axs.scatter(X[:,0][y==0], X[:,1][y==0], color='#1F77B4')
    axs.scatter(X[:,0][y==1], X[:,1][y==1], color='#D62728')
    axs.set_xlim([-1.4, 2.4])
    axs.set_ylim([-0.9, 1.4])
    axs.set_xlabel(r'$x_1$')
    axs.set_ylabel(r'$x_2$')

    fig.tight_layout()
    plt.show()

def display_data_splits(train_loader, val_loader, test_loader, save_Name = None):
    def get_data(loader):
        X = []
        y = []
        for x_batch, y_batch in loader:
            X.append(x_batch)
            y.append(y_batch)
        X = torch.cat(X, dim=0).numpy()
        y = torch.cat(y, dim=0).numpy()
        return X, y

    X_train, y_train = get_data(train_loader)
    X_val, y_val = get_data(val_loader)
    X_test, y_test = get_data(test_loader)

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

    x_limits = (-1.4, 2.4)
    y_limits = (-0.9, 1.4)
    xlabel = r'$x_1$'
    ylabel = r'$x_2$'

    axs[0].scatter(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], color='#1F77B4', label='Class 0', s=10)
    axs[0].scatter(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], color='#D62728', label='Class 1', s=10)
    axs[0].set_title('Training Data')
    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].legend()

    axs[1].scatter(X_val[:, 0][y_val==0], X_val[:, 1][y_val==0], color='#1F77B4', label='Class 0', s=10)
    axs[1].scatter(X_val[:, 0][y_val==1], X_val[:, 1][y_val==1], color='#D62728', label='Class 1', s=10)
    axs[1].set_title('Validation Data')
    axs[1].set_xlim(x_limits)
    axs[1].set_ylim(y_limits)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].legend()

    axs[2].scatter(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], color='#1F77B4', label='Class 0', s=10)
    axs[2].scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], color='#D62728', label='Class 1', s=10)
    axs[2].set_title('Test Data')
    axs[2].set_xlim(x_limits)
    axs[2].set_ylim(y_limits)
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel(ylabel)
    axs[2].legend()

    fig.tight_layout()
    if save_Name == None:
        plt.show()
    else:
        plt.savefig(save_Name)
        plt.close()

def plot_loss(info, save_Name = None):
    epochs = info['epochs']
    loss = info['tr']['loss']
    val_loss = info['va']['loss']
    print(info)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    if save_Name == None:
        plt.show()
    else:
        plt.savefig(save_Name)

def plot_data_colored_by_labels(x_N2, y_N, msize=5, alpha=1.0):
    if y_N is None:
        plt.plot(x_N2[:,0], x_N2[:,1],
            color='#333333', marker='d', linestyle='', markersize=msize,
            mew=2, alpha=0.4)
    else:
        plt.plot(x_N2[y_N==0,0], x_N2[y_N==0,1],
            color='r', marker='x', linestyle='', markersize=msize, alpha=alpha,
            mew=2, label='y=0')
        plt.plot(x_N2[y_N==1,0], x_N2[y_N==1,1],
            color='b', marker='+', linestyle='', markersize=msize, alpha=alpha,
            mew=2, label='y=1')
    ax = plt.gca()
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlim([-2.05, 2.05])
    ax.set_ylim([-2.05, 2.05])
    ax.set_xlabel('x_1')
    yticklabels = ax.get_yticklabels()
    if len(yticklabels) > 0:
        ax.set_ylabel('x_2')
    ax.set_aspect('equal')

## Taken from L3D homework #2 [ Thank you Dr. Hughes :) ]
def plot_probas_over_dense_grid(
        model, x_N2, y_N,
        do_show_colorbar=True,
        x1_ticks=np.asarray([-2, -1, 0, 1, 2]),
        x2_ticks=np.asarray([-2, -1, 0, 1, 2]),
        c_levels=np.linspace(0, 1, 21),
        c_ticks=np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        x1_grid=np.linspace(-3, 3, 100),
        x2_grid=np.linspace(-3, 3, 100),
        x1_lims=(-2.05, 2.05),
        x2_lims=(-2.05, 2.05)):
    cur_ax = plt.gca()
    G = x1_grid.size
    H = x2_grid.size
    
    x1_GH, x2_GH = np.meshgrid(x1_grid, x2_grid)
    
    x_M2 = np.hstack([
        x1_GH.flatten()[:,np.newaxis],
        x2_GH.flatten()[:,np.newaxis]]).astype(np.float32)
        
    # Predict proba for each point in the flattened grid
    with torch.no_grad():
        yproba1_M__pt = model.predict_proba(torch.from_numpy(x_M2))[:,1]
        yproba1_M = yproba1_M__pt.detach().cpu().numpy()
    
    # Reshape the M probas into the GxH 2D field
    yproba1_GH = np.reshape(yproba1_M, x1_GH.shape)
    
    cmap = plt.cm.RdYlBu
    my_contourf_h = plt.contourf(
    	x1_GH, x2_GH, yproba1_GH, levels=c_levels, 
        vmin=0, vmax=1.0, cmap=cmap, alpha=0.5)
    plt.xticks(x1_ticks, x1_ticks)
    plt.yticks(x2_ticks, x2_ticks)
    
    if do_show_colorbar:
        left, bottom, width, height = plt.gca().get_position().bounds
        cax = plt.gcf().add_axes([left+1.03*width, bottom, 0.03, height])
        plt.colorbar(my_contourf_h, orientation='vertical',
            cax=cax, ticks=c_ticks)
        plt.sca(cur_ax)
    plot_data_colored_by_labels(x_N2, y_N)

    #plt.legend(bbox_to_anchor=(1.0, 0.5));
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    plt.gca().set_aspect(1.0)
    plt.gca().set_xlim(x1_lims)
    plt.gca().set_ylim(x2_lims)
    plt.show()

def compute_probabilities_gp(model, loader, preds_filename='preds.pt', compute_covariance=True, device='cpu', num_samples=100):    
    xx, yy = np.meshgrid(np.arange(-4, 4, 0.05),
                         np.arange(-4, 4, 0.05))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    if compute_covariance:
        model.reinitialize_precision()
        model.update_precision_from_loader(loader, device=device)
    ##
    covariance = model.invert_covariance()
    
    preds_list = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        proba = model.predict_proba(grid_points, covariance, num_samples=num_samples)
        preds_list.append(proba.cpu())
    
    preds = torch.cat(preds_list, dim=0)
    preds = preds.cpu()
    torch.save(preds, preds_filename)

def plot_preds(preds_filename, loader, figure_filename, show_fig=False):
    X_numpy, y_numpy = src.utils.get_data_from_loader(loader)
    xx, yy = np.meshgrid(np.arange(-4, 4, 0.05),
                         np.arange(-4, 4, 0.05))
    preds = torch.load(preds_filename, map_location=torch.device('cpu'))
    class1_preds = preds[:, 1].numpy()
    predictions = class1_preds.reshape(xx.shape)
    
    colors = ['#1F77B4', '#5799C7', '#8FBBDA', '#C7DDED', '#FFFFFF',
              '#F5C9CA', '#EB9394', '#E15D5E', '#D62728']
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    contourf_obj1 = ax1.contourf(xx, yy, predictions, levels=np.linspace(0, 1, 256),
                                 alpha=0.9, cmap=cmap, antialiased=True)
    ax1.scatter(X_numpy[:, 0][y_numpy == 0], X_numpy[:, 1][y_numpy == 0],
                color='#1F77B4', label='Class 0', s=2)
    ax1.scatter(X_numpy[:, 0][y_numpy == 1], X_numpy[:, 1][y_numpy == 1],
                color='#D62728', label='Class 1', s=2)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_title("Contour With Scatter")

    ax1.legend()
    
    c_ticks = np.linspace(0, 1, 11)
    cbar1 = fig.colorbar(contourf_obj1, ax=ax1, ticks=c_ticks, orientation='vertical')
    
    contourf_obj2 = ax2.contourf(xx, yy, predictions, levels=np.linspace(0, 1, 256),
                                 alpha=0.9, cmap=cmap, antialiased=True)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.set_title("Contour without Scatter")
    
    cbar2 = fig.colorbar(contourf_obj2, ax=ax2, ticks=c_ticks, orientation='vertical')
    fig.tight_layout()

    # Save BEFORE show/close
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)



def plot_probabilities_gp(model, train_loader, compute_covariance=True, device='cpu', num_samples=100, preds_path='preds.pt', filename='plot_probabilities_gp.png'):
    X_numpy, y_numpy = src.utils.get_data_from_loader(train_loader)
    
    xx, yy = np.meshgrid(np.arange(-4, 4, 0.05),
                         np.arange(-4, 4, 0.05))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    if compute_covariance:
        model.reinitialize_precision()
        model.update_precision_from_loader(train_loader, device=device)
    covariance = model.invert_covariance()
    
    preds_list = []
    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(grid_points, covariance, num_samples=num_samples)
        preds_list.append(proba.cpu())
    
    preds = torch.cat(preds_list, dim=0)
    class1_preds = preds[:, 1].numpy()
    predictions = class1_preds.reshape(xx.shape)
    torch.save(preds, preds_path)
    
    colors = ['#1F77B4', '#5799C7', '#8FBBDA', '#C7DDED', '#FFFFFF',
              '#F5C9CA', '#EB9394', '#E15D5E', '#D62728']
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contourf_obj1 = ax.contourf(xx, yy, predictions, levels=np.linspace(0, 1, 256),
                                 alpha=0.9, cmap=cmap, antialiased=True)
    ax.scatter(X_numpy[:, 0][y_numpy == 0], X_numpy[:, 1][y_numpy == 0],
                color='#1F77B4', label='Class 0', s=2)
    ax.scatter(X_numpy[:, 0][y_numpy == 1], X_numpy[:, 1][y_numpy == 1],
                color='#D62728', label='Class 1', s=2)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title("Contour With Scatter")

    ax.legend()
    
    c_ticks = np.linspace(0, 1, 11)
    cbar1 = fig.colorbar(contourf_obj1, ax=ax, ticks=c_ticks, orientation='vertical')
    
    fig.tight_layout()

    # Save BEFORE show/close
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# def plot_thresholded_predictions(preds_file, train_loader, threshold=0.7, device='cpu'):
#     preds = torch.load(preds_file, map_location=device)
    
#     num_grid_points = preds.shape[0]
#     grid_size = int(np.sqrt(num_grid_points))
#     if grid_size * grid_size != num_grid_points:
#         raise ValueError("The number of prediction points is not a perfect square.")
    
#     x_lin = np.linspace(-4, 4, grid_size)
#     y_lin = np.linspace(-4, 4, grid_size)
#     xx, yy = np.meshgrid(x_lin, y_lin)
    
#     class1_preds = preds[:, 1].numpy()
#     max_confidence = torch.max(preds, dim=1)[0].numpy()

#     class1_preds_thresholded = class1_preds.copy()
#     class1_preds_thresholded[max_confidence < threshold] = np.nan
    
#     predictions_thresholded = class1_preds_thresholded.reshape(xx.shape)
    
#     X_numpy, y_numpy = src.utils.get_data_from_loader(train_loader)
    
#     colors = ['#1F77B4', '#5799C7', '#8FBBDA', '#C7DDED', '#FFFFFF', '#F5C9CA', '#EB9394', '#E15D5E', '#D62728']
#     cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)
    
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     contour1 = ax.contourf(xx, yy, predictions_thresholded, 
#         levels=np.linspace(0, 1, 256), alpha=0.9, cmap=cmap, antialiased=True)
#     ax.scatter(X_numpy[:, 0][y_numpy == 0], X_numpy[:, 1][y_numpy == 0],
#         color='#1F77B4', label='Class 0', s=2)
#     ax.scatter(X_numpy[:, 0][y_numpy == 1], X_numpy[:, 1][y_numpy == 1],
#         color='#D62728', label='Class 1', s=2)
#     ax.set_xlim([-4, 4])
#     ax.set_ylim([-4, 4])
#     ax.set_xlabel(r'$x_1$')
#     ax.set_ylabel(r'$x_2$')
#     ax.set_title(f"Throshold Contour w/ Data Points (Threshold: {threshold:.2f})")
#     ax.legend()
#     c_ticks = np.linspace(0, 1, 11)
#     fig.colorbar(contour1, ax=ax, ticks=c_ticks, orientation='vertical')
    
#     plt.sca(ax)

#     fig.tight_layout()

#     fig.savefig(f"thresholded_predictions", dpi=300, bbox_inches='tight')

#     plt.show()
#     plt.close(fig)

def plot_thresholded_predictions(
    preds_file,
    train_loader,
    thresholds,
    device='cpu',
    filename='thresholded_predictions.png',
):
    """
    thresholds: list of float thresholds, e.g. [0.5, 0.6, 0.7, 0.8]
    We'll plot 4 subplots per row.
    """

    # ----- Load predictions over grid -----
    preds = torch.load(preds_file, map_location=device)  # shape (M, K)
    num_grid_points = preds.shape[0]
    grid_size = int(np.sqrt(num_grid_points))

    if grid_size * grid_size != num_grid_points:
        raise ValueError("The number of prediction points is not a perfect square.")

    # Make the grid coordinates in the SAME order you used to generate preds
    x_lin = np.linspace(-4, 4, grid_size)
    y_lin = np.linspace(-4, 4, grid_size)
    xx, yy = np.meshgrid(x_lin, y_lin)

    # Extract per-class probs and max confidence
    class1_preds = preds[:, 1].cpu().numpy()  # (M,)
    max_confidence = torch.max(preds, dim=1)[0].cpu().numpy()  # (M,)

    # Training data for scatter overlay
    X_numpy, y_numpy = src.utils.get_data_from_loader(train_loader)

    # Colormap
    colors = [
        '#1F77B4', '#5799C7', '#8FBBDA', '#C7DDED', '#FFFFFF',
        '#F5C9CA', '#EB9394', '#E15D5E', '#D62728'
    ]
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)

    # ----- Figure layout -----
    n_thresh = len(thresholds)
    ncols = 4
    nrows = math.ceil(n_thresh / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    # We'll reuse these settings for axes limits / labels
    xlim = (-4, 4)
    ylim = (-4, 4)
    c_ticks = np.linspace(0, 1, 11)
    levels = np.linspace(0, 1, 256)

    for idx, thr in enumerate(thresholds):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        # Threshold mask
        mask = class1_preds.copy()
        mask[max_confidence < thr] = np.nan  # hide low-confidence regions
        mask_2d = mask.reshape(xx.shape)

        # Contourf of class-1 probability, masked by confidence
        contour = ax.contourf(
            xx, yy, mask_2d,
            levels=levels,
            alpha=0.9,
            cmap=cmap,
            antialiased=True,
        )

        # Overlay training scatter
        ax.scatter(
            X_numpy[:, 0][y_numpy == 0],
            X_numpy[:, 1][y_numpy == 0],
            color='#1F77B4',
            label='Class 0',
            s=4,
        )
        ax.scatter(
            X_numpy[:, 0][y_numpy == 1],
            X_numpy[:, 1][y_numpy == 1],
            color='#D62728',
            label='Class 1',
            s=4,
        )

        # Style
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(f"Threshold {thr:.2f}")
        ax.legend(fontsize=8, loc='lower right')

        # Add a colorbar for THIS subplot (right next to it)
        # Slightly smaller colorbar to avoid blowing up layout.
        cbar = fig.colorbar(
            contour,
            ax=ax,
            ticks=c_ticks,
            orientation='vertical',
            fraction=0.046,    # shrink colorbar box size
            pad=0.04           # gap between plot & bar
        )
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots (if thresholds not multiple of 4)
    for empty_idx in range(n_thresh, nrows * ncols):
        row = empty_idx // ncols
        col = empty_idx % ncols
        axes[row][col].axis('off')

    fig.tight_layout()

    # Save BEFORE show/close
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return filename

def plot_probabilities(model, loader, device='cpu', filename='plot_probabilities.png'):
    # Get the data for plotting (assumed to be numpy arrays)
    X_numpy, y_numpy = src.utils.get_data_from_loader(loader)
    
    # Create a grid for predictions
    xx, yy = np.meshgrid(np.arange(-10, 10, 0.05),
                         np.arange(-10, 10, 0.05))
    
    # Move the model to the specified device
    model.to(device)
    model.eval()
    
    # Create the grid points tensor and move it to the device
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    # Compute predictions on the grid
    with torch.no_grad():
        preds = model.predict_proba(grid_points)
        print(preds.shape)
        # Select the probability for class 1 and reshape to match the grid
        class1_preds = preds[:, 1]
        predictions = class1_preds.reshape(xx.shape)
        # Convert predictions to CPU and then to a numpy array for plotting
        predictions = predictions.cpu().numpy()
    
    # Define a custom colormap (if needed)
    colors = ['#1F77B4', '#5799C7', '#8FBBDA', '#C7DDED', '#FFFFFF',
              '#F5C9CA', '#EB9394', '#E15D5E', '#D62728']
    cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_numpy[:, 0][y_numpy == 0], X_numpy[:, 1][y_numpy == 0],
               color='#1F77B4', label='Class 0', s=6)
    ax.scatter(X_numpy[:, 0][y_numpy == 1], X_numpy[:, 1][y_numpy == 1],
               color='#D62728', label='Class 1', s=6)
    levels = np.linspace(0, 1, 256)
    contourf_obj = ax.contourf(xx, yy, predictions, levels=levels, alpha=0.5, cmap=cmap, antialiased=True)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.legend()

    c_ticks = np.linspace(0, 1, 11)
    left, bottom, width, height = ax.get_position().bounds
    cax = plt.gcf().add_axes([left + 1.2 * width, bottom, 0.03, height])
    boundaries = np.linspace(0, 1, 257)
    plt.colorbar(contourf_obj, orientation='vertical', cax=cax, ticks=c_ticks, boundaries=boundaries)
    plt.sca(ax)
    fig.tight_layout()

    # Save BEFORE show/close
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)


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

def save_sngp_fig(sngp_model, X_test, y_test, covariance, dataset, fig_path, seed=42):
    np.random.seed(seed)
    
    # ---- Determine extended range ----
    x_test_1d = X_test.squeeze(-1)
    x_min = x_test_1d.min().item()
    x_max = x_test_1d.max().item()
    x_range = x_max - x_min
    
    # Extend by 10% on each side
    x_extended_min = x_min - 0.2*x_range
    x_extended_max = x_max + 0.2*x_range
    
    # ---- Sample 100 random points from test set for plotting ----
    n_plot_points = min(100, len(X_test))  # Don't sample more than available
    plot_indices = np.random.choice(len(X_test), n_plot_points, replace=False)
    X_plot = X_test[plot_indices]
    y_plot = y_test[plot_indices]
    
    # ---- Generate additional noisy test points in extended regions ----
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
    mean, var = sngp_model.get_mean_variance(X_test_extended, covariance, noise=0.1, device='cpu')
    mean = mean.squeeze(-1)
    var = var.squeeze(-1)
    std = torch.sqrt(var.clamp(min=1e-6))
    
    # Sort everything for plotting
    x_extended_1d = X_test_extended.squeeze(-1)
    idx_extended = torch.argsort(x_extended_1d)
    x_extended_sorted = x_extended_1d[idx_extended]
    mean_sorted = mean[idx_extended]
    std_sorted = std[idx_extended]
    
    # ---- Generate clean data for true function (dense for smooth line) ----
    x_clean = np.linspace(x_extended_min, x_extended_max, 5_000)
    
    if dataset == "Sin":
        y_clean = getSin(x_clean, noise=0.0)
    elif dataset == "CrazySin":
        y_clean = getCrazySin(x_clean, noise=0.0)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # ---- Plotting ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Confidence band
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
    ax.set_title('SNGP Regression', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_extended_sorted.min().item(), x_extended_sorted.max().item())
    ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.close()