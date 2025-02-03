# https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA
import torch
import torch.nn.functional as F


def gaussian_kernel(x, y, kernel_bandwidth=1.0):
    """
    Computes the Gaussian kernel between x and y.
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    pairwise_sq_dists = torch.sum((x - y) ** 2, dim=2)
    kernel = torch.exp(-pairwise_sq_dists / (2 * kernel_bandwidth ** 2))
    return kernel


def mmd_loss(x, y, kernel_bandwidth=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss between two samples x and y.
    """
    # Compute the kernels
    k_xx = gaussian_kernel(x, x, kernel_bandwidth)
    k_yy = gaussian_kernel(y, y, kernel_bandwidth)
    k_xy = gaussian_kernel(x, y, kernel_bandwidth)

    # Calculate MMD
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd




def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*4)
    return loss


def wasserstein_distance(u_values, v_values):
    # Sort and accumulate the differences
    u_values = torch.sort(u_values)[0]
    v_values = torch.sort(v_values)[0]
    return torch.mean(torch.abs(u_values - v_values))


class MKMMDLoss(torch.nn.Module):
    def __init__(self, kernels=None):
        """
        Initialize MK-MMD Loss module.

        Args:
            kernels (list of callable): List of kernel functions to compute the MMD.
        """
        super(MKMMDLoss, self).__init__()
        if kernels is None:
            self.kernels = [self.gaussian_kernel()]
        else:
            self.kernels = kernels

    def gaussian_kernel(self, sigma_list=[1, 5, 10]):

        def _gaussian_kernel(x, y):
            K = torch.zeros(x.size(0), y.size(0)).to(x.device)
            for sigma in sigma_list:
                gamma = 1 / (2 * sigma ** 2)
                K += torch.exp(-gamma * (torch.cdist(x, y, p=2) ** 2))
            return K / len(sigma_list)

        return _gaussian_kernel

    def forward(self, features_s, features_t):
       
        mmd_loss = 0
        for kernel in self.kernels:
            # Compute kernel matrices
            K_ss = kernel(features_s, features_s)
            K_tt = kernel(features_t, features_t)
            K_st = kernel(features_s, features_t)

            # Compute MMD loss for this kernel
            loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
            mmd_loss += loss

        return mmd_loss


