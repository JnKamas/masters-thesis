'''
I integrated here many metrics, but only some of them are use in the thesis
'''

import argparse
import glob
import os
import numpy as np
import math

from statistics import mean, median
from scipy.linalg import logm, svd
from scipy.spatial.transform import Rotation as sciR
from scipy.special import iv, erf

def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t - gt_t), ord=2) / 10 # convert mm to cm

def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(gt_R @ pr_R.T) - 1
    numerator = np.clip(numerator, -2, 2)
    return np.arccos(numerator / 2)

def calculate_eGD(gt_R, pr_R):
    argument = logm(np.matmul(gt_R, np.transpose(pr_R)))
    numerator = np.linalg.norm(argument, ord='fro')
    return numerator / (2 ** .5)

def read_transform_file(file):
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                    [float(P[1]), float(P[5]), float(P[9])],
                    [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t
        
def mean_rotation_SVD(Rs):
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R_mean = np.dot(U, Vt)
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = np.dot(U, Vt)
    return R_mean


def crps_gaussian(mu, sigma, y):
    """
    mu, sigma, y: [B, D]
    """
    sigma = np.clip(sigma, 1e-6, None)
    z = (y - mu) / sigma

    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(z / np.sqrt(2)))

    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

    return np.mean(crps)


def crps_translation(mu_t, sigma_t, t_gt):
    """
    mu_t: [B, 3]
    sigma_t: [B, 3]
    t_gt: [B, 3]
    """
    return crps_gaussian(mu_t, sigma_t, t_gt)


def crps_rotation(R_samples, R_gt):
    """
    CRPS on SO(3) using geodesic distances.

    R_samples: [T, B, 3, 3]
    R_gt: [B, 3, 3]
    """
    # symmetry 
    S = sciR.from_euler('z', np.pi).as_matrix()
    
    T, B = R_samples.shape[:2]

    crps_vals = []

    for b in range(B):
        # compute geodesic angles to GT → [T]
        thetas = np.array([
            min(
                np.arccos(np.clip(
                    (np.trace(R_samples[t, b].T @ R_gt[b]) - 1) / 2,
                    -1.0, 1.0
                )),
                np.arccos(np.clip(
                    (np.trace(R_samples[t, b].T @ (R_gt[b] @ S)) - 1) / 2,
                    -1.0, 1.0
                ))
            )
            for t in range(T)
        ])

        # CRPS (sample-based, y = 0)
        term1 = np.mean(np.abs(thetas))
        term2 = 0.5 * np.mean(np.abs(thetas[:, None] - thetas[None, :]))

        crps_vals.append(term1 - term2)

    return float(np.mean(crps_vals))

def compute_sharpness_translation(all_preds_t):
    """
    Computes the average predicted uncertainty magnitude (Sharpness) for translation.
    all_preds_t: list of np.arrays, each [K, 3] (MC dropout samples for one item)
    Returns: scalar (mean sharpness in mm), and per-dimension values
    """
    # Standard deviation per sample, per axis
    stds = np.stack([np.std(p, axis=0) for p in all_preds_t])  # [N, 3]
    # Mean L2-norm of stds (vector sharpness)
    sharpness_vec = np.mean(np.linalg.norm(stds, axis=1)) / 10 # convert mm to cm
    # Mean per-dimension sharpness (macro-style)
    sharpness_dims = np.mean(stds, axis=0)
    return float(sharpness_vec), sharpness_dims

def compute_sharpness_rotation(all_preds_R, all_gts_R=None):
    """
    Computes average predicted rotation uncertainty (radians).
    all_preds_R: list of np.arrays, each [K, 3x3]
    all_gts_R: obsolete. TODO remove from flows that call this
    """
    sharp_list = []
    for Rs in all_preds_R:
        mean_R = mean_rotation_SVD(Rs)
        errs = [np.arccos(np.clip((np.trace(mean_R.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        sharp_list.append(np.std(errs))
    return float(np.mean(sharp_list))
    


# ChatGPT says this: We use a simplified isotropic approximation of the normalization constant.
# Still requires more reserch, but we are very close.
def matrix_fisher_nll(R_pred, R_gt, kappa, eps=1e-8):
    """
    Matrix–Fisher negative log-likelihood on SO(3).
    Simplification as Isotropic (kappa1=kappa2=kappa3) for stable normalization constant.
    Can be extended as anisotropic with more complex C(kappa) if needed. 
    We use isotropic for epistepic and anisotropic for aleatoric, but this is a design choice that can be revisited.

    Args:
        R_pred : (3,3) predicted rotation matrix
        R_gt   : (3,3) ground-truth rotation matrix
        kappa  : (3,) concentration parameters (must be >= 0)

    Returns:
        nll : float
    """

    # enforce isotropy
    k = float(np.mean(kappa))

    # rotation error
    R_err = R_pred.T @ R_gt

    # isotropic alignment term
    align = k * np.trace(R_err)

    # approximate isotropic normalization constant
    log_c = np.log(np.sinh(k) / (k + eps) + eps)

    # NLL
    return -align + log_c


# ---- SO(3) Metrics Helper Functions ----

def credible_region_radius(Rs, R_bar, alpha=0.95):
    distances = np.array([geodesic_distance(R, R_bar) for R in Rs])
    r_alpha = np.percentile(distances, alpha * 100)
    prop_in_region = np.mean(distances <= r_alpha)
    return r_alpha, prop_in_region


def correlation_translation(t_pred_mean, t_gt, t_samples):
    error = np.linalg.norm(t_pred_mean - t_gt, axis=1)

    std = np.std(t_samples, axis=0)
    uncertainty = np.linalg.norm(std, axis=1)

    return pearson_corr(error, uncertainty)

def geodesic_distance(Ra, Rb):
    return np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))


def correlation_rotation(R_mean, R_gt, R_samples):
    B = R_mean.shape[0]
    T = R_samples.shape[0]

    S = sciR.from_euler('z', np.pi).as_matrix()

    error = np.array([
        min(
            geodesic_distance(R_mean[i], R_gt[i]),
            geodesic_distance(R_mean[i], R_gt[i] @ S)
        )
        for i in range(B)
    ])

    angles = []
    for t in range(T):
        angles.append([
            geodesic_distance(R_samples[t, i], R_mean[i])
            for i in range(B)
        ])

    angles = np.array(angles)           # [T, B]
    uncertainty = np.std(angles, axis=0)

    return pearson_corr(error, uncertainty)

def pearson_corr(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))

    return num / (den + 1e-8)