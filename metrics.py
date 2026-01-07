import argparse
import glob
import os
import numpy as np
from statistics import mean, median
from scipy.linalg import logm, svd
from scipy.spatial.transform import Rotation as sciR

def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t - gt_t), ord=2) / 10

def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(np.matmul(gt_R, np.linalg.inv(pr_R))) - 1
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

def compute_ece_translation(all_preds_t, all_gts_t, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for translation uncertainty.
    all_preds_t : list of np.arrays, each of shape [K, 3] (MC samples for one item)
    all_gts_t   : list of np.arrays, each of shape [3] (ground-truth translation)
    n_bins      : number of bins for uncertainty partitioning
    """
    sigma_list = []
    error_list = []

    for preds_t, gt_t in zip(all_preds_t, all_gts_t):
        mu = np.mean(preds_t, axis=0)
        sigma = np.mean(np.std(preds_t, axis=0))  # average std across 3 dims
        error = np.linalg.norm(mu - gt_t)         # L2 translation error (mm)
        sigma_list.append(sigma)
        error_list.append(error)

    sigma_list = np.array(sigma_list)
    error_list = np.array(error_list)

    # Bin edges between min and max predicted std
    bin_edges = np.linspace(np.min(sigma_list), np.max(sigma_list), n_bins + 1)
    ece = 0.0
    N = len(sigma_list)

    for i in range(n_bins):
        # indices of samples within this bin
        mask = (sigma_list >= bin_edges[i]) & (sigma_list < bin_edges[i + 1])
        n_b = np.sum(mask)
        if n_b == 0:
            continue

        mean_sigma = np.mean(sigma_list[mask])
        mean_error = np.mean(error_list[mask])
        ece += (n_b / N) * np.abs(mean_error - mean_sigma)

    return ece
def compute_ece_translation_perdim(all_preds_t, all_gts_t, n_bins=10):
    """
    Regression ECE for translation: per-dimension calibration.
    Binning by σ quantiles. Compares mean |error| to mean σ in each bin.
    Returns macro-average over x,y,z and also per-dim values.
    """
    N = len(all_preds_t)
    preds_mu = np.stack([p.mean(axis=0) for p in all_preds_t])   # [N,3]
    preds_std = np.stack([p.std(axis=0)  for p in all_preds_t])  # [N,3]
    gts = np.stack(all_gts_t)                                    # [N,3]
    abs_err = np.abs(preds_mu - gts)                             # [N,3]

    ece_dims = []
    for d in range(3):
        sigma_d = preds_std[:, d]
        err_d   = abs_err[:, d]

        # Quantile bins to avoid empty bins
        q = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(sigma_d, q)
        # Ensure strictly increasing (handle ties)
        edges = np.unique(edges)
        if len(edges) < 2:
            # no spread at all => cannot calibrate
            ece_dims.append(float(np.mean(np.abs(err_d - sigma_d))))
            continue

        ece_d = 0.0
        Ntot = len(sigma_d)
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            # include right edge on last bin
            if i == len(edges)-2:
                mask = (sigma_d >= lo) & (sigma_d <= hi)
            else:
                mask = (sigma_d >= lo) & (sigma_d <  hi)
            nb = mask.sum()
            if nb == 0: 
                continue
            ece_d += (nb / Ntot) * abs(err_d[mask].mean() - sigma_d[mask].mean())
        ece_dims.append(float(ece_d))

    return float(np.mean(ece_dims)), ece_dims  # macro-average, [x,y,z]

def compute_sharpness_translation(all_preds_t):
    """
    Computes the average predicted uncertainty magnitude (Sharpness) for translation.
    all_preds_t: list of np.arrays, each [K, 3] (MC dropout samples for one item)
    Returns: scalar (mean sharpness in mm), and per-dimension values
    """
    # Standard deviation per sample, per axis
    stds = np.stack([np.std(p, axis=0) for p in all_preds_t])  # [N, 3]
    # Mean L2-norm of stds (vector sharpness)
    sharpness_vec = np.mean(np.linalg.norm(stds, axis=1))
    # Mean per-dimension sharpness (macro-style)
    sharpness_dims = np.mean(stds, axis=0)
    return float(sharpness_vec), sharpness_dims

def compute_sharpness_rotation(all_preds_R, all_gts_R):
    """
    Computes average predicted rotation uncertainty (radians).
    all_preds_R: list of np.arrays, each [K, 3x3]
    all_gts_R: list of 3x3 GT rotations
    """
    from scipy.spatial.transform import Rotation as sciR
    sharp_list = []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        errs = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        sharp_list.append(np.std(errs))
    return float(np.mean(sharp_list))
    
def compute_ece_rotation(all_preds_R, all_gts_R, n_bins=10):
    from scipy.spatial.transform import Rotation as sciR
    errs, sigmas = [], []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        eR = np.arccos(np.clip((np.trace(Rgt.T @ mean_R) - 1) / 2, -1, 1))
        errs.append(eR)
        sample_errs = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        sigmas.append(np.std(sample_errs))
    errs, sigmas = np.array(errs), np.array(sigmas)
    # binning
    edges = np.quantile(sigmas, np.linspace(0,1,n_bins+1))
    edges = np.unique(edges)
    ece, N = 0, len(sigmas)
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mask = (sigmas >= lo) & (sigmas < hi if i < len(edges)-2 else sigmas <= hi)
        nb = mask.sum()
        if nb == 0: continue
        ece += (nb/N) * abs(errs[mask].mean() - sigmas[mask].mean())
    return float(ece)

# This is oversimplified NLL for rotation, assuming Gaussian over geodesic angle errors.
def compute_nll_rotation(all_preds_R, all_gts_R):
    from scipy.spatial.transform import Rotation as sciR
    nlls = []
    for Rs, Rgt in zip(all_preds_R, all_gts_R):
        mean_R = mean_rotation_SVD(Rs)
        eR_mean = np.arccos(np.clip((np.trace(Rgt.T @ mean_R) - 1) / 2, -1, 1))
        eR_samples = [np.arccos(np.clip((np.trace(Rgt.T @ R) - 1) / 2, -1, 1)) for R in Rs]
        mu, sigma = np.mean(eR_samples), np.std(eR_samples) + 1e-8
        nll = 0.5 * (((eR_mean - mu)**2) / (sigma**2) + np.log(sigma**2) + np.log(2*np.pi))
        nlls.append(nll)
    return np.mean(nlls)


# ---- SO(3) Metrics Helper Functions ----

def geodesic_distance(Ra, Rb):
    return np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0))

def credible_region_radius(Rs, R_bar, alpha=0.95):
    distances = np.array([geodesic_distance(R, R_bar) for R in Rs])
    r_alpha = np.percentile(distances, alpha * 100)
    prop_in_region = np.mean(distances <= r_alpha)
    return r_alpha, prop_in_region

def eaad(Rs, R_bar):
    deviations = np.array([geodesic_distance(R_bar, R) for R in Rs])
    return deviations.mean()

# ======== UCS: Reliability + Score (Wursthorn et al., 2024) ========

def _pit_from_gaussian_1d(y, mu, sigma, eps=1e-9):
    """Probability integral transform (PIT) for 1D Gaussian:
       u = Phi((y - mu) / sigma). Returns scalar in (0,1)."""
    from math import erf, sqrt
    z = (y - mu) / max(sigma, eps)
    # standard normal CDF via erf
    u = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    # numerical safety
    return float(np.clip(u, 1e-9, 1.0 - 1e-9))

def _reliability_and_ucs_from_pit(pit_values, p_grid=None):
    """
    Build reliability curve and UCS from a list/array of PIT values (u in (0,1)).
    UCS = 1 - (area_between_curves / 0.25), with area via trapezoid on |p_hat - p|.
    """
    if p_grid is None:
        p_grid = np.linspace(0.1, 1.0, 10)  # Δp = 0.1 as in the paper
    pit_values = np.asarray(pit_values, dtype=float)
    T = len(pit_values)
    if T == 0:
        return p_grid, np.zeros_like(p_grid), 0.0

    p_hat = []
    for p in p_grid:
        p_hat.append(float(np.mean(pit_values <= p)))
    p_hat = np.array(p_hat)

    # Area between observed and diagonal: A = ∫ |p_hat - p| dp  (approx)
    A = np.trapz(np.abs(p_hat - p_grid), p_grid)
    UCS = 1.0 - (A / 0.25)
    UCS = float(np.clip(UCS, 0.0, 1.0))
    return p_grid, p_hat, UCS

def compute_ucs_translation(all_preds_t, all_gts_t, p_grid=None):
    """
    UCS for translation, per-dimension (x,y,z) and macro-average.
    For each sample and each dim: build PIT using Gaussian(mu, sigma) vs GT dim.
    Returns: macro_ucs, [ucs_x, ucs_y, ucs_z]
    """
    # Collect PITs per dimension
    pits = [[], [], []]  # x,y,z

    for preds_t, gt_t in zip(all_preds_t, all_gts_t):
        mu = preds_t.mean(axis=0)          # [3]
        std = preds_t.std(axis=0) + 1e-9   # [3]
        for d in range(3):
            u = _pit_from_gaussian_1d(gt_t[d], mu[d], std[d])
            pits[d].append(u)

    ucs_dims = []
    for d in range(3):
        _, _, ucs_d = _reliability_and_ucs_from_pit(pits[d], p_grid=p_grid)
        ucs_dims.append(ucs_d)

    return float(np.mean(ucs_dims)), ucs_dims

def _best_symmetry_gt_for_rotation(R_mean, gt_R1, gt_R2):
    """Choose GT orientation (to handle 180° symmetry) that is closer to mean."""
    d1 = geodesic_distance(R_mean, gt_R1)
    d2 = geodesic_distance(R_mean, gt_R2)
    return gt_R1 if d1 <= d2 else gt_R2


def compute_ucs_rotation_geodesic(all_preds_R, all_gts_R, p_grid=None):
    """
    UCS for rotation using a single scalar: geodesic angle to the mean rotation.
    For each sample:
      - Compute mean rotation (SVD).
      - Pick the symmetric GT closer to mean (your 180° symmetry).
      - y  = geodesic_distance(mean_R, GT_best)              # scalar target
      - ai = geodesic_distance(mean_R, R_i) for each MC sample
      - Fit Gaussian N(mu, sigma^2) to {ai}; PIT with y; collect across dataset.
    Returns: ucs_angle (scalar in [0,1]).
    """
    eps = 1e-9
    pits = []
    for Rs, gt_R1 in zip(all_preds_R, all_gts_R):
        R_mean = mean_rotation_SVD(Rs)

        # handle your 180° symmetry
        gt_R2 = np.matrix.copy(gt_R1)
        gt_R2[:, :2] *= -1
        gt_R = _best_symmetry_gt_for_rotation(R_mean, gt_R1, gt_R2)

        # scalar target: angle between mean and (best) GT
        y = geodesic_distance(R_mean, gt_R)

        # predictive 1D distribution over angles around mean
        angles = [geodesic_distance(R_mean, R) for R in Rs]
        mu = float(np.mean(angles))
        sigma = float(np.std(angles) + eps)

        u = _pit_from_gaussian_1d(y, mu, sigma)
        pits.append(u)

    _, _, ucs_angle = _reliability_and_ucs_from_pit(pits, p_grid=p_grid)
    return float(ucs_angle)
