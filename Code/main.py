"""
Finite-step error bounds for second-order (Strang) Trotterization of 1D Heisenberg chains
(XXX, XXZ, XYZ), with H = A + B split into even/odd bonds.

Compares the exact error ||exp(-iHt) - S_2(tau)^r|| (step size tau = t/r), the Childs et al. (PRX 2021) commutator
upper bound, and the leading-order estimate r * tau^3 * ||C_3||, in three sweeps:
error vs. step count r, vs. time t (both across all models), and vs. chain length n.
A computed lower bound, from a theorem currently in progress, will be added alongside these.

"Source:" comments name screenshots in ../Sources (Childs, Hahn and Chen, External Sources folders).
"""

import os
import numpy as np
from numpy import kron as tensor
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt

# Imaginary Unit
i = 1.0j

# Pauli Matrices
I = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -i], [i, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# Couplings (Jx, Jy, Jz) per model
MODELS = {"XXX (Isotropic: 1.0, 1.0, 1.0)": (1.0, 1.0, 1.0),
          "XXZ (Anisotropic: 1.0, 1.0, 1.5)": (1.0, 1.0, 1.5),
          "XYZ (Fully Anisotropic: 1.0, 1.2, 1.5)": (1.0, 1.2, 1.5)}

# Tensor Product Utility
def two_site_operator(n, j, op1, op2):
    # Construct a 2^n x 2^n operator acting as `op1` on site `j` and `op2` on site `j+1` and I elsewhere,
    # with sites numbered 1 to n (list positions j-1 and j)
    operators, operators[j-1], operators[j] = [I] * n, op1, op2
    # Compute the tensor product of a list of 2x2 matrices
    result = [1]
    for operator in operators:
        result = tensor(result, operator)
    return result

# Hamiltonian Builder (Even-Odd Bond Partition: H = A + B)
# Source: Childs 06 (Heisenberg H, XXX with no field here), Chen 02 (general Jx, Jy, Jz),
# Childs 07 (even-odd split; terms within A, or within B, commute).
def build_heisenberg_even_odd(n, Jx, Jy, Jz):
    """
    H = sum_{j=1}^{n-1} (Jx X_j X_{j+1} + Jy Y_j Y_{j+1} + Jz Z_j Z_{j+1}) on n spin-1/2 sites numbered 1 to n,
    open boundaries, Pauli matrices (no 1/4 factor). Returns Hermitian A (bonds (1,2), (3,4), ...) and
    B (bonds (2,3), (4,5), ...) with H = A + B; terms within A, or within B, commute.
    XXX: Jx = Jy = Jz.  XXZ: Jx = Jy != Jz.  XYZ: Jx, Jy, Jz all distinct.
    """
    A = np.zeros((2 ** n, 2 ** n), dtype=complex)
    B = np.zeros((2 ** n, 2 ** n), dtype=complex)

    for j in range(1, n):
        if j % 2 == 1:  # bond (j, j+1) starts on an odd site: A, as in Childs 07
            A += (Jx * two_site_operator(n, j, X, X) + Jy * two_site_operator(n, j, Y, Y) + Jz * two_site_operator(n, j, Z, Z))
        else:
            B += (Jx * two_site_operator(n, j, X, X) + Jy * two_site_operator(n, j, Y, Y) + Jz * two_site_operator(n, j, Z, Z))

    return A, B

# Commutator Invariants & Second-Order Trotter Engine
def compute_commutator_metrics(A, B):
    """
    Spectral norms (ord=2) for the Strang upper bound and leading-order estimate. Returns a dict:
      C_3_norm: ||C_3||, C_3 = -1/24 [A,[A,B]] + 1/12 [B,[B,A]] (leading 3rd-order BCH error operator)
      childs_upper_coeff: 1/12 ||[B,[B,A]]|| + 1/24 ||[A,[A,B]]|| (Childs et al. 2021)
      A_norm, B_norm, comm_AB_norm, comm_AAB_norm, comm_BBA_norm: norms of A, B, [A,B], [A,[A,B]], [B,[B,A]]
    """
    comm_AB = A @ B - B @ A
    comm_BA = B @ A - A @ B
    comm_AAB = A @ comm_AB - comm_AB @ A
    comm_BBA = B @ comm_BA - comm_BA @ B

    # Leading 3rd-order error operator in symmetric BCH expansion:
    # ||S_2(tau) - exp(-iH tau)|| ≈ tau^3 * ||C_3|| for small tau
    # Source: HLW 02 (C_3 has the same form as S_3; with A -> -i tau A and B -> -i tau B, t^3 S_3 becomes i tau^3 C_3),
    # HLW 01 (only odd powers of tau).
    C_3 = -(1.0 / 24.0) * comm_AAB + (1.0 / 12.0) * comm_BBA

    # childs_upper_coeff source: Childs 05 (Prop 10 with H_1 = A, the outer half-step operator, and H_2 = B)
    return {
        "childs_upper_coeff": (1.0 / 12.0) * norm(comm_BBA, 2) + (1.0 / 24.0) * norm(comm_AAB, 2),
        "C_3_norm": norm(C_3, 2),
        "A_norm": norm(A, 2),
        "B_norm": norm(B, 2),
        "comm_AB_norm": norm(comm_AB, 2),
        "comm_AAB_norm": norm(comm_AAB, 2),
        "comm_BBA_norm": norm(comm_BBA, 2),
    }

def evaluate_strang_step_and_bounds(A, B, U_exact, metrics, t, r):
    """Returns (exact_error, upper_bound, asymptotic_estimate) for S_2(tau)^r against U_exact = exp(-iHt)."""
    tau = t / r

    # Single Strang step: S_2(tau) = exp(-iA tau/2) exp(-iB tau/2) exp(-iB tau/2) exp(-iA tau/2)
    # Source: Childs 03 (Eq 11 with H_1 = -iA, H_2 = -iB),
    # Childs 04 (Childs' H_gamma are anti-Hermitian, so his e^{tH} is our e^{-iHt}), Childs 05 (Childs' t is one step, our tau)
    S_2_step = expm(-i * A * tau / 2.0) @ expm(-i * B * tau / 2.0) @ expm(-i * B * tau / 2.0) @ expm(-i * A * tau / 2.0)

    # Multi-step Strang evolution: S_2(tau)^r   Source: Childs 01
    U_trotter = np.linalg.matrix_power(S_2_step, r)

    # Exact spectral norm error: ||exp(-iHt) - S_2(tau)^r||_2
    exact_error = norm(U_exact - U_trotter, 2)

    # Childs et al. (2021) Commutator Upper Bound (uncapped; raw-error plots cap it at the trivial unitary bound 2.0)
    # Source: Childs 04 (r steps)
    upper_bound = r * tau ** 3 * metrics["childs_upper_coeff"]

    # Leading-order asymptotic approximation: r * tau^3 * ||C_3||_2
    # Source: Childs 02 (the upper bound is this term after the triangle inequality, so it sits above it),
    # Chen 01 (step errors partly cancel at larger t, so the exact error falls below this estimate)
    asymptotic_estimate = r * tau ** 3 * metrics["C_3_norm"]

    return exact_error, upper_bound, asymptotic_estimate

def normalize_or_cap(curves, scale, mask_threshold, normalize):
    """
    curves: 3xN array (upper_bounds, exact_errors, asymptotic_estimates). If normalize, divides every curve by
    scale = r * tau^3 and hides all three curves wherever the raw upper bound exceeds mask_threshold,
    i.e. outside the perturbative regime where r * tau^3 scaling cannot hold yet; otherwise caps the upper bound
    at the trivial unitary bound 2.0.
    """
    if normalize:
        # Masking source: Childs 04 (bound works well only for small t), Hahn 01, Hahn 02 (when a bound is trivial)
        outside = curves[0] > mask_threshold  # raw upper bound too large: non-perturbative regime
        curves = curves / scale
        curves[:, outside] = np.nan           # matplotlib leaves gaps at NaN
        return curves
    else:
        curves[0] = np.minimum(curves[0], 2.0)  # cap upper bound at trivial unitary bound. Source: Chen 03 (errors saturate at 2)
    return curves

# Plotting
def plot_bounds_figure(x, panels, xlabel, ylabel, title, save_path, logx, logy, xticks, normalized, mask_threshold):
    """
    Builds and saves a 1xN figure, one panel per (panel_title, (upper_bounds, exact_errors, asymptotic_estimates))
    in `panels`; panel_title may be None. logx / logy set log scale per axis; xticks may be None.
    normalized labels the asymptotic curve as ||C_3|| (curves divided by r * tau^3) instead of r * tau^3 * ||C_3||.
    mask_threshold: if given and normalized, a note below the panels states the masking criterion; None for no note.
    """
    asymp_formula = r"\|C_3\|" if normalized else r"r\tau^3 \, \|C_3\|"
    curve_styles = (("r--", "Childs et al. Upper Bound"), ("k-", "Exact Spectral Error"), ("b:", f"Asymptotic Leading Term (${asymp_formula}$)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(max(7.0, 5.0 * len(panels)), 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (panel_title, curves) in zip(axes, panels):
        for y, (fmt, label) in zip(curves, curve_styles):
            ax.plot(x, y, fmt, lw=2, label=label)
        ax.set_xscale("log" if logx else "linear")
        ax.set_yscale("log" if logy else "linear")
        if xticks is not None:
            ax.set_xticks(list(xticks))
        ax.set_title(panel_title or "", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    axes[0].set_ylabel(ylabel, fontsize=11)
    axes[0].legend(fontsize=9.5)
    fig.suptitle(title, fontsize=12)
    if normalized and mask_threshold is not None:
        footnote = f"Points with $r\\tau^3\\cdot$(Childs coefficient) $> {mask_threshold:g}$ omitted (non-perturbative regime)"
        fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=9.5, style="italic")
        fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave room at the bottom for the footnote
    else:
        fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# Experiments
def r_sweep(n, t, r_values, mask_threshold, normalize, save_path):
    """
    Error vs. step count r (over r_values, a NumPy array of ints) at fixed total evolution time t, one panel per model.
    Normalize divides all curves by r * tau^3 and hides points whose raw upper bound exceeds mask_threshold;
    otherwise plots raw error on log-log axes.
    """
    panels = []
    for label, (Jx, Jy, Jz) in MODELS.items():
        A, B = build_heisenberg_even_odd(n, Jx, Jy, Jz)
        H = A + B
        U_exact = expm(-i * H * t)
        metrics = compute_commutator_metrics(A, B)

        exact_errors, upper_bounds, asymptotic_estimates = [], [], []
        for r in r_values:
            exact_error, upper_bound, asymptotic_estimate = evaluate_strang_step_and_bounds(A, B, U_exact, metrics, t, r)
            exact_errors.append(exact_error)
            upper_bounds.append(upper_bound)
            asymptotic_estimates.append(asymptotic_estimate)

        curves = np.array([upper_bounds, exact_errors, asymptotic_estimates])
        panels.append((label, normalize_or_cap(curves, r_values * (t / r_values) ** 3, mask_threshold, normalize)))

    ylabel = "Normalized Error $\\|e^{-iHt} - S_2(\\tau)^r\\|_2 \\,/\\, (r\\tau^3)$" if normalize else "Spectral Norm Error $\\|e^{-iHt} - S_2(\\tau)^r\\|_2$"
    plot_bounds_figure(r_values, panels, "Number of Trotter Steps ($r$)", ylabel, f"Second-Order Trotter Bounds vs. Step Count (n={n}, t={t})", save_path, logx=True, logy=not normalize, xticks=None, normalized=normalize, mask_threshold=mask_threshold)

def t_sweep(n, r, t_values, mask_threshold, normalize, save_path):
    """
    Error vs. time t (over t_values, a NumPy array of floats) at fixed step count r, one panel per model. Normalize as in r_sweep.
    Source: Chen 01, Chen 03 (exact error tracks ||C_3|| at small t, then falls below it as step errors interfere).
    """
    panels = []
    for label, (Jx, Jy, Jz) in MODELS.items():
        A, B = build_heisenberg_even_odd(n, Jx, Jy, Jz)
        H = A + B
        metrics = compute_commutator_metrics(A, B)

        exact_errors, upper_bounds, asymptotic_estimates = [], [], []
        for t in t_values:
            U_exact = expm(-i * H * t)
            exact_error, upper_bound, asymptotic_estimate = evaluate_strang_step_and_bounds(A, B, U_exact, metrics, t, r)
            exact_errors.append(exact_error)
            upper_bounds.append(upper_bound)
            asymptotic_estimates.append(asymptotic_estimate)

        curves = np.array([upper_bounds, exact_errors, asymptotic_estimates])
        panels.append((label, normalize_or_cap(curves, r * (t_values / r) ** 3, mask_threshold, normalize)))

    ylabel = "Spectral Norm Error $/\\, (r\\tau^3)$" if normalize else "Spectral Norm Error"
    plot_bounds_figure(t_values, panels, "Evolution Time ($t$)", ylabel, f"Leading-Order Breakdown vs. Time (n={n}, r={r})", save_path, logx=True, logy=not normalize, xticks=None, normalized=normalize, mask_threshold=mask_threshold)

def system_size_scaling(n_list, Jx, Jy, Jz, t, r, mask_threshold, normalize, save_path):
    """
    Error vs. chain length n at fixed t and r (so fixed step size tau = t/r) for one model (single panel, linear axes).
    normalize divides by r * tau^3, so the upper-bound and asymptotic curves equal the
    Childs coefficient and ||C_3||, and hides points whose raw upper bound exceeds mask_threshold;
    otherwise plots raw error.
    Source: Childs 08 (bounds for this even-odd Heisenberg chain as system size grows).
    """
    tau = t / r
    exact_errors, upper_bounds, asymptotic_estimates = [], [], []

    for n in n_list:
        A, B = build_heisenberg_even_odd(n, Jx, Jy, Jz)
        H = A + B
        U_exact = expm(-i * H * t)
        metrics = compute_commutator_metrics(A, B)

        exact_error, upper_bound, asymptotic_estimate = evaluate_strang_step_and_bounds(A, B, U_exact, metrics, t, r)
        exact_errors.append(exact_error)
        upper_bounds.append(upper_bound)
        asymptotic_estimates.append(asymptotic_estimate)

    curves = normalize_or_cap(np.array([upper_bounds, exact_errors, asymptotic_estimates]), r * tau ** 3, mask_threshold, normalize)

    ylabel = "Spectral Norm Error $/\\, (r\\tau^3)$" if normalize else "Spectral Norm Error"
    plot_bounds_figure(n_list, [(None, curves)], "Number of Spins ($n$)", ylabel, f"System-Size Scaling at Fixed Step Size ($t={t}, r={r}, \\tau={tau:g}$)", save_path, logx=False, logy=False, xticks=n_list, normalized=normalize, mask_threshold=None)

if __name__ == "__main__":
    n = 10                     # chain length for r_sweep and t_sweep
    n_list = (4, 6, 8, 10, 12) # chain lengths for system_size_scaling
    Jx, Jy, Jz = 1.0, 1.2, 1.5 # XYZ couplings for system_size_scaling
    t = 1.0                    # fixed total time for r_sweep
    t_scaling = 0.05           # short total time for system_size_scaling, where the exact error matches the C_3 estimate
    r = 20                     # fixed step count (t_sweep, system_size_scaling)
    r_values = np.unique(np.logspace(0, 2.5, 25, dtype=int))  # log-spaced step counts from 1 to 316 (22 unique values)
    t_values = np.geomspace(0.05, 10, 40)   # 40 log-spaced times from 0.05 to 10 (through the breakdown and saturation at 2)
    mask_threshold = 1.0                    # normalized plots hide points with raw upper bound > this
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")  # Plots folder next to Code, wherever the script is run from

    r_sweep(n, t, r_values, mask_threshold, normalize=False, save_path=os.path.join(plots_dir, "error_bounds_vs_r.png"))
    t_sweep(n, r, t_values, mask_threshold, normalize=False, save_path=os.path.join(plots_dir, "error_bounds_vs_t.png"))
    system_size_scaling(n_list, Jx, Jy, Jz, t_scaling, r, mask_threshold, normalize=False, save_path=os.path.join(plots_dir, "error_bounds_n_scaling.png"))

    r_sweep(n, t, r_values, mask_threshold, normalize=True, save_path=os.path.join(plots_dir, "error_bounds_vs_r_normalized.png"))
    t_sweep(n, r, t_values, mask_threshold, normalize=True, save_path=os.path.join(plots_dir, "error_bounds_vs_t_normalized.png"))
    system_size_scaling(n_list, Jx, Jy, Jz, t_scaling, r, mask_threshold, normalize=True, save_path=os.path.join(plots_dir, "error_bounds_n_scaling_normalized.png"))
