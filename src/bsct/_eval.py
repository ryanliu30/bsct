"""Evaluation: compute FSD and MAE metrics from inference results."""

import numpy as np

from bsct._infer import System

METRIC_KEYS = [
    "full_smoothness",
    "full_energy_mae",
    "full_forces_mae",
    "compress_smoothness",
    "compress_energy_mae",
    "compress_forces_mae",
    "stretch_smoothness",
    "stretch_energy_mae",
    "stretch_forces_mae",
]


def _max_derivative(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the maximum absolute first derivative of y with respect to x.

    Parameters
    ----------
    x : np.ndarray
        Monotonically increasing coordinate array.
    y : np.ndarray
        Value array of the same length as ``x``.

    Returns
    -------
    float
        Maximum of ``|Δy / Δx|`` over all consecutive pairs.

    """
    return np.max(np.abs(np.diff(y) / np.diff(x)))


def compute_system_metrics(system: System, eps: float = 1.0) -> dict:
    """
    Compute FSD and MAE metrics for a single bond-perturbation scan.

    The potential energy is reconstructed by integrating the projected forces
    along the displacement trajectory, making it applicable to non-conservative
    models. The Force Smoothness Discrepancy (FSD) is the maximum absolute
    derivative of the log-ratio of predicted to ground-truth squared force
    deviations from equilibrium, evaluated over the full range, the compression
    sub-range (bond length ≤ equilibrium), and the stretch sub-range
    (bond length ≥ equilibrium).

    Parameters
    ----------
    system : System
        Inference results for a single bond-perturbation scan.
    eps : float
        Clipping floor applied to squared force magnitudes before taking the
        log, preventing log(0). Units are (eV/Å)². Default is 1.0.

    Returns
    -------
    dict
        Dictionary with intermediate arrays (``potential``, ``gt_potential``,
        ``force_magnitude``, ``gt_force_magnitude``, ``log_ratios``,
        ``equi_point``) and scalar metrics (``full_smoothness``,
        ``full_energy_mae``, ``full_forces_mae``, ``compress_smoothness``,
        ``compress_energy_mae``, ``compress_forces_mae``, ``stretch_smoothness``,
        ``stretch_energy_mae``, ``stretch_forces_mae``).

    """
    disp = system.positions[1:] - system.positions[:-1]
    avg_forces = 0.5 * (system.calc_forces[1:] + system.calc_forces[:-1])
    forces_proj = np.sum(avg_forces * disp, axis=(1, 2))
    potential = np.append(0, -np.cumsum(forces_proj))
    equi_point = np.argmin(system.gt_energy)
    potential = potential - potential[equi_point]
    gt_potential = system.gt_energy - np.min(system.gt_energy)

    equilibrium_forces = system.gt_forces[equi_point, None]
    force_diff = system.calc_forces - equilibrium_forces
    gt_force_diff = system.gt_forces - equilibrium_forces
    force_magnitude = np.sum(force_diff**2, axis=(1, 2))
    gt_force_magnitude = np.sum(gt_force_diff**2, axis=(1, 2))
    log_ratios = np.log(np.clip(force_magnitude, eps**2, None)) - np.log(
        np.clip(gt_force_magnitude, eps**2, None)
    )

    smoothness = _max_derivative(system.lincoords, log_ratios)
    compress_smoothness = _max_derivative(
        system.lincoords[: equi_point + 1], log_ratios[: equi_point + 1]
    )
    stretch_smoothness = _max_derivative(
        system.lincoords[equi_point:], log_ratios[equi_point:]
    )

    energy_mae = np.mean(np.abs(system.gt_energy - system.calc_energy))
    compress_energy_mae = np.mean(
        np.abs(
            system.gt_energy[: equi_point + 1] - system.calc_energy[: equi_point + 1]
        )
    )
    stretch_energy_mae = np.mean(
        np.abs(system.gt_energy[equi_point:] - system.calc_energy[equi_point:])
    )

    forces_mae = np.mean(np.abs(system.gt_forces - system.calc_forces))
    compress_forces_mae = np.mean(
        np.abs(
            system.gt_forces[: equi_point + 1] - system.calc_forces[: equi_point + 1]
        )
    )
    stretch_forces_mae = np.mean(
        np.abs(system.gt_forces[equi_point:] - system.calc_forces[equi_point:])
    )

    return {
        "potential": potential,
        "gt_potential": gt_potential,
        "force_magnitude": force_magnitude,
        "gt_force_magnitude": gt_force_magnitude,
        "log_ratios": log_ratios,
        "equi_point": equi_point,
        "full_smoothness": float(smoothness),
        "full_energy_mae": float(energy_mae),
        "full_forces_mae": float(forces_mae),
        "compress_smoothness": float(compress_smoothness),
        "compress_energy_mae": float(compress_energy_mae),
        "compress_forces_mae": float(compress_forces_mae),
        "stretch_smoothness": float(stretch_smoothness),
        "stretch_energy_mae": float(stretch_energy_mae),
        "stretch_forces_mae": float(stretch_forces_mae),
    }
