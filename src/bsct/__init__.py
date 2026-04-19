"""Bond Smoothness Characterization Test (BSCT) evaluation library."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.calculator import Calculator
from tqdm import tqdm

from bsct._eval import METRIC_KEYS, compute_system_metrics
from bsct._infer import run_inference
from bsct._plot import make_plot

__version__ = "0.0.0"


def evaluate_bsct(
    calc: Calculator,
    data_path: str = "./bsct_spice",
    output_path: str = "./output/default",
    store_metrics: bool = False,
    store_plots: bool = False,
    store_xyz: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Evaluate a calculator on the BSCT benchmark.

    Runs the calculator on every bond-perturbation scan in ``data_path``,
    computes the Force Smoothness Discrepancy (FSD) and energy/force MAE
    metrics, and optionally saves plots, XYZ files, and per-system metrics.

    Parameters
    ----------
    calc : Calculator
        ASE calculator to evaluate.
    data_path : str
        Path to the BSCT dataset directory.
    output_path : str
        Root directory for all saved outputs.
    store_metrics : bool
        If ``True``, write per-system metrics to ``output_path/metrics.json``.
    store_plots : bool
        If ``True``, save per-system figures to ``output_path/plots/``.
    store_xyz : bool
        If ``True``, cache calculator XYZ files under
        ``output_path/calc_systems/`` and reuse them on subsequent calls.
    verbose : bool
        If ``True``, display progress bars during inference and evaluation.

    Returns
    -------
    dict[str, float]
        Averaged metric values across all systems with keys
        ``full_smoothness``, ``full_energy_mae``, ``full_forces_mae``,
        ``compress_smoothness``, ``compress_energy_mae``,
        ``compress_forces_mae``, ``stretch_smoothness``,
        ``stretch_energy_mae``, and ``stretch_forces_mae``.

    """
    systems = run_inference(calc, data_path, output_path, store_xyz, verbose)

    per_system = {
        "dataset": [],
        "system_id": [],
        **{k: [] for k in METRIC_KEYS},
    }

    for system in tqdm(systems, desc="Evaluating", disable=not verbose):
        result = compute_system_metrics(system)

        per_system["dataset"].append(system.dataset)
        per_system["system_id"].append(system.system_id)
        for k in METRIC_KEYS:
            per_system[k].append(result[k])

        if store_plots:
            fig = make_plot(system, result)
            plot_dir = os.path.join(output_path, "plots", system.dataset)
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, f"{system.system_id}.pdf"))
            plt.close(fig)

    if store_metrics:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "metrics.json"), "w") as f:
            json.dump(per_system, f)

    return {k: float(np.mean(per_system[k])) for k in METRIC_KEYS}
