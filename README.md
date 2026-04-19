# BSCT: Bond Smoothness Characterization Test

**BSCT** is a benchmark and evaluation library for measuring the physical smoothness of Machine Learning Interatomic Potential (MLIP) predictions. It accompanies the [paper](https://arxiv.org/abs/2602.04861):

> **From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide Machine Learning Interatomic Potential Architectures**
> Ryan Liu, Eric Qu, Tobias Kreiman, Samuel M. Blau, and Aditi S. Krishnapriyan
> California Institute of Technology, UC Berkeley, Lawrence Berkeley National Laboratory

---

## Overview

MLIPs can produce unphysical potential energy surfaces (PES) — with spurious discontinuities, artificial minima, or erratic forces — that standard energy and force regression metrics fail to detect. BSCT addresses this by probing the PES along controlled 1D bond perturbation trajectories, covering both near- and far-from-equilibrium regimes.

The core metric, the **Force Smoothness Deviation (FSD)**, measures the maximum rate of change of the log-ratio of predicted to ground-truth squared force deviations from equilibrium:

$$
    \text{FSD} = \max_{\alpha}\left\vert\frac{\mathrm d}{\mathrm d\alpha} \log\frac{\Vert \Delta \vec F_{\text{MLIP}}\Vert^2}{\Vert \Delta \vec F_{\text{DFT}}\Vert^2}\right\vert
$$

A lower FSD indicates a smoother, more physically reliable PES. Crucially, FSD correlates strongly with molecular dynamics (MD) stability while requiring only ~40 minutes on one GPU, compared to ~40 hours for equivalent MD simulations.

---

## Installation

```bash
pip install -e .
```

---

## Dataset

The **BSCT-SPICE dataset**, located at `bsct_spice/`, contains 485 molecules from the SPICE test set. Each molecule has 100 DFT single-point calculations along a bond perturbation trajectory spanning 0.5× to 2× the sum of the bonded atoms' covalent radii. All calculations use the same level of theory as SPICE (ωB97M-D3(BJ)/def2-TZVPPD via Psi4).

Bond types covered: C–C, C–N, C–O, C–P, C–S, N–N, N–O, N–P, and O–P.

---

## Usage

```python
from bsct import evaluate_bsct

calc = YourCalculator(...)

results = evaluate_bsct(
    calc=calc,
    data_path="./bsct_spice",        # path to the BSCT dataset
    output_path="./output/my_model", # root directory for all saved outputs
    store_metrics=True,              # save per-system metrics to metrics.json
    store_plots=True,                # save per-system PDF plots
    store_xyz=True,                  # cache calculator results for reuse
    verbose=True,                    # show progress bars
)

print(results)
# {
#   "full_smoothness":       ...,   # FSD, full bond range       (Å⁻¹, lower is better)
#   "full_energy_mae":       ...,   # Energy MAE, full range     (eV)
#   "full_forces_mae":       ...,   # Forces MAE, full range     (eV/Å)
#   "compress_smoothness":   ...,   # FSD, compression sub-range
#   "compress_energy_mae":   ...,
#   "compress_forces_mae":   ...,
#   "stretch_smoothness":    ...,   # FSD, stretch sub-range
#   "stretch_energy_mae":    ...,
#   "stretch_forces_mae":    ...,
# }
```

### Metrics

| Key | Description | Units |
|---|---|---|
| `full_smoothness` | FSD over the full bond scan | Å⁻¹ |
| `compress_smoothness` | FSD over the compression sub-range (bond ≤ equilibrium) | Å⁻¹ |
| `stretch_smoothness` | FSD over the stretch sub-range (bond ≥ equilibrium) | Å⁻¹ |
| `*_energy_mae` | Mean absolute error of calculator vs. DFT energies | eV |
| `*_forces_mae` | Mean absolute error of calculator vs. DFT forces | eV/Å |

All metrics are averaged across all systems in the dataset.

---

## Outputs

When `store_plots=True`, a PDF is saved for each system at `output_path/plots/<bond_type>/<system_id>.pdf`. Each plot contains two panels:
- **Top**: Integrated MLIP potential energy vs. the DFT reference along the bond scan.
- **Bottom**: Squared force deviations (‖ΔF‖²) on a symlog scale, with the log-ratio overlaid on a twin axis. A side legend reports FSD and MAE metrics for the full range, compression sub-range, and stretch sub-range.

When `store_metrics=True`, per-system metrics are written to `output_path/metrics.json`.

When `store_xyz=True`, calculator results are cached under `output_path/calc_systems/` and reused on subsequent calls to avoid redundant forward passes.

---

## Note on Inference Efficiency

This release prioritizes compatibility over throughput. For maximum performance, we recommend adapting the inference loop to use batched evaluation, as different MLIPs expose different batching interfaces and none is assumed here.

---

## Module Structure

```
bsct/
  __init__.py   # evaluate_bsct() — top-level entry point
  _infer.py     # run_inference() — runs an ASE calculator over all systems
  _eval.py      # compute_system_metrics() — computes FSD and MAE metrics
  _plot.py      # make_plot() — generates per-system diagnostic figures
```

---

## Citation

If you use BSCT in your work, please cite:

```bibtex
@article{liu2026evaluation,
    title={From Evaluation to Design: Using Potential Energy Surface Smoothness
           Metrics to Guide Machine Learning Interatomic Potential Architectures},
    author={Liu, Ryan and Qu, Eric and Kreiman, Tobias and Blau, Samuel M
            and Krishnapriyan, Aditi S},
    journal={arXiv preprint arXiv:2602.04861},
    year={2026}
}
```

---

## License

See `LICENSE` for details.