"""Plotting: generate per-system potential energy and force smoothness figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import Locator

from bsct._infer import System

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.style": "normal",
        "font.size": 8,
        "figure.dpi": 300,
        "mathtext.fontset": "dejavuserif",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.latex.preamble": "\\usepackage{amsmath} \\usepackage{amssymb}",
        "text.usetex": True,
        "xtick.direction": "in",
        "xtick.major.size": 3.0,
        "xtick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "xtick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.major.size": 3.0,
        "ytick.major.width": 0.5,
        "ytick.minor.size": 1.5,
        "ytick.minor.visible": True,
        "ytick.minor.width": 0.5,
        "ytick.right": True,
    }
)


class MinorSymLogLocator(Locator):
    """
    Minor tick locator for symlog-scaled axes.

    Dynamically finds minor tick positions from the current major tick
    positions, using 10 subdivisions in the linear region and 9 in the
    logarithmic region. Adapted from
    https://stackoverflow.com/questions/20470892.

    Parameters
    ----------
    linthresh : float
        Half-width of the linear region of the symlog scale.

    """

    def __init__(self, linthresh):
        self.linthresh = linthresh

    def __call__(self):
        """
        Return minor tick locations between the current major ticks.

        Returns
        -------
        np.ndarray
            Array of minor tick positions clipped to the axis view limits.

        """
        majorlocs = self.axis.get_majorticklocs()

        minorlocs = []

        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        """
        Not implemented for this locator type.

        Raises
        ------
        NotImplementedError
            Always raised; use ``__call__`` to obtain tick locations.

        """
        raise NotImplementedError(
            "Cannot get tick locations for a %s type." % type(self)
        )


def make_plot(system: System, result: dict) -> plt.Figure:
    """
    Generate a two-panel figure for a bond-perturbation scan.

    The upper panel shows the integrated MLIP potential energy and the DFT
    reference potential energy as a function of bond length. The lower panel
    shows the squared force deviations from equilibrium on a symlog scale,
    with the log-ratio overlaid on a twin axis. A side legend reports FSD
    and MAE metrics for the full range, compression sub-range, and stretch
    sub-range.

    Parameters
    ----------
    system : System
        Inference results providing ``lincoords`` for the x-axis.
    result : dict
        Output of :func:`~bsct._eval.compute_system_metrics`, providing
        ``potential``, ``gt_potential``, ``force_magnitude``,
        ``gt_force_magnitude``, ``log_ratios``, and all scalar metric values.

    Returns
    -------
    plt.Figure
        Matplotlib figure. The caller is responsible for saving and closing it.

    """
    potential = result["potential"]
    gt_potential = result["gt_potential"]
    force_magnitude = result["force_magnitude"]
    gt_force_magnitude = result["gt_force_magnitude"]
    log_ratios = result["log_ratios"]

    smoothness = result["full_smoothness"]
    energy_mae = result["full_energy_mae"]
    forces_mae = result["full_forces_mae"]
    compress_smoothness = result["compress_smoothness"]
    compress_energy_mae = result["compress_energy_mae"]
    compress_forces_mae = result["compress_forces_mae"]
    stretch_smoothness = result["stretch_smoothness"]
    stretch_energy_mae = result["stretch_energy_mae"]
    stretch_forces_mae = result["stretch_forces_mae"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax3 = ax2.twinx()

    (line1,) = ax1.plot(system.lincoords, potential, color="C0")
    (line2,) = ax1.plot(system.lincoords, gt_potential, color="C0", linestyle="--")
    ax1.set_ylim(-1, 11)
    ax1.set_ylabel("Potential Energy (eV)")
    ax1.set_title("Potential Energy Surface of Bond Perturbation")
    ax1.legend(
        [line1, line2],
        ["MLIP Potential Energy", "DFT Potential Energy"],
        loc="upper right",
    )

    (line1,) = ax2.plot(system.lincoords, force_magnitude, color="C0")
    (line2,) = ax2.plot(
        system.lincoords, gt_force_magnitude, color="C0", linestyle="--"
    )
    (line3,) = ax3.plot(system.lincoords, log_ratios, color="C1")
    ax3.hlines(
        0,
        system.lincoords[0],
        system.lincoords[-1],
        color="C1",
        linestyle="--",
    )
    ax2.set_ylim(0, 1e4)
    ax2.set_yscale("symlog", linthresh=10, linscale=1)
    ax2.tick_params(axis="y", colors="C0", which="both")
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=10))
    ax2.set_xlabel("Bond Length (\u00c5)")
    ax2.set_ylabel(
        r"$\Vert \Delta \vec{F} \Vert^2\ (\mathrm{eV/\AA})^2$",
        color="C0",
    )
    ax2.set_title(r"$\Vert \Delta \vec{F} \Vert^2$" " Due to Bond Perturbation")
    ax3.set_ylim(-10, 10)
    ax3.set_ylabel(r"Log Ratio of $\Vert \Delta \vec{F} \Vert^2$", color="C1")
    ax3.tick_params(axis="y", colors="C1", which="both")
    ax3.legend(
        [line1, line2, line3],
        [
            r"$\Vert \Delta \vec{F}_{\mathrm{MLIP}} \Vert^2$",
            r"$\Vert \Delta \vec{F}_{\mathrm{DFT}} \Vert^2$",
            r"$\log\!\left(\Vert \Delta \vec{F}_{\mathrm{MLIP}} \Vert^2"
            r"/ \Vert \Delta \vec{F}_{\mathrm{DFT}} \Vert^2\right)$",
        ],
        loc="upper right",
        ncol=3,
    )
    fig.tight_layout()

    ax4 = fig.add_axes([0.95, 0.1, 0.2, 0.8])
    ax4.axis("off")
    ax4.legend(
        [ax4.plot([], [], " ")[0] for _ in range(13)],
        [
            r"\textbf{Metrics}",
            r"\textbf{Full Range}",
            f"FSD: {smoothness:.1f} " r"$\mathrm{\AA}^{-1}$",
            r"$E$" f" MAE: {energy_mae:.3g} eV",
            r"$F$" f" MAE: {forces_mae:.3g} eV/\u00c5",
            r"\textbf{Compress}",
            f"FSD: {compress_smoothness:.1f} " r"$\mathrm{\AA}^{-1}$",
            r"$E$" f" MAE: {compress_energy_mae:.3g} eV",
            r"$F$" f" MAE: {compress_forces_mae:.3g} eV/\u00c5",
            r"\textbf{Stretch}",
            f"FSD: {stretch_smoothness:.1f} " r"$\mathrm{\AA}^{-1}$",
            r"$E$" f" MAE: {stretch_energy_mae:.3g} eV",
            r"$F$" f" MAE: {stretch_forces_mae:.3g} eV/\u00c5",
        ],
        loc="center left",
        frameon=False,
    )

    return fig
