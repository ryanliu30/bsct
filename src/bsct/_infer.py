"""Inference: run a calculator over all BSCT systems."""

import os
from dataclasses import dataclass

import ase
import ase.io
import numpy as np
from ase.calculators.calculator import Calculator
from tqdm import tqdm


@dataclass
class System:
    """
    Ground-truth and calculator results for a single bond-perturbation scan.

    Attributes
    ----------
    dataset : str
        Bond-type directory name (e.g. ``"C-C"``).
    system_id : str
        Molecule identifier within the dataset.
    positions : np.ndarray
        Atomic positions for each frame, shape ``(n_frames, n_atoms, 3)`` in Å.
    gt_energy : np.ndarray
        DFT reference energies, shape ``(n_frames,)`` in eV.
    gt_forces : np.ndarray
        DFT reference forces, shape ``(n_frames, n_atoms, 3)`` in eV/Å.
    calc_energy : np.ndarray
        Calculator energies, shape ``(n_frames,)`` in eV.
    calc_forces : np.ndarray
        Calculator forces, shape ``(n_frames, n_atoms, 3)`` in eV/Å.
    lincoords : np.ndarray
        Bond-length coordinate for each frame, shape ``(n_frames,)`` in Å.

    """

    dataset: str
    system_id: str
    positions: np.ndarray
    gt_energy: np.ndarray
    gt_forces: np.ndarray
    calc_energy: np.ndarray
    calc_forces: np.ndarray
    lincoords: np.ndarray


def run_inference(
    calc: Calculator,
    data_path: str,
    output_path: str,
    store_xyz: bool,
    verbose: bool,
) -> list:
    """
    Run a calculator over all BSCT systems and return structured results.

    For each bond-perturbation scan in ``data_path``, the calculator is applied
    to every frame. When ``store_xyz`` is ``True``, results are written to
    ``output_path/calc_systems`` and cached frames are reused on subsequent
    calls, avoiding redundant forward passes.

    Parameters
    ----------
    calc : Calculator
        ASE calculator used to compute energies and forces.
    data_path : str
        Root directory of the BSCT dataset (contains one sub-directory per
        bond type, each containing one sub-directory per system).
    output_path : str
        Root directory for cached XYZ output files.
    store_xyz : bool
        If ``True``, save calculator results as XYZ files and skip frames
        that have already been written on a previous call.
    verbose : bool
        If ``True``, display per-dataset progress bars.

    Returns
    -------
    list[System]
        One :class:`System` per bond-perturbation scan containing ground-truth
        and calculator energies, forces, positions, and bond-length coordinates.

    """
    systems = []
    datasets = os.listdir(data_path)
    for dataset in datasets:
        dataset_dir = os.path.join(data_path, dataset)
        system_ids = os.listdir(dataset_dir)
        for system_id in tqdm(system_ids, desc=dataset, disable=not verbose):
            system_dir = os.path.join(dataset_dir, system_id)
            partition = np.load(os.path.join(system_dir, "partition.npz"))
            n = int(partition["perturb_range"][2])
            lincoords = np.linspace(*partition["perturb_range"][:2], num=n)

            cache_dir = os.path.join(output_path, "calc_systems", dataset, system_id)
            all_cached = store_xyz and all(
                os.path.exists(os.path.join(cache_dir, f"atoms_{i}.xyz"))
                for i in range(n)
            )

            # Index-keyed dicts so we can insert results in any order
            calc_energy_map: dict = {}
            calc_forces_map: dict = {}

            if all_cached:
                for i in range(n):
                    cached = ase.io.read(os.path.join(cache_dir, f"atoms_{i}.xyz"))
                    calc_energy_map[i] = cached.get_potential_energy()
                    calc_forces_map[i] = cached.get_forces()
            else:
                # Process ALL xyz files in filesystem order (matching the original
                # script) so that CUDA scatter accumulation is identical.
                xyz_files = [f for f in os.listdir(system_dir) if f.endswith(".xyz")]
                for fname in xyz_files:
                    gt_atoms = ase.io.read(os.path.join(system_dir, fname))
                    atoms = gt_atoms.copy()
                    atoms.calc = calc
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()

                    if fname == "unperturbed.xyz":
                        continue

                    idx = int(fname.removeprefix("atoms_").removesuffix(".xyz"))
                    calc_energy_map[idx] = energy
                    calc_forces_map[idx] = forces

                    if store_xyz:
                        xyz_path = os.path.join(cache_dir, fname)
                        os.makedirs(os.path.dirname(xyz_path), exist_ok=True)
                        ase.io.write(xyz_path, atoms)

            positions = []
            gt_energies = []
            gt_forces_list = []
            for i in range(n):
                gt_atoms = ase.io.read(os.path.join(system_dir, f"atoms_{i}.xyz"))
                positions.append(gt_atoms.get_positions())
                gt_energies.append(gt_atoms.get_potential_energy())
                gt_forces_list.append(gt_atoms.get_forces())

            calc_energy = np.array([calc_energy_map[i] for i in range(n)])
            calc_forces = np.array([calc_forces_map[i] for i in range(n)])
            gt_energy = np.array(gt_energies)
            gt_forces = np.array(gt_forces_list)
            positions_arr = np.array(positions)

            mask = ~np.isnan(gt_energy)
            systems.append(
                System(
                    dataset=dataset,
                    system_id=system_id,
                    positions=positions_arr[mask],
                    gt_energy=gt_energy[mask],
                    gt_forces=gt_forces[mask],
                    calc_energy=calc_energy[mask],
                    calc_forces=calc_forces[mask],
                    lincoords=lincoords[mask],
                )
            )
    return systems
