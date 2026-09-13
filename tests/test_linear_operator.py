import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from scripts.optimization.formulations import (
    _xyz_to_flat,
    build_energy_matrix,
    build_inside_energy_matrix,
)
from scripts.model.simulation import simulate


def _basis_truth_from_jax_jacobian(target_indices, C, kernel, domain_dim, param, n_voxels):
    C_jax = jnp.asarray(np.asarray(C))
    zero_input = jnp.zeros(n_voxels, dtype=jnp.asarray(kernel.weights).dtype)

    def inside_constraint_without_shift(I):
        E = simulate(I, kernel, domain_dim, param)
        return E[C_jax == 1]

    jacobian = jax.jacrev(inside_constraint_without_shift)(zero_input)
    return np.asarray(jacobian, dtype=np.float64), jacobian.dtype

def _simulate_inside(I, target_indices, kernel, domain_dim, param):
    simulated = simulate(I, kernel, domain_dim, param)
    return np.asarray(simulated, dtype=np.float64)[target_indices]


def _representative_basis_indices(domain_dim, basis_count):
    n_voxels = int(np.prod(domain_dim))
    if basis_count is None or basis_count >= n_voxels:
        return np.arange(n_voxels, dtype=int)

    dim_x, dim_y, dim_z = domain_dim
    coords = [
        (0, 0, 0),
        (dim_x - 1, 0, 0),
        (0, dim_y - 1, 0),
        (dim_x - 1, dim_y - 1, 0),
        (0, 0, dim_z - 1),
        (dim_x - 1, 0, dim_z - 1),
        (0, dim_y - 1, dim_z - 1),
        (dim_x - 1, dim_y - 1, dim_z - 1),
        (dim_x // 2, dim_y // 2, dim_z // 2),
    ]

    basis_indices = [
        _xyz_to_flat(x, y, z, domain_dim)
        for x, y, z in coords
    ]

    if len(basis_indices) < basis_count:
        basis_indices.extend(
            int(idx)
            for idx in np.linspace(0, n_voxels - 1, num=basis_count, dtype=int)
        )

    return np.array(sorted(set(basis_indices)))[:basis_count]


def verify_inside_energy_matrix(
    A_inside,
    target_indices,
    C,
    kernel,
    domain_dim,
    param,
    *,
    random_count=5,
    basis_count=None,
    seed=0,
    basis_atol=1e-5,
    basis_rtol=1e-5,
    random_atol=None,
    random_rtol=None,
):
    n_voxels = A_inside.shape[1]
    rng = np.random.default_rng(seed)
    random_errors = []
    basis_errors = []
    random_vector_dtype = None
    simulate_output_dtype = None
    kernel_jax_dtype = jnp.asarray(kernel.weights).dtype

    def compare_random_vector(name, vector):
        nonlocal random_vector_dtype, simulate_output_dtype
        vector = np.asarray(vector, dtype=np.float64)
        random_vector_dtype = vector.dtype
        sparse_result = np.asarray(A_inside @ vector, dtype=np.float64)
        simulated_full = simulate(vector, kernel, domain_dim, param)
        simulate_output_dtype = simulated_full.dtype
        simulated = np.asarray(simulated_full, dtype=np.float64)[target_indices]
        diff = sparse_result - simulated
        abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
        scale = np.maximum(np.abs(simulated), 1.0)
        rel_err = float(np.max(np.abs(diff) / scale)) if diff.size else 0.0
        random_errors.append((name, abs_err, rel_err))

    for random_idx in range(random_count):
        compare_random_vector(
            f"random[{random_idx}]",
            rng.uniform(0.0, param.I_max, size=n_voxels),
        )

    if basis_count is None:
        basis_truth, jacobian_dtype = _basis_truth_from_jax_jacobian(
            target_indices,
            C,
            kernel,
            domain_dim,
            param,
            n_voxels,
        )
        diff = np.asarray(A_inside.toarray(), dtype=np.float64) - basis_truth
        abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
        rel_err = float(np.max(np.abs(diff) / np.maximum(np.abs(basis_truth), 1.0))) if diff.size else 0.0
        basis_errors.append(("basis_jacobian_all", abs_err, rel_err))
        basis_mode = "jacobian_all"
        basis_checked = n_voxels
    else:
        jacobian_dtype = None
        basis_indices = _representative_basis_indices(domain_dim, basis_count)
        for basis_idx in basis_indices:
            basis = np.zeros(n_voxels, dtype=np.float64)
            basis[int(basis_idx)] = 1.0
            sparse_result = np.asarray(A_inside @ basis, dtype=np.float64)
            simulated = _simulate_inside(basis, target_indices, kernel, domain_dim, param)
            diff = sparse_result - simulated
            abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
            rel_err = float(np.max(np.abs(diff) / np.maximum(np.abs(simulated), 1.0))) if diff.size else 0.0
            basis_errors.append((f"basis[{int(basis_idx)}]", abs_err, rel_err))
        basis_mode = "all" if len(basis_indices) == n_voxels else "sampled"
        basis_checked = int(len(basis_indices))

    if random_atol is None:
        random_atol = 1e-3 if simulate_output_dtype == np.dtype("float32") else basis_atol
    if random_rtol is None:
        random_rtol = 2e-4 if simulate_output_dtype == np.dtype("float32") else basis_rtol

    random_max_abs = max((entry[1] for entry in random_errors), default=0.0)
    random_max_rel = max((entry[2] for entry in random_errors), default=0.0)
    basis_max_abs = max((entry[1] for entry in basis_errors), default=0.0)
    basis_max_rel = max((entry[2] for entry in basis_errors), default=0.0)

    random_passed = random_max_abs <= random_atol and random_max_rel <= random_rtol
    basis_passed = basis_max_abs <= basis_atol and basis_max_rel <= basis_rtol
    passed = random_passed and basis_passed

    return {
        "passed": passed,
        "random_passed": random_passed,
        "basis_passed": basis_passed,
        "random_max_abs_error": random_max_abs,
        "random_max_rel_error": random_max_rel,
        "basis_max_abs_error": basis_max_abs,
        "basis_max_rel_error": basis_max_rel,
        "random_errors": random_errors,
        "basis_errors": basis_errors,
        "basis_count": basis_checked,
        "basis_mode": basis_mode,
        "random_count": int(random_count),
        "basis_atol": float(basis_atol),
        "basis_rtol": float(basis_rtol),
        "random_atol": float(random_atol),
        "random_rtol": float(random_rtol),
        "dtypes": {
            "A_inside": str(A_inside.dtype),
            "verification_input_vectors": str(random_vector_dtype),
            "kernel_weights_jax": str(kernel_jax_dtype),
            "simulate_output": str(simulate_output_dtype),
            "basis_truth_jacobian": None if jacobian_dtype is None else str(jacobian_dtype),
        },
    }


class InsideEnergyMatrixTest(unittest.TestCase):
    def test_sparse_matrix_matches_simulation(self):
        domain_dim = [3, 4, 2]
        weights = jnp.zeros((3, 3, 3))
        weights = weights.at[1, 1, 1].set(1.0)
        weights = weights.at[2, 1, 1].set(0.25)
        weights = weights.at[1, 0, 2].set(0.5)
        kernel = SimpleNamespace(weights=weights, r_xy=1, r_z=1)
        param = SimpleNamespace(exposure_time=2.0, I_max=1.93)
        C = np.zeros(int(np.prod(domain_dim)), dtype=np.float64)
        C[_xyz_to_flat(1, 2, 1, domain_dim)] = 1
        C[_xyz_to_flat(0, 0, 0, domain_dim)] = 1

        A_inside, target_indices = build_inside_energy_matrix(
            C, kernel, domain_dim, param
        )
        verification = verify_inside_energy_matrix(
            A_inside,
            target_indices,
            C,
            kernel,
            domain_dim,
            param,
            random_count=3,
        )

        self.assertTrue(verification["passed"], verification)

        full_matrix = build_energy_matrix(kernel, domain_dim, param)
        intensity = np.linspace(0.0, param.I_max, num=full_matrix.shape[1])
        matrix_energy = np.asarray(full_matrix @ intensity)
        simulated_energy = np.asarray(
            simulate(intensity, kernel, domain_dim, param)
        )
        np.testing.assert_allclose(
            matrix_energy,
            simulated_energy,
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
