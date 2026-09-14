# Accuracy Optimization Benchmark

The project separates the shared benchmark, mathematical optimization problems,
and solver algorithms.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running One Experiment

Guven with projected gradient descent:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem guven \
  --solver projected_gradient_descent
```

Guven as a linear program with HiGHS:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem guven \
  --solver linprog
```

Wang with projected gradient descent:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem wang \
  --solver projected_gradient_descent
```

Run all problems compatible with one solver:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem all \
  --solver projected_gradient_descent
```

Run selected problems with one solver, using spaces or commas:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem guven wang \
  --solver linprog
```

The equivalent comma syntax is `--problem guven,wang`. With `--problem all`,
incompatible problems are skipped; an explicitly selected incompatible problem
causes a clear compatibility error.

The module form `python -m scripts.main ...` is supported as well.

## Architecture

```text
scripts/
|-- main.py
|-- benchmark/
|   |-- config.py
|   `-- core.py
|-- optimization/
|   |-- config.py
|   |-- formulations.py
|   |-- objectives.py
|   |-- problems.py
|   |-- solvers.py
|   `-- legacy.py
|-- model/
|   |-- geometry.py
|   |-- indexing.py
|   |-- kernel.py
|   |-- parameters.py
|   `-- simulation.py
|-- visualization/
|   `-- plots.py
`-- tools/
    |-- kernel_demo.py
    `-- process_image_stack.py
```

`scripts/benchmark/config.py` loads only shared experimental settings:

- geometry and domain dimensions
- physical process parameters
- initialization and target energy
- evaluation metrics
- visualization and output settings

`scripts/optimization/objectives.py` contains only base objectives.
`scripts/optimization/problems.py` combines those objectives with semantic
constraints and bounds. Neither module contains solver-dependent penalties.

`scripts/optimization/formulations.py` adapts a base problem to a penalty,
direct-constrained, or linear representation according to solver capabilities.

`scripts/optimization/solvers.py` owns solver algorithms and declares which
representation each solver requires. The compatibility check selects the
representation before the benchmark is built or solved.

`scripts/main.py` combines one benchmark, one or more selected problems, and one
solver at runtime. It contains no problem-specific or solver-specific dispatch
branches.

## Benchmark Configuration

`configs/benchmark_testcube_20.yaml` contains no problem or solver selection:

```yaml
benchmark:
  name: testcube_20
  random_seed: 0

geometry:
  type: testcube
  domain_dim: [20, 20, 20]
  offset: [5, 5, 5]

process:
  layer_height: 0.05
  exposure_time: 2.0
  atten_coef: 11.82
  sigma: 2.0
  I_max: 1.93
  E_crit: 0.97
  min_infl_factor: 0.001
```

`target_energy.reference: initial_intensity` records the existing benchmark
behavior explicitly: changing `initialization.scale` also changes `T` for
`obj_fun_1` and `wang`.

## Parameters

User-editable defaults live in `configs/optimization_defaults.yaml`, separately
from the physical benchmark configuration:

```yaml
problems: {}

penalties:
  guven:
    penalty_weight: 100.0
  wang:
    penalty_weight: 50.0
  reverse_guven:
    penalty_weight: 100.0

solvers:
  projected_gradient_descent:
    step_size: 0.03
    max_iterations: 100
    objective_tolerance: null
  linprog_highs: {}
```

The file separates mathematical problem parameters, penalty-adapter parameters,
and solver parameters. It does not contain problem-solver combinations. Penalty
values are loaded only when the selected solver needs an exterior penalty.

For `guven + projected_gradient_descent` the resolved defaults include:

```text
problem:  no parameters
penalty:  penalty_weight=100.0
solver:   step_size=0.03, max_iterations=100, objective_tolerance=null
```

For `guven + linprog` they include only:

```text
problem: no parameters
penalty: none
solver:  no parameters
```

Values can be overridden without changing the benchmark file:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --problem guven \
  --solver projected_gradient_descent \
  --penalty-param penalty_weight=250 \
  --solver-param step_size=0.02 \
  --solver-param max_iterations=200
```

Use a different defaults file when needed:

```bash
python scripts/main.py \
  --config configs/benchmark_testcube_20.yaml \
  --optimization-config configs/optimization_defaults.yaml \
  --problem wang \
  --solver projected_gradient_descent
```

The precedence is code fallback, then `optimization_defaults.yaml`, then CLI
override. The fully resolved values are recorded in `resolved_config.yaml`.

Irrelevant or unknown parameters are rejected. For example, `step_size` is not
accepted for HiGHS and `penalty_weight` is not accepted for the constrained
Guven formulation.

## Compatibility

| Problem | Projected gradient descent | SLSQP | trust-constr | HiGHS |
|---|---:|---:|---:|---:|
| `obj_fun_1` | supported | supported | supported | unsupported |
| `obj_fun_2` | supported | unsupported | supported | supported |
| `guven` | supported | supported | supported | supported |
| `wang` | supported | unsupported | supported | supported |
| `reverse_guven` | supported | supported | supported | supported |

Guven always means energy minimization on the one-voxel-wide exterior boundary
of the target geometry, using an 8-neighbor XY boundary with no Z expansion,
with a hard inside cure constraint. This differs from the benchmark's separate
all-outside energy metrics. PGD cannot enforce that constraint directly, so its
formulation adapter creates the configured quadratic exterior penalty. SLSQP,
trust-constr, and HiGHS receive the constraint directly and never receive
`penalty_weight`.

Wang always means L1 target matching with hard inside and outside cure
constraints. PGD receives the configured linear exterior penalty. The exact LP
uses target-error auxiliaries and hard cure constraints; it contains no cure
violation slacks and no `penalty_weight`.

The outside constraint uses `E <= E_crit` as specified by the mathematical
problem. The physical `cured` and `overcured_voxels` evaluation uses
`E >= E_crit`; an outside voxel exactly on the threshold is therefore feasible
for the optimizer but counted as cured by the evaluation metric.

SLSQP selects the first compatible representation from a general priority list:
explicit constraints, then a smooth objective with bounds. Constraints in the
first representation may themselves be linear or nonlinear. Consequently Guven
and Reverse Guven use their hard constrained formulations. Wang and `obj_fun_2`
remain unsupported by SLSQP because their direct
objectives are nonsmooth. Routing Wang's auxiliary-variable LP through SLSQP
would require a dense matrix of several gigabytes for the default benchmark;
HiGHS remains the correct solver for that representation.

`obj_fun_2` intentionally remains the original soft cure-violation objective. Its
LP adapter is an exact epigraph reformulation of that same objective.

## Results

Each command writes its artifacts to:

```text
results/<benchmark>/<problem>/<solver>/
|-- result.json
|-- resolved_config.yaml
|-- optimized_intensity.npy
`-- plots/
```

`resolved_config.yaml` contains both the benchmark settings and the selected,
fully resolved problem, penalty-adapter, and solver parameters. Results also
record the selected adapter and `constraint_handling` as `direct`, `penalty`, or
`none`.

All solutions are evaluated with the same physical metrics. The raw
`base_objective_value` and the solver-facing `formulation_objective_value` are
stored separately. They differ for exterior-penalty runs. Objective values
should not be compared across different mathematical problems.

## Extending Problems And Solvers

A new problem defines only its base objective, constraints, and problem
parameters in `scripts/optimization/problems.py`. Generic objective, penalty, and
constrained adapters work without solver changes. A special exact LP builder, if
available, belongs in `scripts/optimization/formulations.py`.

A new solver registers one or more formulation handlers in `SOLVERS`, in priority
order, together with parameter defaults and result labels. It never needs to
reference concrete problem names. No changes to `main.py` or the benchmark
configuration loader are required.

JAX 64-bit mode is enabled by default so JAX and SciPy use consistent precision.
Set `ACCURACY_OPT_ENABLE_JAX_X64=false` only for a deliberate float32 benchmark.
