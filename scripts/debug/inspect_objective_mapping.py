"""Inspect the current optimization problem, objective, and solver mappings."""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path


# Allow `python scripts/debug/inspect_objective_mapping.py` from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.optimization import formulations, objectives, problems, solvers


EXPECTED_MAPPINGS = {
    "obj_fun_1": ("EnergyProblem", "obj_fun_1"),
    "obj_fun_2": ("CureProblem", "obj_fun_2"),
    "guven": ("GuvenProblem", "obj_fun_guven"),
    "wang": ("WangProblem", "obj_fun_wang"),
    "reverse_guven": ("ReverseGuvenProblem", "obj_fun_reverse_guven"),
}
SOLVER_NAMES = (
    "projected_gradient_descent",
    "slsqp",
    "trust_constr",
    "linprog_highs",
)


def _function_tree(function):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    function_node = next(
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return source, function_node


def _return_expressions(function) -> list[str]:
    _, function_node = _function_tree(function)
    return [
        ast.unparse(node.value)
        for node in ast.walk(function_node)
        if isinstance(node, ast.Return) and node.value is not None
    ]


def _called_objective_function(problem) -> tuple[str | None, str | None]:
    """Resolve the objective-module function called by a problem method.

    This is static inspection: evaluating an objective produces a scalar and
    cannot by itself reveal the source-level function that produced it.
    """
    objective_method = type(problem).objective
    _, function_node = _function_tree(objective_method)
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        candidate = objective_method.__globals__.get(node.func.id)
        if (
            inspect.isfunction(candidate)
            and candidate.__module__ == objectives.__name__
        ):
            return candidate.__name__, candidate.__module__
    return None, None


def _residual_convention() -> str:
    """Read the lower/upper residual bounds from the constrained adapter."""
    _, function_node = _function_tree(formulations._build_constrained)
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "FunctionalConstraintSpec":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        lower = keywords.get("lower_bound")
        upper = keywords.get("upper_bound")
        if lower is not None and upper is not None:
            return f"{ast.unparse(lower)} <= residual <= {ast.unparse(upper)}"
    return "could not determine from _build_constrained"


def _constraint_interpretation(expression: str) -> str:
    try:
        parsed = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return "every returned residual must satisfy the configured bounds"
    if isinstance(parsed, ast.BinOp) and isinstance(parsed.op, ast.Sub):
        return f"{ast.unparse(parsed.left)} >= {ast.unparse(parsed.right)}"
    return "every returned residual must satisfy the configured bounds"


def _guven_voxel_selection(function) -> str:
    _, function_node = _function_tree(function)
    calls_boundary_helper = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "guven_boundary_mask"
        and function.__globals__.get(node.func.id)
        is getattr(objectives, "guven_boundary_mask", None)
        for node in ast.walk(function_node)
    )
    return_expressions = _return_expressions(function)
    indexes_all_outside = any("E[C == 0]" in expression for expression in return_expressions)
    if calls_boundary_helper:
        return "Guven boundary voxels only (calls guven_boundary_mask)"
    if indexes_all_outside:
        return "all outside voxels (indexes E[C == 0])"
    return "could not determine from the current source"


def _print_problem(problem_name: str) -> bool:
    problem = problems.PROBLEMS[problem_name]
    expected_class, expected_objective = EXPECTED_MAPPINGS[problem_name]
    actual_objective, objective_module = _called_objective_function(problem)
    objective_method = type(problem).objective
    mapping_matches = (
        type(problem).__name__ == expected_class
        and problem.name == problem_name
        and actual_objective == expected_objective
    )

    print("=" * 68)
    print(f"Problem: {problem_name}")
    print("=" * 68)
    print(f"Registry key:          {problem_name}")
    print(f"Problem class:         {type(problem).__name__}")
    print(f"problem.name:          {problem.name}")
    print(f"objective_is_smooth:   {problem.objective_is_smooth}")
    print(f"has_constraints:       {problem.has_constraints}")
    print(f"Objective method:      {objective_method.__qualname__}")
    print(f"Base objective:        {actual_objective or 'NOT DETECTED'}")
    print(f"Objective module:      {objective_module or 'NOT DETECTED'}")
    print(f"Expected objective:    {expected_objective}")
    print(f"Status:                {'PASS' if mapping_matches else 'FAIL'}")
    if not mapping_matches:
        print(f"Expected:              {expected_class} -> {expected_objective}")
        print(
            "Actual:                "
            f"{type(problem).__name__} -> {actual_objective or 'NOT DETECTED'}"
        )

    objective_function = getattr(objectives, actual_objective, None)
    if objective_function is None:
        print("Objective expression:  NOT AVAILABLE")
    else:
        print("Objective expression:")
        for expression in _return_expressions(objective_function):
            print(f"  {expression}")
        if problem_name == "guven":
            print(f"Guven voxel selection: {_guven_voxel_selection(objective_function)}")

    constraint_method = type(problem).__dict__.get("constraint_residuals")
    if constraint_method is None:
        print("Constraint residuals:  inherited empty residual vector")
    else:
        expressions = _return_expressions(constraint_method)
        print("Constraint residuals:")
        for expression in expressions:
            print(f"  {expression}")
            print(f"  therefore: {_constraint_interpretation(expression)}")
        print(f"Convention:            {_residual_convention()}")
    print()
    return mapping_matches


def _compatibility(problem_name: str, solver_name: str) -> tuple[str, str]:
    try:
        formulation_name = solvers.check_compatibility(problem_name, solver_name)
    except ValueError as error:
        return "unsupported", str(error)
    handler = solvers.SOLVERS[solver_name].formulation_handlers[formulation_name]
    return formulation_name, handler.solve.__qualname__


def _print_solver_mappings() -> None:
    headings = ("Problem", "PGD", "SLSQP", "trust-constr", "HiGHS")
    short_names = (
        "projected_gradient_descent",
        "slsqp",
        "trust_constr",
        "linprog_highs",
    )
    rows = []
    for problem_name in EXPECTED_MAPPINGS:
        rows.append(
            (problem_name,)
            + tuple(_compatibility(problem_name, solver_name)[0] for solver_name in short_names)
        )

    widths = [
        max(len(heading), *(len(row[index]) for row in rows))
        for index, heading in enumerate(headings)
    ]
    print("SOLVER / FORMULATION COMPATIBILITY")
    print("=" * 68)
    print("  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()

    print("RESOLVED EXECUTION PATHS")
    print("=" * 68)
    for problem_name in EXPECTED_MAPPINGS:
        problem = problems.PROBLEMS[problem_name]
        base_objective, _ = _called_objective_function(problem)
        for solver_name in SOLVER_NAMES:
            formulation_name, detail = _compatibility(problem_name, solver_name)
            if formulation_name == "unsupported":
                print(f"{problem_name} + {solver_name}: unsupported ({detail})")
                continue
            print(
                f"{problem_name} -> {type(problem).__name__} -> {base_objective} "
                f"-> {formulation_name} -> {detail}"
            )
    print()


def main() -> int:
    print("OBJECTIVE MAPPING INSPECTION")
    print("Static method inspection is used to identify objective calls.")
    print()
    passed = [_print_problem(problem_name) for problem_name in EXPECTED_MAPPINGS]

    print("OBJECTIVE MAPPING VALIDATION")
    print("=" * 68)
    for problem_name, is_correct in zip(EXPECTED_MAPPINGS, passed, strict=True):
        problem = problems.PROBLEMS[problem_name]
        actual_objective, _ = _called_objective_function(problem)
        status = "PASS" if is_correct else "FAIL"
        print(
            f"[{status}] {problem_name:<16} -> {type(problem).__name__:<20} "
            f"-> {actual_objective or 'NOT DETECTED'}"
        )
    print()
    print(f"{sum(passed)} / {len(passed)} objective mappings correct.")
    print()
    _print_solver_mappings()
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
