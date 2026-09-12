"""NWSO example with an inequality constraint g(x) <= 0"""

import numpy as np

from NWSO import NWSOConfig, nwso_optimize


def objective(x: np.ndarray) -> float:
    """Squared distance from the unconstrained optimum (1, 2)."""
    return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)


def inequalities(x: np.ndarray) -> np.ndarray:
    """Enforce x[0] + x[1] <= 2."""
    return np.array([x[0] + x[1] - 2.0])


lower_bounds = np.full(2, -5.0)
upper_bounds = np.full(2, 5.0)

result = nwso_optimize(
    objective,
    lower_bounds,
    upper_bounds,
    config=NWSOConfig(pop_size=40, max_evals=8000, seed=1),
    g_ineq=inequalities,
)

best_solution = result["x_best"]

print("Algorithm: NWSO with an inequality constraint")
print(f"Penalized objective: {result['f_best']:.12e}")
print(f"Unpenalized objective: {objective(best_solution):.12e}")
print(
    f"g(x), feasible when <= 0: "
    f"{inequalities(best_solution)[0]:.12e}"
)
print(f"Best solution: {best_solution}")