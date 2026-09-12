"""Basic example: Minimizing Sphere with NWSO"""

import numpy as np

from NWSO import NWSOConfig, nwso_optimize


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


dimension = 30
lower_bounds = np.full(dimension, -100.0)
upper_bounds = np.full(dimension, 100.0)

config = NWSOConfig(
    pop_size=40,
    max_evals=8000,
    seed=1,
)

result = nwso_optimize(
    sphere,
    lower_bounds,
    upper_bounds,
    config=config,
    return_history=True,
)

print("Best objective value:", result["f_best"])
print("Best solution:", result["x_best"])
print("Function evaluations:", result["n_evals"])
