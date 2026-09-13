# NWSO and H-NWSO Optimization

Reference Python implementations of **New World Screwworm-Inspired
Optimization (NWSO)** and its hybrid extension **H-NWSO**.

Both algorithms use a normalized fixed-frame axial-orthogonal penetration
operator. H-NWSO additionally incorporates pbest-guided differential
recombination.

## Main features

- Bounded continuous minimization.
- Exact control of the objective-function evaluation budget.
- Reproducible runs through explicit random seeds.
- Optional inequality and equality constraints.
- Optional convergence histories.
- Penetration diagnostics.
- Standalone implementations with minimal dependencies.

## Repository structure

```text
nwso-hnwso-optimization/
├── NWSO.py
├── H_NWSO.py
├── Example_1_NWSO.py
├── Example_2_H_NWSO.py
├── Example_3_constrained.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements

- Python 3.10 or later
- NumPy 1.24 or later

Install the required dependency with:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy
```

## Installation

Clone the repository:

```bash
git clone https://github.com/franciscolopez-bit/nwso-hnwso-optimization.git
cd nwso-hnwso-optimization
```

No package installation is required. The algorithm files can be imported
directly from the repository directory.

## Basic NWSO example

```python
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
```

## Basic H-NWSO example

```python
import numpy as np

from H_NWSO import HNWSOConfig, h_nwso_optimize


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


dimension = 30
lower_bounds = np.full(dimension, -100.0)
upper_bounds = np.full(dimension, 100.0)

config = HNWSOConfig(
    pop_size=40,
    max_evals=8000,
    seed=1,
)

result = h_nwso_optimize(
    sphere,
    lower_bounds,
    upper_bounds,
    config=config,
    return_history=True,
)

print("Best objective value:", result["f_best"])
print("Best solution:", result["x_best"])
print("Function evaluations:", result["n_evals"])
```

## Constrained optimization

Inequality constraints must follow the convention

```text
g(x) <= 0
```

and equality constraints must follow

```text
h(x) = 0
```

They can be supplied through the `g_ineq` and `h_eq` arguments:

```python
def inequalities(x):
    return np.array([x[0] + x[1] - 2.0])

result = nwso_optimize(
    objective,
    lower_bounds,
    upper_bounds,
    config=config,
    g_ineq=inequalities,
)
```

When constraints are supplied, `f_best` contains the penalized comparison
value used by the optimizer. The unpenalized objective and constraint
violations should also be evaluated when reporting the final design.

## Returned results

Both optimization functions return a dictionary containing:

| Key | Description |
|---|---|
| `x_best` | Best solution found |
| `f_best` | Best comparison value found |
| `n_evals` | Number of objective-function evaluations |
| `cycles` | Number of completed lifecycle iterations |
| `history` | Best-so-far history when requested |
| `history_evals` | Evaluation indices associated with the history |
| `penetration_diagnostics` | Diagnostics of the penetration operator |

## Reproducibility

A run is controlled by the `seed` parameter in `NWSOConfig` or
`HNWSOConfig`. The same seed and evaluation budget should be used when
performing paired comparisons between algorithms.

The frozen default parameterizations correspond to the implementations
described in the associated manuscript. The exact archival version used
for the reported experiments is identified by the repository release and
commit hash.

## Examples

Run the included examples with:

```bash
python Example_1_NWSO.py
python Example_2_H_NWSO.py
python Example_3_constrained.py
```

## Associated publication

**Normalized axial-orthogonal search for evaluation-limited engineering
design: New World Screwworm-Inspired Optimization**

Complete bibliographic information and the article DOI will be added after
publication.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE)
for details.

## Contact

**Francisco Javier López-Flores**  
Universidad Autónoma de Sinaloa  
Email: francisco.lopez@uas.edu.mx
