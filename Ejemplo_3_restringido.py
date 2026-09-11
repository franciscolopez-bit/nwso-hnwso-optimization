"""Ejemplo de NWSO con una restriccion g(x) <= 0."""

import numpy as np

from NWSO import NWSOConfig, nwso_optimize


def objetivo(x: np.ndarray) -> float:
    """Distancia cuadratica al punto no restringido (1, 2)."""
    return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)


def desigualdades(x: np.ndarray) -> np.ndarray:
    """Imponer x0 + x1 <= 2."""
    return np.array([x[0] + x[1] - 2.0])


limite_inferior = np.full(2, -5.0)
limite_superior = np.full(2, 5.0)

resultado = nwso_optimize(
    objetivo,
    limite_inferior,
    limite_superior,
    config=NWSOConfig(pop_size=40, max_evals=8000, seed=1),
    g_ineq=desigualdades,
)

mejor_solucion = resultado["x_best"]
print("Algoritmo: NWSO con restriccion")
print(f"Objetivo penalizado: {resultado['f_best']:.12e}")
print(f"Objetivo sin penalizar: {objetivo(mejor_solucion):.12e}")
print(f"g(x), factible si <= 0: {desigualdades(mejor_solucion)[0]:.12e}")
print(f"Mejor solucion: {mejor_solucion}")
