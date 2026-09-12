"""
New World Screwworm-Inspired Optimization (NWSO)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np


# =============================================================================
# Section 1. Parameters
# =============================================================================

@dataclass(frozen=True)
class PenetrationConfig:
    """Parameters of the normalized axial-orthogonal penetration operator"""

    mode: str = "full"
    instars_max: int = 3
    instar_power: float = 1.0
    axial_start: float = 0.70
    axial_end: float = 0.05
    axial_power: float = 1.0
    radius0: float = 0.04
    radius_power: float = 2.0
    radius_div_scale: float = 0.03
    radial_contraction: float = 0.65
    turns: float = 1.0


@dataclass
class NWSOConfig:
    """Frozen parameterization used in the reported NWSO experiments"""

    pop_size: int = 40
    max_evals: Optional[int] = None
    seed: Optional[int] = 42
    beta_levy: float = 1.5
    p_flight_max: float = 0.35
    p_flight_min: float = 0.05
    flight_scale0: float = 0.12
    wound_frac: float = 0.25
    attraction_temperature: float = 0.35
    eggs_min: int = 2
    eggs_max: int = 9
    egg_sigma0: float = 0.08
    reinfestation_bias: float = 0.20
    penetration: PenetrationConfig = field(default_factory=PenetrationConfig)
    stall_fraction: float = 0.08
    diversity_min: float = 0.02
    p_pupate: float = 0.10
    emergence_sigma: float = 0.12
    emergence_mode: str = "mixed"
    p_mating: float = 0.15
    mating_noise: float = 0.02
    elitism: int = 1
    rho_g: float = 1.0e6
    rho_h: float = 1.0e6
    eq_tol: float = 1.0e-6


# =============================================================================
# Section 2. Main optimization cycle (read this section first)
# =============================================================================

def nwso_optimize(
    f: Objective,
    lb: Array,
    ub: Array,
    config: Optional[NWSOConfig] = None,
    g_ineq: Optional[Constraint] = None,
    h_eq: Optional[Constraint] = None,
    return_history: bool = False) -> dict[str, object]:

    """Minimize an objective with the biologically ordered NWSO algorithm
    Parameters
    ----------
    f
        Objective function ``f(x)`` to minimize
    lb, ub
        One-dimensional lower and upper bounds
    config
        Algorithm parameters. The defaults reproduce the frozen version
    g_ineq
        Optional inequalities in the convention ``g(x) <= 0``
    h_eq
        Optional equalities in the convention ``h(x) = 0``
    return_history
        Include best-so-far values and their evaluation indices

    Returns
    -------
    dict
        Best point, best penalized value, evaluation count, cycle count,
        optional convergence history, and penetration diagnostics.
    """

    cfg = config or NWSOConfig()
    max_evaluations = int(cfg.max_evals or cfg.pop_size * 501)
    lower, upper = validate_problem(
        lb,
        ub,
        cfg.pop_size,
        max_evaluations)

    rng = np.random.default_rng(cfg.seed)
    population_size = int(cfg.pop_size)
    dimension = lower.size
    span = upper - lower

    objective = prepare_objective(
        f,
        g_ineq,
        h_eq,
        cfg.rho_g,
        cfg.rho_h,
        cfg.eq_tol)
    evaluator = BudgetEvaluator(objective, max_evaluations)

    # Stage 0: initialize and evaluate the adult population.
    adults = latin_hypercube_sampling(
        population_size,
        dimension,
        lower,
        upper,
        rng)

    adult_values = evaluator.evaluate(adults)
    cycles = 0
    penetration_totals = new_penetration_diagnostics()

    while evaluator.remaining > 0:
        cycles += 1
        cycle_start = evaluator.evaluations
        progress = evaluator.progress

        dispersion_probability = (
            cfg.p_flight_max * (1.0 - progress)
            + cfg.p_flight_min * progress)
        dispersion_scale = cfg.flight_scale0 * (1.0 - progress)
        egg_scale = cfg.egg_sigma0 * (1.0 - progress) + 1.0e-12
        elites, elite_values = save_elites(
            adults,
            adult_values,
            cfg.elitism)

        # Stage 1: adults disperse by bounded Levy flights.
        dispersed = np.flatnonzero(
            rng.random(population_size) < dispersion_probability
        )[: evaluator.remaining]
        if dispersed.size:
            trials = clip_to_bounds(
                adults[dispersed]
                + dispersion_scale
                * levy_flight(
                    (dispersed.size, dimension),
                    cfg.beta_levy,
                    rng)
                * span,
                lower,
                upper)

            trial_values = evaluator.evaluate(trials)
            adults[dispersed] = trials
            adult_values[dispersed] = trial_values
        restore_elites(adults, adult_values, elites, elite_values)

        # Stage 2: rank adults and retain the most attractive wound sites.
        wound_count = max(
            1, min(population_size,
                int(round(cfg.wound_frac * population_size))))

        wound_indices = np.argsort(adult_values)[:wound_count]
        wounds = adults[wound_indices].copy()
        wound_values = adult_values[wound_indices]
        scaled_values = (wound_values - np.min(wound_values)) / (
            np.ptp(wound_values) + 1.0e-12)

        attraction = np.exp( -scaled_values / max(cfg.attraction_temperature, 1.0e-12))
        attraction = ((1.0 - cfg.reinfestation_bias) * attraction + cfg.reinfestation_bias)
        attraction /= np.sum(attraction)

        # Stage 3: accepted wounds receive rank-dependent egg batches.
        larval_batches: list[Array] = []
        for rank, wound in enumerate(wounds):
            rank_weight = (
                1.0
                if wound_count == 1
                else 1.0 - rank / (wound_count - 1))

            egg_count = int(
                round( cfg.eggs_min + (cfg.eggs_max - cfg.eggs_min) * rank_weight))
            egg_count = max(cfg.eggs_min, min(cfg.eggs_max, egg_count))
            acceptance_probability = min( 1.0, attraction[rank] * wound_count)
            if rng.random() <= acceptance_probability:
                batch = (wound + rng.normal(
                        		0.0,
					egg_scale,
                        		size=(egg_count, dimension)) * span)
                larval_batches.append(clip_to_bounds(batch, lower, upper))
        larvae = (
            np.vstack(larval_batches)
            if larval_batches
            else np.empty((0, dimension)))

        # Stage 4: larvae follow coherent penetration paths toward wound guides.
        if larvae.size and evaluator.remaining:
            larvae = larvae[: evaluator.remaining]
            diversity = population_diversity(
                adults,
                adult_values,
                lower,
                upper)

            guide_indices = rng.integers(
                0,
                wounds.shape[0],
                size=larvae.shape[0])

            guides = wounds[guide_indices]
            guide_values = wound_values[guide_indices]
            larvae, batch_diagnostics = coherent_larval_penetration(
                larvae,
                guides,
                lower,
                upper,
                progress,
                diversity,
                cfg.penetration,
                rng)

            add_penetration_diagnostics(
                penetration_totals,
                batch_diagnostics)

            # Stage 5: terminal larvae compete with the current adults.
            larval_values = evaluator.evaluate(larvae)
            penetration_totals["guide_successes"] += float(
                np.sum(larval_values < guide_values))

            candidate_pool = np.vstack((adults, larvae))
            candidate_values = np.concatenate(
                (adult_values, larval_values))

            survivors = np.argsort(candidate_values)[:population_size]
            penetration_totals["survivors"] += float(
                np.sum(survivors >= population_size))

            adults = candidate_pool[survivors].copy()
            adult_values = candidate_values[survivors].copy()
            restore_elites(adults, adult_values, elites, elite_values)

        # Stage 6: stagnation and low diversity activate adult emergence.
        current_best = adults[int(np.argmin(adult_values))]
        diversity = population_diversity(
            adults,
            adult_values,
            lower,
            upper)

        stall_limit = max(
            population_size,
            int(math.ceil(cfg.stall_fraction * max_evaluations)))

        is_stalled = (
            evaluator.evaluations
            - evaluator.last_improvement_evaluation
            >= stall_limit)

        emergence_is_active = (
            is_stalled
            and diversity <= cfg.diversity_min
            and cfg.p_pupate > 0
            and evaluator.remaining)

        if emergence_is_active:
            emergence_count = min(
                max(1, int(round(cfg.p_pupate * population_size))),
                evaluator.remaining)

            replaced = np.argsort(adult_values)[-emergence_count:]
            if cfg.emergence_mode == "uniform":
                emerged = lower + rng.random(
                    (emergence_count, dimension)) * span

            elif cfg.emergence_mode == "near_best":
                emerged = (
                    current_best
                    + rng.normal(
                        0.0,
                        cfg.emergence_sigma,
                        size=(emergence_count, dimension))* span)
            else:
                local_emergence = (
                    current_best
                    + rng.normal(
                        0.0,
                        cfg.emergence_sigma,
                        size=(emergence_count, dimension))* span)
                global_emergence = lower + rng.random(
                    (emergence_count, dimension)) * span
                emerged = np.where(
                    rng.random((emergence_count, 1)) < 0.5,
                    local_emergence,
                    global_emergence)
            emerged = clip_to_bounds(emerged, lower, upper)
            adults[replaced] = emerged
            adult_values[replaced] = evaluator.evaluate(emerged)

        # Stage 7: a fraction of emerged adults produces mating offspring.
        if cfg.p_mating > 0 and evaluator.remaining:
            mating_count = min(
                max(1, int(round(cfg.p_mating * population_size))),
                population_size,
                evaluator.remaining)
            mothers = rng.choice(
                population_size,
                size=mating_count,
                replace=False)
            fathers = rng.choice(
                population_size,
                size=mating_count,
                replace=True)
            maternal_weight = rng.random((mating_count, 1))
            offspring = (
                maternal_weight * adults[mothers]
                + (1.0 - maternal_weight) * adults[fathers])
            offspring += (
                rng.normal(
                    0.0,
                    cfg.mating_noise,
                    size=(mating_count, dimension))* span)

            offspring = clip_to_bounds(offspring, lower, upper)
            offspring_values = evaluator.evaluate(offspring)
            improved = offspring_values < adult_values[mothers]
            adults[mothers[improved]] = offspring[improved]
            adult_values[mothers[improved]] = offspring_values[improved]

        # A fallback evaluation guarantees progress for disabled stage settings.
        if evaluator.evaluations == cycle_start:
            replaced = int(np.argmax(adult_values))
            point = lower + rng.random((1, dimension)) * span
            adults[replaced] = point[0]
            adult_values[replaced] = evaluator.evaluate(point)[0]

    return build_result(
        evaluator,
        cycles,
        return_history,
        penetration_totals)


# =============================================================================
# Section 3. Internal numerical utilities
# =============================================================================

Array = np.ndarray


Objective = Callable[[Array], float]


Constraint = Callable[[Array], Array]


def clip_to_bounds(points: Array, lower: Array, upper: Array) -> Array:
    """Project coordinates onto their box constraints."""
    return np.minimum(np.maximum(points, lower), upper)


def levy_flight( shape: Tuple[int, ...], beta: float, rng: np.random.Generator,) -> Array:

    """Generate Levy-stable increments with Mantegna's method"""
    numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    denominator = ( math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (numerator / denominator) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    return u / (np.abs(v) ** (1.0 / beta) + 1.0e-12)

def latin_hypercube_sampling(population_size: int, dimension: int, lower: Array, upper: Array, rng: np.random.Generator,) -> Array:

    """Generate one Latin-hypercube population inside the search bounds"""
    population = np.empty((population_size, dimension), dtype=float)

    for coordinate in range(dimension):
        unit_samples = (rng.permutation(population_size) + rng.random(population_size)) / population_size
        population[:, coordinate] = (lower[coordinate] + unit_samples * (upper[coordinate] - lower[coordinate]))
    return population


def validate_problem( lower: Array, upper: Array, population_size: int, max_evaluations: int) -> tuple[Array, Array]:
    """Validate bounds and the minimum population/budget requirements"""

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.ndim != 1 or lower.shape != upper.shape or np.any(upper <= lower):
        raise ValueError( "lower and upper must be one-dimensional arrays with upper > lower")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if max_evaluations < population_size:
        raise ValueError("max_evaluations must be at least population_size")
    return lower, upper


def quadratic_penalty( objective: Objective, point: Array, inequalities: Optional[Constraint], equalities: Optional[Constraint], inequality_weight: float, equality_weight: float, equality_tolerance: float) -> float:
    """Evaluate an objective plus squared inequality/equality penalties"""
    value = float(objective(point))
    if inequalities is not None:
        violation = np.maximum( 0.0, np.atleast_1d(inequalities(point)).astype(float))
        value += inequality_weight * float(np.dot(violation, violation))
    if equalities is not None:
        violation = np.maximum( 0.0, np.abs(np.atleast_1d(equalities(point)).astype(float)) - equality_tolerance)
        value += equality_weight * float(np.dot(violation, violation))
    return value


def prepare_objective( objective: Objective, inequalities: Optional[Constraint], equalities: Optional[Constraint], inequality_weight: float, equality_weight: float, equality_tolerance: float) -> Objective:
    """Return the raw objective or its constrained penalized form"""
    if inequalities is None and equalities is None:
        return objective

    def penalized(point: Array) -> float:
        return quadratic_penalty( objective, point, inequalities, equalities, inequality_weight, equality_weight, equality_tolerance)

    return penalized


class BudgetEvaluator:
    """Evaluate candidates while enforcing an exact objective-call budget"""

    def __init__(self, objective: Objective, max_evaluations: int):
        self.objective = objective
        self.max_evaluations = int(max_evaluations)
        self.evaluations = 0
        self.best_value = math.inf
        self.best_point: Optional[Array] = None
        self.last_improvement_evaluation = 0
        self.history_evaluations: list[int] = []
        self.history_values: list[float] = []

    @property
    def remaining(self) -> int:
        """Number of objective evaluations still available"""
        return self.max_evaluations - self.evaluations

    @property
    def progress(self) -> float:
        """Consumed fraction of the objective-evaluation budget"""
        return min(1.0, self.evaluations / max(1, self.max_evaluations))

    def evaluate(self, points: Array) -> Array:
        """Evaluate a two-dimensional candidate matrix within the budget"""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be a two-dimensional array")
        if points.shape[0] > self.remaining:
            raise RuntimeError("attempted to exceed the evaluation budget")

        if hasattr(self.objective, "evaluate_batch"):
            raw_values = np.asarray(
                self.objective.evaluate_batch(points),
                dtype=float,
            )
        else:
            raw_values = np.asarray(
                [self.objective(point) for point in points],
                dtype=float,
            )

        values = np.empty(points.shape[0], dtype=float)
        for index, (point, raw_value) in enumerate(zip(points, raw_values)):
            value = float(raw_value)
            if math.isnan(value):
                value = math.inf
            values[index] = value
            self.evaluations += 1
            if value < self.best_value:
                self.best_value = value
                self.best_point = point.copy()
                self.last_improvement_evaluation = self.evaluations
        self._record_history()
        return values

    def _record_history(self) -> None:
        if (
            not self.history_evaluations
            or self.history_evaluations[-1] != self.evaluations
        ):
            self.history_evaluations.append(self.evaluations)
            self.history_values.append(float(self.best_value))


def population_diversity( population: Array, values: Array, lower: Array, upper: Array) -> float:

    """Mean normalized distance from the current best population member"""
    span = upper - lower
    dimension = lower.size
    current_best = population[int(np.argmin(values))]
    normalized_population = (population - lower) / span
    normalized_best = (current_best - lower) / span
    return float(np.mean(np.linalg.norm(normalized_population - normalized_best, axis=1)/ math.sqrt(dimension)))


def save_elites(population: Array, values: Array, elite_count: int) -> tuple[Array, Array]:

    """Copy the requested number of best adults before non-greedy movement"""
    count = max(0, min(int(elite_count), population.shape[0]))
    indices = np.argsort(values)[:count]
    return population[indices].copy(), values[indices].copy()


def restore_elites( population: Array, values: Array, elites: Array, elite_values: Array) -> None:

    """Restore a missing elite only when it improves the current worst adult"""
    for elite, elite_value in zip(elites, elite_values):
        is_present = np.any(np.all(population == elite, axis=1))
        if not is_present:
            worst = int(np.argmax(values))
            if elite_value < values[worst]:
                population[worst] = elite
                values[worst] = elite_value


def new_penetration_diagnostics() -> dict[str, float]:
    """Create the counters accumulated over all larval batches"""
    return {
        "batches": 0.0,
        "larvae": 0.0,
        "instar_steps": 0.0,
        "axial_path": 0.0,
        "transverse_path": 0.0,
        "clipped_rows": 0.0,
        "guide_successes": 0.0,
        "survivors": 0.0,
        "instars_used_sum": 0.0,
        "radius_sum": 0.0,
        "axial_total_sum": 0.0}


def add_penetration_diagnostics( totals: dict[str, float], batch: dict[str, float]) -> None:

    """Accumulate diagnostic values from one penetration batch."""
    totals["batches"] += 1.0
    for key in ( "larvae", "instar_steps", "axial_path", "transverse_path", "clipped_rows"):

        totals[key] += float(batch[key])
    totals["instars_used_sum"] += float(batch["instars_used"])
    totals["radius_sum"] += float(batch["radius"])
    totals["axial_total_sum"] += float(batch["axial_total"])


def summarize_penetration_diagnostics( totals: dict[str, float]) -> dict[str, float]:

    """Convert accumulated counters into reportable means and rates"""
    larvae = max(1.0, totals["larvae"])
    batches = max(1.0, totals["batches"])
    return {
        **{key: float(value) for key, value in totals.items()},
        "guide_success_rate": float(totals["guide_successes"] / larvae),
        "survival_rate": float(totals["survivors"] / larvae),
        "clipping_rate": float(
            totals["clipped_rows"] / max(1.0, totals["instar_steps"])),

        "mean_axial_path": float(totals["axial_path"] / larvae),
        "mean_transverse_path": float(totals["transverse_path"] / larvae),
        "mean_instars": float(totals["instars_used_sum"] / batches),
        "mean_radius": float(totals["radius_sum"] / batches),
        "mean_axial_total": float(totals["axial_total_sum"] / batches)}


def build_result(evaluator: BudgetEvaluator, cycles: int, return_history: bool, diagnostics: Optional[dict[str, float]] = None) -> dict[str, Any]:

    """Build the public result dictionary returned by both algorithms"""
    if evaluator.best_point is None:
        raise RuntimeError("the objective was never evaluated")

    result: dict[str, Any] = {
        "x_best": evaluator.best_point.copy(),
        "f_best": float(evaluator.best_value),
        "n_evals": int(evaluator.evaluations),
        "cycles": int(cycles)}

    if return_history:
        result["history"] = np.asarray(evaluator.history_values, dtype=float)
        result["history_evals"] = np.asarray(
            evaluator.history_evaluations,
            dtype=int)

    if diagnostics is not None:
        result["penetration_diagnostics"] = (summarize_penetration_diagnostics(diagnostics))

    return result


# =============================================================================
# Section 4. Coherent larval-penetration operator
# =============================================================================

def _validate_config(config: PenetrationConfig) -> None:
    if config.mode not in {"full", "axial", "transverse", "none"}:
        raise ValueError(f"unknown penetration mode: {config.mode}")
    if config.instars_max < 1:
        raise ValueError("instars_max must be at least one")
    if (
        config.instar_power <= 0
        or config.axial_power <= 0
        or config.radius_power <= 0):

        raise ValueError("penetration schedule powers must be positive")
    if not 0.0 <= config.axial_start < 1.0:
        raise ValueError("axial_start must lie in [0, 1)")
    if not 0.0 <= config.axial_end < 1.0:
        raise ValueError("axial_end must lie in [0, 1)")
    if config.radius0 < 0 or config.radius_div_scale <= 0:
        raise ValueError("invalid penetration-radius parameters")
    if not 0.0 < config.radial_contraction <= 1.0:
        raise ValueError("radial_contraction must lie in (0, 1]")
    if config.turns < 0:
        raise ValueError("turns must be nonnegative")


def build_axial_transverse_frames(starts: Array, guides: Array, rng: np.random.Generator) -> tuple[Array, Array, Array]:

    """Build one axial direction and two orthogonal transverse directions"""
    starts = np.asarray(starts, dtype=float)
    guides = np.asarray(guides, dtype=float)
    if starts.shape != guides.shape or starts.ndim != 2:
        raise ValueError("starts and guides must be equally shaped matrices")

    larvae_count, dimension = starts.shape
    raw_axis = guides - starts
    axis_norm = np.linalg.norm(raw_axis, axis=1, keepdims=True)

    fallback_axis = rng.normal(size=(larvae_count, dimension))
    fallback_axis /= (
        np.linalg.norm(fallback_axis, axis=1, keepdims=True) + 1.0e-12)
    axis = np.where( axis_norm > 1.0e-12, raw_axis / (axis_norm + 1.0e-12), fallback_axis)

    if dimension == 1:
        zeros = np.zeros_like(axis)
        return axis, zeros, zeros

    first = rng.normal(size=(larvae_count, dimension))
    first -= np.sum(first * axis, axis=1, keepdims=True) * axis
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    invalid = first_norm[:, 0] <= 1.0e-10
    if np.any(invalid):
        rows = np.flatnonzero(invalid)
        coordinates = np.argmin(np.abs(axis[rows]), axis=1)
        replacement = np.zeros((rows.size, dimension), dtype=float)
        replacement[np.arange(rows.size), coordinates] = 1.0
        replacement -= ( np.sum(replacement * axis[rows], axis=1, keepdims=True) * axis[rows])
        first[rows] = replacement
        first_norm[rows] = np.linalg.norm(
            replacement,
            axis=1,
            keepdims=True)
    transverse_u = first / (first_norm + 1.0e-12)

    if dimension == 2:
        return axis, transverse_u, np.zeros_like(transverse_u)

    second = rng.normal(size=(larvae_count, dimension))
    second -= np.sum(second * axis, axis=1, keepdims=True) * axis
    second -= ( np.sum(second * transverse_u, axis=1, keepdims=True) * transverse_u)
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    invalid = second_norm[:, 0] <= 1.0e-10
    if np.any(invalid):
        rows = np.flatnonzero(invalid)
        coordinates = np.argmin(
            axis[rows] ** 2 + transverse_u[rows] ** 2,
            axis=1)
        replacement = np.zeros((rows.size, dimension), dtype=float)
        replacement[np.arange(rows.size), coordinates] = 1.0
        replacement -= (
            np.sum(replacement * axis[rows], axis=1, keepdims=True)
            * axis[rows])
        replacement -= (
            np.sum(replacement * transverse_u[rows], axis=1, keepdims=True)
            * transverse_u[rows])
        second[rows] = replacement
        second_norm[rows] = np.linalg.norm(
            replacement,
            axis=1,
            keepdims=True)
    transverse_v = second / (second_norm + 1.0e-12)
    return axis, transverse_u, transverse_v


def coherent_larval_penetration(
    larvae: Array,
    guides: Array,
    lower: Array,
    upper: Array,
    progress: float,
    diversity: float,
    config: PenetrationConfig,
    rng: np.random.Generator) -> tuple[Array, dict[str, float]]:

    """Generate one terminal penetration endpoint for every larva

    The internal instar positions form a latent coherent axial-orthogonal path.
    They do not consume objective evaluations. Only the final endpoint returned
    by this function is evaluated by NWSO or H-NWSO.
    """
    _validate_config(config)
    larvae = np.asarray(larvae, dtype=float)
    guides = np.asarray(guides, dtype=float)
    if larvae.shape != guides.shape or larvae.ndim != 2:
        raise ValueError("larvae and guides must be equally shaped matrices")
    if larvae.shape[0] == 0:
        return larvae.copy(), {
            "larvae": 0.0,
            "instar_steps": 0.0,
            "axial_path": 0.0,
            "transverse_path": 0.0,
            "clipped_rows": 0.0,
            "instars_used": 0.0,
            "radius": 0.0,
            "axial_total": 0.0}

    progress = float(np.clip(progress, 0.0, 1.0))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    normalized_larvae = (larvae - lower) / span
    normalized_guides = (guides - lower) / span

    _, transverse_u, transverse_v = build_axial_transverse_frames(
        normalized_larvae,
        normalized_guides,
        rng)

    initial_phase = rng.uniform(0.0, 2.0 * math.pi, size=(larvae.shape[0], 1))
    orientation = np.where(
        rng.random((larvae.shape[0], 1)) < 0.5,-1.0, 1.0)

    instars = max( 1, int( math.ceil( config.instars_max * (1.0 - progress) ** config.instar_power - 1.0e-12)))

    axial_total = config.axial_end + (
        config.axial_start - config.axial_end
    ) * (1.0 - progress) ** config.axial_power
    axial_per_instar = 1.0 - (1.0 - axial_total) ** (1.0 / instars)

    radius = config.radius0 * (1.0 - progress) ** config.radius_power
    radius *= diversity / (diversity + config.radius_div_scale)

    normalized_scale = math.sqrt(larvae.shape[1])
    initial_distance = np.linalg.norm(
        normalized_guides - normalized_larvae,
        axis=1,
        keepdims=True) / max(normalized_scale, 1.0e-12)
    distance_scale = 0.1 + 0.9 * np.clip(initial_distance, 0.0, 1.0)

    centerline = normalized_larvae.copy()
    normalized_positions = normalized_larvae.copy()
    previous_offset = np.zeros_like(normalized_positions)
    axial_path = 0.0
    transverse_path = 0.0
    clipped_rows = 0
    use_axial = config.mode in {"full", "axial"}
    use_transverse = config.mode in {"full", "transverse"}

    for instar in range(1, instars + 1):
        phase = initial_phase + orientation * (
            2.0 * math.pi * config.turns * instar / instars)
        transverse_direction = (
            np.cos(phase) * transverse_u + np.sin(phase) * transverse_v)
        instar_radius = radius * config.radial_contraction ** (instar - 1)
        offset = instar_radius * distance_scale * transverse_direction

        previous_centerline = centerline.copy()
        if use_axial:
            centerline += axial_per_instar * (normalized_guides - centerline)
        axial_step = centerline - previous_centerline
        transverse_step = (
            offset - previous_offset
            if use_transverse
            else np.zeros_like(offset))

        proposal = centerline + offset if use_transverse else centerline
        clipped = np.clip(proposal, 0.0, 1.0)
        clipped_rows += int(
            np.sum(np.any(np.abs(clipped - proposal) > 1.0e-12, axis=1)))
        axial_path += float(np.sum( np.linalg.norm(axial_step, axis=1) / max(normalized_scale, 1.0e-12)))
        transverse_path += float(
            np.sum(np.linalg.norm(transverse_step, axis=1)/ max(normalized_scale, 1.0e-12)))
        normalized_positions = clipped
        previous_offset = offset if use_transverse else np.zeros_like(offset)

    positions = lower + normalized_positions * span
    diagnostics = {
        "larvae": float(larvae.shape[0]),
        "instar_steps": float(larvae.shape[0] * instars),
        "axial_path": axial_path,
        "transverse_path": transverse_path,
        "clipped_rows": float(clipped_rows),
        "instars_used": float(instars),
        "radius": float(radius),
        "axial_total": float(axial_total)}
    return positions, diagnostics


__all__ = ["PenetrationConfig", "NWSOConfig", "nwso_optimize"]