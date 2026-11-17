"""Declared low-dimensional target families and sample discrepancy diagnostics."""

import numpy as np
from scipy.spatial.distance import cdist


# Families follow arXiv:2611.20139, Figs. 4–5. Parameters absent from its
# captions are declared here; these are analogues, not an exact replication.
TARGETS = {
    "bimodal": {"dimensions": 1, "means": [-2., 2.], "std": .45},
    "peak_exponential": {"dimensions": 1, "peak_weight": .65,
                         "peak_mean": 0., "peak_std": .3, "exponential_scale": 4.},
    "student_t": {"dimensions": 1, "df": 3},
    "gaussian": {"dimensions": 1, "mean": 0., "std": 1.},
    "log_normal": {"dimensions": 1, "log_mean": 0., "log_std": 1.},
    "skewed_mixture": {"dimensions": 1, "weights": [.55, .3, .15],
                       "means": [-2., 1., 5.], "stds": [.4, .8, 1.4]},
    "correlated_gaussian": {"dimensions": 2, "covariance": [[1., .9], [.9, 1.]]},
    "banana": {"dimensions": 2, "x_std": 1.5, "curvature": .45,
               "offset": 2.5, "noise_std": .3},
    "anticorrelated_gaussian": {"dimensions": 2,
                               "covariance": [[1., -.65], [-.65, 1.6]]},
    "funnel": {"dimensions": 2, "x_std": 1., "log_variance_slope": 1.2},
    "two_moons": {"dimensions": 2, "noise_std": .08},
    "ring": {"dimensions": 2, "radius": 2., "radial_std": .12},
}


def sample_target(name, count, seed):
    if name not in TARGETS or not isinstance(count, int) or count < 1:
        raise ValueError("unknown target or invalid sample count")
    spec = TARGETS[name]
    rng = np.random.default_rng(seed)
    if name == "bimodal":
        values = rng.choice(spec["means"], count) + spec["std"] * rng.normal(size=count)
    elif name == "peak_exponential":
        peak = rng.normal(spec["peak_mean"], spec["peak_std"], count)
        tail = rng.exponential(spec["exponential_scale"], count)
        values = np.where(rng.random(count) < spec["peak_weight"], peak, tail)
    elif name == "student_t":
        values = rng.standard_t(spec["df"], count)
    elif name == "gaussian":
        values = rng.normal(spec["mean"], spec["std"], count)
    elif name == "log_normal":
        values = rng.lognormal(spec["log_mean"], spec["log_std"], count)
    elif name == "skewed_mixture":
        component = rng.choice(3, count, p=spec["weights"])
        values = np.array(spec["means"])[component] + (
            np.array(spec["stds"])[component] * rng.normal(size=count))
    elif "gaussian" in name:
        return rng.multivariate_normal([0., 0.], spec["covariance"], count)
    elif name == "banana":
        x = rng.normal(0, spec["x_std"], count)
        return np.column_stack((x, spec["curvature"] * (x*x - spec["offset"])
                                + rng.normal(0, spec["noise_std"], count)))
    elif name == "funnel":
        x = rng.normal(0, spec["x_std"], count)
        return np.column_stack((x, rng.normal(size=count)
                                * np.exp(.5 * spec["log_variance_slope"] * x)))
    elif name == "two_moons":
        angle = rng.uniform(0, np.pi, count)
        second = rng.integers(0, 2, count).astype(bool)
        x = np.where(second, 1 - np.cos(angle), np.cos(angle))
        y = np.where(second, .5 - np.sin(angle), np.sin(angle))
        return np.column_stack((x, y)) + rng.normal(0, spec["noise_std"], (count, 2))
    elif name == "ring":
        angle = rng.uniform(0, 2*np.pi, count)
        radius = rng.normal(spec["radius"], spec["radial_std"], count)
        return radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    else:
        raise ValueError("target sampler is missing")
    return values[:, None]


def sample_discrepancies(reference, generated, energy_count=1024):
    """W1 (1D) / 32-direction sliced W1 (2D), and biased energy V-statistic.

    Inputs have equal counts and have already been scaled using training data.
    Energy is 2 mean|X-Y| - mean|X-X'| - mean|Y-Y'|, including diagonals.
    This is a finite-sample diagnostic, not a calibrated hypothesis test.
    """
    reference, generated = np.asarray(reference), np.asarray(generated)
    if (reference.ndim != 2 or reference.shape != generated.shape
            or reference.shape[1] not in (1, 2) or not len(reference)
            or not np.isfinite(reference).all() or not np.isfinite(generated).all()):
        raise ValueError("expected equal finite arrays of shape (N, 1 or 2)")
    if reference.shape[1] == 1:
        projections = np.ones((1, 1))
    else:
        angles = np.arange(32) * np.pi / 32
        projections = np.column_stack((np.cos(angles), np.sin(angles)))
    difference = np.sort(reference @ projections.T, axis=0) - (
        np.sort(generated @ projections.T, axis=0))
    x, y = reference[:energy_count], generated[:energy_count]
    energy = 2*cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()
    return {"sliced_w1": float(np.abs(difference).mean()), "energy_v": float(energy)}
