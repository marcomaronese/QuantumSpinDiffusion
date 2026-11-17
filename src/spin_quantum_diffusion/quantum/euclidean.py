"""A fixed bounded chart connecting Euclidean observations and spin directions."""

from dataclasses import dataclass

import numpy as np


def _observations(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (1, 2) or not len(values):
        raise ValueError("expected a nonempty (N, 1) or (N, 2) array")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    return values


@dataclass(frozen=True)
class EuclideanSpinChart:
    """Project standardized data onto a disk and lift to the upper hemisphere.

    Location/scale are fitted on training data only. Radial clipping is explicit;
    decoding a POVM direction is a fixed affine coordinate projection, with no
    trained decoder, singular inverse or state-dependent normalization.
    """

    location: np.ndarray
    scale: np.ndarray
    radius: float = 3.0

    def __post_init__(self):
        location = np.array(self.location, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if (location.ndim != 1 or len(location) not in (1, 2)
                or scale.shape != location.shape or not np.isfinite(location).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()
                or not np.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("invalid chart location, scale or radius")
        location.flags.writeable = scale.flags.writeable = False
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(cls, training, radius=3.0):
        training = _observations(training)
        return cls(training.mean(axis=0), training.std(axis=0), radius)

    def standardize(self, values):
        values = _observations(values)
        if values.shape[1] != len(self.location):
            raise ValueError("observation dimension disagrees with the chart")
        return (values - self.location) / self.scale

    def encode(self, values):
        coordinates = self.standardize(values) / self.radius
        length = np.linalg.norm(coordinates, axis=1)
        coordinates = coordinates / np.maximum(1.0, length[:, None])
        planar = np.zeros((len(coordinates), 2))
        planar[:, :coordinates.shape[1]] = coordinates
        # max only protects the analytic square root from floating-point roundoff.
        z = np.sqrt(np.maximum(0.0, 1.0 - (planar * planar).sum(axis=1)))
        return np.column_stack((planar, z)), length > 1.0

    def decode(self, directions):
        directions = np.asarray(directions, dtype=np.float64)
        if (directions.ndim != 2 or directions.shape[1] != 3
                or not np.isfinite(directions).all()
                or not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-10,
                                   rtol=0)):
            raise ValueError("expected finite unit directions of shape (N, 3)")
        return self.location + self.radius * self.scale * directions[:, :len(self.location)]

    def to_dict(self):
        return {"location": self.location.tolist(), "scale": self.scale.tolist(),
                "radius": self.radius, "dimensions": len(self.location)}
