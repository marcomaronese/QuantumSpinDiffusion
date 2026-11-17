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


@dataclass(frozen=True)
class RadialSpinChart:
    """Unclipped radial charts onto the sphere minus the south pole.

    With standardized radius r, stereographic uses tan(theta/2)=r/length;
    sinh-stereographic uses tan(theta/2)=sinh(r/length). The latter has a
    logarithmically divergent inverse, so a bounded spherical density induces
    exponentially bounded radial tails rather than infinite second moments.
    """

    location: np.ndarray
    scale: np.ndarray
    length: float = 1.0
    kind: str = "stereographic"

    def __post_init__(self):
        validated = EuclideanSpinChart(self.location, self.scale, self.length)
        if self.kind not in ("stereographic", "sinh_stereographic"):
            raise ValueError("unknown radial chart")
        object.__setattr__(self, "location", validated.location)
        object.__setattr__(self, "scale", validated.scale)

    @classmethod
    def fit(cls, training, length=1., kind="stereographic"):
        fitted = EuclideanSpinChart.fit(training)
        return cls(fitted.location, fitted.scale, length, kind)

    def standardize(self, values):
        return EuclideanSpinChart(self.location, self.scale).standardize(values)

    def encode(self, values):
        values = self.standardize(values)
        radius = np.linalg.norm(values, axis=1)
        unit = np.divide(values, radius[:, None], out=np.zeros_like(values), where=radius[:, None]>0)
        a = radius/self.length
        if self.kind == "stereographic":
            inverse = 1/np.hypot(1., a)
            sine, cosine = 2*(a*inverse)*inverse, 2*inverse**2-1
        else:
            exponential = np.exp(-a)
            sech = 2*exponential/(1+exponential**2)
            sine, cosine = 2*np.tanh(a)*sech, 2*sech**2-1
        planar = np.zeros((len(values), 2))
        planar[:, :values.shape[1]] = unit*sine[:, None]
        if np.any((radius>0) & (np.hypot(*planar.T)==0)):
            raise FloatingPointError("finite precision reached the inverse pole")
        return np.column_stack((planar, cosine)), np.zeros(len(values), dtype=bool)

    def decode(self, directions):
        directions = np.asarray(directions, dtype=np.float64)
        if (directions.ndim != 2 or directions.shape[1] != 3
                or not np.isfinite(directions).all()
                or not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-10, rtol=0)):
            raise ValueError("expected finite unit directions")
        x, y, z = directions.T
        planar = np.hypot(x, y)
        if np.any((planar==0) & (z<0)):
            raise ValueError("the south pole has no finite inverse")
        half_tangent = np.zeros_like(planar)
        north = z>=0
        np.divide(planar, 1+z, out=half_tangent, where=north)
        np.divide(1-z, planar, out=half_tangent, where=~north)
        radius = self.length*(half_tangent if self.kind == "stereographic" else np.arcsinh(half_tangent))
        unit = np.divide(directions[:, :2], planar[:, None], out=np.zeros((len(directions), 2)),
                         where=planar[:, None]>0)
        decoded = radius[:, None]*unit
        if not np.isfinite(decoded).all():
            raise FloatingPointError("nonfinite inverse chart")
        return self.location + self.scale*decoded[:, :len(self.location)]

    def local_stretches(self, values):
        """Radial/tangential angular length per standardized Euclidean unit."""
        radius = np.linalg.norm(self.standardize(values), axis=1)
        a = radius/self.length
        if self.kind == "stereographic":
            stretch = 2/self.length/(1+a*a)
            return np.column_stack((stretch, stretch))
        exponential = np.exp(-a)
        sech = 2*exponential/(1+exponential**2)
        radial = 2*sech/self.length
        tangential = np.divide(2*np.tanh(a)*sech, radius,
                               out=np.full_like(radius, 2/self.length), where=radius>0)
        return np.column_stack((radial, tangential))

    def to_dict(self):
        return {"location": self.location.tolist(), "scale": self.scale.tolist(),
                "length": self.length, "kind": self.kind, "dimensions": len(self.location)}
