"""Quantitative heatmaps on the spin-coherent (Bloch-direction) unit sphere.

For j > 1/2 this sphere labels coherent-state directions, not all quantum
states. Radius is always one; only color represents the scalar field.
"""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, lpmv


DENSITY_CMAP = "cividis"
SIGNED_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "sphere_blue_orange", ["#2166ac", "#f7f7f7", "#b35806"]
)
OPPOSITE_VIEWS = ((45.0, 60.0), (-45.0, 240.0))


def sphere_grid(n_theta: int = 49, n_phi: int = 96):
    """Polar theta in [0, pi], azimuth phi in [0, 2 pi), including poles."""
    if n_theta < 3 or n_phi < 4:
        raise ValueError("sphere grid needs n_theta >= 3 and n_phi >= 4")
    return np.linspace(0, np.pi, n_theta), np.linspace(0, 2 * np.pi, n_phi, endpoint=False)


def _angles(thetas, phis):
    theta, phi = np.asarray(thetas, dtype=float), np.asarray(phis, dtype=float)
    if theta.ndim != 1 or phi.ndim != 1 or not theta.size or not phi.size:
        raise ValueError("theta and phi must be nonempty one-dimensional arrays")
    if not (np.isfinite(theta).all() and np.isfinite(phi).all()):
        raise ValueError("angles must be finite")
    if np.any((theta < 0) | (theta > np.pi)) or np.any((phi < 0) | (phi >= 2 * np.pi)):
        raise ValueError("expected theta in [0, pi] and phi in [0, 2 pi)")
    if np.any(np.diff(theta) <= 0) or np.any(np.diff(phi) <= 0):
        raise ValueError("theta and phi must be strictly increasing")
    return theta, phi


def real_spherical_harmonic(ell: int, m: int, thetas, phis):
    """Orthonormal real harmonics, with R_11 = sqrt(3/4pi) x.

    With Condon--Shortley complex Y_lm: R_l0 = Y_l0,
    R_lm = sqrt(2) (-1)^m Re Y_lm for m > 0, and
    R_lm = sqrt(2) (-1)^m Im Y_l,|m| for m < 0.
    Negative m therefore denotes sine, positive m cosine. These are basis
    functions, not nonnegative probability densities or multipole powers.
    """
    if not isinstance(ell, (int, np.integer)) or not isinstance(m, (int, np.integer)):
        raise ValueError("ell and m must be integers")
    if ell < 0 or abs(m) > ell:
        raise ValueError("expected ell >= 0 and |m| <= ell")
    theta, phi = _angles(thetas, phis)
    order = abs(m)
    normalization = np.exp(0.5 * (
        np.log((2 * ell + 1) / (4 * np.pi))
        + gammaln(ell - order + 1) - gammaln(ell + order + 1)
    ))
    polar = normalization * lpmv(order, ell, np.cos(theta))[:, None]
    if m == 0:
        return np.broadcast_to(polar, (theta.size, phi.size)).copy()
    angular = np.cos(order * phi) if m > 0 else np.sin(order * phi)
    return np.sqrt(2) * (-1) ** order * polar * angular[None, :]


def husimi_density(rho, j: float, thetas, phis):
    """Evaluate Q = (2j+1)/(4pi) <Omega|rho|Omega>, per solid angle.

    No sin(theta) factor, grid renormalization, or state repair is applied.
    The input state must already be physical within 1e-10.
    """
    if not np.isfinite(j) or j < 0 or abs(2 * j - round(2 * j)) > 1e-12:
        raise ValueError("j must be a nonnegative integer or half-integer")
    theta, phi = _angles(thetas, phis)
    n = round(2 * j)
    state = np.asarray(rho, dtype=complex)
    if state.shape != (n + 1, n + 1) or not np.isfinite(state).all():
        raise ValueError("density matrix shape/values are inconsistent with j")
    if abs(np.trace(state) - 1) > 1e-10 or np.linalg.norm(state - state.conj().T) > 1e-10:
        raise ValueError("density matrix must be trace-one and Hermitian")
    if np.linalg.eigvalsh(state).min() < -1e-10:
        raise ValueError("density matrix must be positive semidefinite")
    t, p = np.meshgrid(theta, phi, indexing="ij")
    psi = np.stack([
        math.sqrt(math.comb(n, k)) * np.cos(t / 2) ** (n - k)
        * np.sin(t / 2) ** k * np.exp(1j * k * p)
        for k in range(n + 1)
    ], axis=-1)
    return (n + 1) / (4 * np.pi) * np.einsum(
        "...i,ij,...j->...", psi.conj(), state, psi, optimize=True
    ).real


def plot_sphere_heatmap(ax, thetas, phis, values, *, norm, cmap, view=OPPOSITE_VIEWS[0]):
    """Draw an unshaded unit sphere, closing the periodic azimuthal seam.

    Face colors use the mean of their four corner values before colormapping.
    No surface downsampling or lighting alters the quantitative color scale.
    """
    theta, phi = _angles(thetas, phis)
    field = np.asarray(values, dtype=float)
    if theta.size < 3 or phi.size < 4:
        raise ValueError("sphere rendering needs at least 3 theta and 4 phi values")
    if field.shape != (theta.size, phi.size) or not np.isfinite(field).all():
        raise ValueError("field must be finite with shape (n_theta, n_phi)")
    periodic = np.concatenate([field, field[:, :1]], axis=1)
    t, p = np.meshgrid(theta, np.append(phi, phi[0] + 2 * np.pi), indexing="ij")
    faces = (periodic[:-1, :-1] + periodic[1:, :-1]
             + periodic[:-1, 1:] + periodic[1:, 1:]) / 4
    color_map = mpl.colormaps.get_cmap(cmap)
    surface = ax.plot_surface(
        np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t),
        facecolors=color_map(norm(faces)), rstride=1, cstride=1,
        shade=False, linewidth=0, antialiased=False, rasterized=True,
    )
    # Three great circles and explicit axes orient the sphere without lighting.
    a = np.linspace(0, 2 * np.pi, 181)
    c, s, z = 1.002 * np.cos(a), 1.002 * np.sin(a), np.zeros_like(a)
    for coordinates in ((c, s, z), (c, z, s), (z, c, s)):
        ax.plot(*coordinates, color="0.25", alpha=0.20, lw=0.5)
    for label, xyz in zip("xyz", np.eye(3) * 1.16, strict=True):
        ax.text(*xyz, label, color="0.25", fontsize=8, ha="center", va="center")
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))
    ax.set_box_aspect((1, 1, 1), zoom=1.20)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    return surface


def plot_husimi_comparison(thetas, phis, target, reconstructed, *, title="Husimi-Q reconstruction"):
    """Target/model on one absolute scale, signed residual on its own scale."""
    target, reconstructed = np.asarray(target), np.asarray(reconstructed)
    expected = (len(thetas), len(phis))
    if target.shape != expected or reconstructed.shape != expected:
        raise ValueError("target and reconstructed fields must match the angular grid")
    if not (np.isfinite(target).all() and np.isfinite(reconstructed).all()):
        raise ValueError("density fields must be finite")
    if min(target.min(), reconstructed.min()) < -1e-10:
        raise ValueError("density fields must be nonnegative")
    difference = reconstructed - target
    density_norm = mpl.colors.Normalize(0, max(float(target.max()), float(reconstructed.max()), 1e-15))
    bound = max(float(np.abs(difference).max()), np.finfo(float).eps)
    residual_norm = mpl.colors.Normalize(-bound, bound)
    figure = plt.figure(figsize=(11.5, 7.4), layout="constrained")
    grid = figure.add_gridspec(2, 5, width_ratios=(1, 1, 0.045, 1, 0.045))
    for row, view in enumerate(OPPOSITE_VIEWS):
        for column, field, label, norm, cmap in (
            (0, target, "Target (encoded)", density_norm, DENSITY_CMAP),
            (1, reconstructed, "Reconstructed", density_norm, DENSITY_CMAP),
            (3, difference, "Reconstructed − target", residual_norm, SIGNED_CMAP),
        ):
            ax = figure.add_subplot(grid[row, column], projection="3d")
            plot_sphere_heatmap(ax, thetas, phis, field, norm=norm, cmap=cmap, view=view)
            ax.set_title(label if row == 0 else "Opposite view", fontsize=11)
    for column, norm, cmap, label in (
        (2, density_norm, DENSITY_CMAP, r"$Q(\Omega)$ [sr$^{-1}$]"),
        (4, residual_norm, SIGNED_CMAP, r"$\Delta Q(\Omega)$ [sr$^{-1}$]"),
    ):
        figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=figure.add_subplot(grid[:, column]), label=label)
    figure.suptitle(title + "\nSpin-coherent direction sphere; color encodes density, radius = 1", fontsize=13)
    return figure


def plot_harmonic_atlas(thetas, phis, ell_max=3, *, view=OPPOSITE_VIEWS[0]):
    """Real harmonic basis atlas with a single signed, absolute color scale."""
    if not isinstance(ell_max, (int, np.integer)) or ell_max < 0:
        raise ValueError("ell_max must be a nonnegative integer")
    keys = [(ell, m) for ell in range(ell_max + 1) for m in range(-ell, ell + 1)]
    fields = [real_spherical_harmonic(ell, m, thetas, phis) for ell, m in keys]
    bound = max(float(np.abs(field).max()) for field in fields)
    norm = mpl.colors.Normalize(-bound, bound)
    columns = min(4, len(keys))
    rows = math.ceil(len(keys) / columns)
    figure = plt.figure(figsize=(3 * columns + 0.7, 2.65 * rows + 0.8), layout="constrained")
    axes = []
    for index, ((ell, m), field) in enumerate(zip(keys, fields, strict=True), 1):
        ax = figure.add_subplot(rows, columns, index, projection="3d")
        plot_sphere_heatmap(ax, thetas, phis, field, norm=norm, cmap=SIGNED_CMAP, view=view)
        ax.set_title(rf"$R_{{{ell},{m}}}(\theta,\phi)$", fontsize=11, pad=0)
        axes.append(ax)
    figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=SIGNED_CMAP), ax=axes,
                    shrink=0.65, label="Real harmonic amplitude (signed)")
    figure.suptitle("Orthonormal real spherical harmonics\n"
                   rf"$0 \leq \ell \leq {ell_max}$; unit radius; view $(elev,azim)=({view[0]:g}°, {view[1]:g}°)$",
                   fontsize=13)
    return figure
