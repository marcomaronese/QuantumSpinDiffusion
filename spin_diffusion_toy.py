#!/usr/bin/env python3
"""
Minimal proof-of-concept: Spin phase-space quantum diffusion as a generative
model for directional data on S^2.

Task
----
Learn a bimodal probability distribution on the sphere.

Representation
--------------
A point Omega=(theta,phi) is encoded as a spin-j coherent state |Omega>_j.
The empirical data distribution is represented by

    rho_data = E_Omega[ |Omega><Omega| ].

Forward diffusion
-----------------
We use the isotropic Lindbladian

    d rho / dt = -D sum_a [J_a,[J_a,rho]],  a=x,y,z,

whose multipoles decay as exp[-D l(l+1)t].

Reverse model
-------------
Each reverse step is a trainable CPTP collision channel:
1. append a fresh ancilla qubit |0>,
2. apply a symmetry-preserving parameterized unitary U_theta,
3. trace/reset the ancilla.

The reverse stack is trained against the known forward trajectory and starts
from I/(2j+1). After training, the generated density matrix defines a Husimi-Q
distribution on S^2, from which classical directions can be sampled.

This first toy problem intentionally models a distribution of directions, not
full images. It is the cleanest test of the spin/spherical construction.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
import matplotlib.pyplot as plt

from spin_losses import (
    build_husimi_grid,
    frobenius_loss,
    multipole_loss,
    objective_loss,
    quantum_fidelity,
    trace_distance,
)
from spin_multipoles import (
    irreducible_spherical_tensors,
    max_multipole_decay_error,
    multipole_powers,
    multipole_trajectory,
)


torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
RDTYPE = torch.float64


def spin_operators(j: float):
    """Spin-j matrices in the |j,m> basis, m=j,j-1,...,-j."""
    d = int(round(2 * j + 1))
    m = torch.arange(j, -j - 1, -1, dtype=RDTYPE)

    Jz = torch.diag(m).to(CDTYPE)
    Jp = torch.zeros((d, d), dtype=CDTYPE)

    # J_+ |j,m> = sqrt[(j-m)(j+m+1)] |j,m+1>
    for src in range(1, d):
        mm = float(m[src])
        coeff = math.sqrt((j - mm) * (j + mm + 1))
        Jp[src - 1, src] = coeff

    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2j)
    I = torch.eye(d, dtype=CDTYPE)
    return Jx, Jy, Jz, I


def spin_coherent_state(j: float, theta: float, phi: float):
    """
    Symmetric N=2j qubit coherent state written directly in the Dicke basis.

    Basis index k is the number of |1> excitations, with m=j-k.
    """
    N = int(round(2 * j))
    amps = []
    for k in range(N + 1):
        amp = (
            math.sqrt(math.comb(N, k))
            * (math.cos(theta / 2) ** (N - k))
            * (math.sin(theta / 2) ** k)
            * np.exp(1j * k * phi)
        )
        amps.append(amp)

    psi = torch.tensor(amps, dtype=CDTYPE)
    return psi / torch.linalg.norm(psi)


def xyz_to_angles(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    theta = np.arccos(np.clip(v[2], -1.0, 1.0))
    phi = np.arctan2(v[1], v[0]) % (2 * np.pi)
    return float(theta), float(phi)


def make_direction_dataset(n_samples: int, sigma: float, seed: int):
    """
    Simple bimodal distribution on S^2.

    This is not an exact von Mises-Fisher sampler; it is deliberately simple:
    draw around two 3D mean directions and renormalize to the sphere.
    """
    rng = np.random.default_rng(seed)

    mus = [
        np.array([0.80, 0.00, 0.60]),
        np.array([-0.40, 0.70, 0.60]),
    ]
    mus = [m / np.linalg.norm(m) for m in mus]

    points = []
    for _ in range(n_samples):
        mu = mus[int(rng.random() > 0.5)]
        v = mu + sigma * rng.normal(size=3)
        v = v / np.linalg.norm(v)
        points.append(xyz_to_angles(v))
    return points


def empirical_density(points, j: float):
    d = int(round(2 * j + 1))
    rho = torch.zeros((d, d), dtype=CDTYPE)

    for theta, phi in points:
        psi = spin_coherent_state(j, theta, phi)
        rho = rho + torch.outer(psi, psi.conj())

    return rho / len(points)


def liouvillian_matrix(Js, D: float):
    """
    Matrix representation of:
        L(rho) = -D sum_a [J_a,[J_a,rho]]

    using column-vectorization vec(rho).
    """
    d = Js[0].shape[0]
    I = torch.eye(d, dtype=CDTYPE)
    L = torch.zeros((d * d, d * d), dtype=CDTYPE)

    for J in Js:
        C = (
            torch.kron(I.contiguous(), J.contiguous())
            - torch.kron(J.T.contiguous(), I.contiguous())
        )
        L = L - D * (C @ C)

    return L


def apply_superoperator(S, rho):
    """Apply a linear superoperator using column-vectorization."""
    vec = rho.T.contiguous().reshape(-1)
    out = S @ vec
    return out.reshape(rho.shape).T.contiguous()


def build_forward_trajectory(rho0, Jx, Jy, Jz, D, dt, T):
    L = liouvillian_matrix([Jx, Jy, Jz], D=D)
    S = torch.tensor(expm((dt * L).detach().cpu().numpy()), dtype=CDTYPE)

    trajectory = [rho0]
    rho = rho0
    for _ in range(T):
        rho = apply_superoperator(S, rho)
        trajectory.append(rho)
    return trajectory


def normalized_generator(G):
    return G / torch.linalg.norm(G)


def build_reverse_generators(Jx, Jy, Jz, I):
    """
    Hardware-motivated generators on spin-j system + one ancilla qubit.

    In the symmetric N=2j qubit encoding:
      J_a = 1/2 sum_i sigma_a^(i)
    and J_z^2 corresponds to permutation-symmetric pair interactions.
    """
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)
    Ia = torch.eye(2, dtype=CDTYPE)

    def kron(a, b):
        return torch.kron(a.contiguous(), b.contiguous())

    Jz2 = Jz @ Jz

    generators = [
        kron(Jx, Ia),
        kron(Jy, Ia),
        kron(Jz, Ia),
        kron(Jz2, Ia),
        kron(Jx, X),
        kron(Jy, Y),
        kron(Jz, Z),
        kron(I, X),
        kron(I, Y),
    ]
    return [normalized_generator(g) for g in generators]


def partial_trace_ancilla(joint, d):
    """
    joint acts on system(d) x ancilla(2).
    """
    joint = joint.reshape(d, 2, d, 2)
    return joint[:, 0, :, 0] + joint[:, 1, :, 1]


def reverse_collision_unitary(theta, generators):
    """Build the system--ancilla unitary for one reverse step."""
    dim = generators[0].shape[0]
    U = torch.eye(dim, dtype=CDTYPE, device=theta.device)

    for layer in range(theta.shape[0]):
        for k, G in enumerate(generators):
            Uk = torch.matrix_exp(-1j * theta[layer, k] * G)
            U = Uk @ U
    return U


def collision_kraus_operators(U, system_dimension):
    """Extract the two Kraus operators ``<a|U|0>`` for the ancilla."""
    if U.shape != (2 * system_dimension, 2 * system_dimension):
        raise ValueError("unitary dimension is inconsistent with the system")
    blocks = U.reshape(system_dimension, 2, system_dimension, 2)
    return [blocks[:, a, :, 0] for a in range(2)]


def choi_matrix_from_kraus(kraus_operators):
    """Construct the unnormalized Choi matrix from Kraus operators."""
    vectors = [K.T.contiguous().reshape(-1) for K in kraus_operators]
    return sum(torch.outer(vec, vec.conj()) for vec in vectors)


def reverse_collision_channel(rho, theta, generators):
    """
    theta shape: [layers, n_generators].

    Append |0><0|_a, apply U_theta, trace the ancilla.
    """
    d = rho.shape[0]
    U = reverse_collision_unitary(theta, generators)

    anc0 = torch.tensor([[1, 0], [0, 0]], dtype=CDTYPE)
    joint_in = torch.kron(rho.contiguous(), anc0)
    joint_out = U @ joint_in @ U.conj().T

    return partial_trace_ancilla(joint_out, d)


def train_reverse(
    forward_states,
    generators,
    layers,
    epochs,
    lr,
    seed,
    objective="frobenius",
    multipole_tensors=None,
    multipole_weighting="rank-balanced",
    husimi_grid=None,
):
    torch.manual_seed(seed)

    T = len(forward_states) - 1
    d = forward_states[0].shape[0]
    prior = torch.eye(d, dtype=CDTYPE) / d

    params = torch.nn.Parameter(
        0.03 * torch.randn(T, layers, len(generators), dtype=RDTYPE)
    )
    optimizer = torch.optim.Adam([params], lr=lr)

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        rho = prior
        loss = torch.tensor(0.0, dtype=RDTYPE)

        # params[0] implements the T -> T-1 reverse step.
        for p_index, t in enumerate(range(T, 0, -1)):
            rho = reverse_collision_channel(rho, params[p_index], generators)
            loss = loss + objective_loss(
                rho,
                forward_states[t - 1],
                objective,
                tensors=multipole_tensors,
                multipole_weighting=multipole_weighting,
                husimi_grid=husimi_grid,
            )

        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            final_err = float(frobenius_loss(rho, forward_states[0]).detach())
            print(
                f"epoch={epoch:5d}  "
                f"{objective}_path_loss={float(loss.detach()):.6e}  "
                f"final_state_error={final_err:.6e}"
            )

    return params.detach(), history


@torch.no_grad()
def generate_trajectory(prior, params, generators):
    """Return the prior and every state produced by the reverse stack."""
    rho = prior
    trajectory = [rho]
    for step in range(params.shape[0]):
        rho = reverse_collision_channel(rho, params[step], generators)
        trajectory.append(rho)
    return trajectory


@torch.no_grad()
def generate_density(prior, params, generators):
    return generate_trajectory(prior, params, generators)[-1]


def density_matrix_metrics(rho):
    """Return trace, Hermiticity residual, and smallest Hermitian eigenvalue."""
    trace = torch.trace(rho)
    hermiticity_error = torch.linalg.norm(rho - rho.conj().T)
    hermitian_part = 0.5 * (rho + rho.conj().T)
    minimum_eigenvalue = torch.linalg.eigvalsh(hermitian_part).min()
    return {
        "trace_real": float(torch.real(trace)),
        "trace_imag": float(torch.imag(trace)),
        "hermiticity_error": float(hermiticity_error),
        "minimum_eigenvalue": float(minimum_eigenvalue),
    }


def assert_physical_density(rho, label, tolerance=1e-10):
    """Raise if a state is not trace-one, Hermitian, and positive semidefinite."""
    metrics = density_matrix_metrics(rho)
    if abs(metrics["trace_real"] - 1.0) > tolerance:
        raise AssertionError(f"{label}: trace is not one: {metrics}")
    if abs(metrics["trace_imag"]) > tolerance:
        raise AssertionError(f"{label}: trace has an imaginary part: {metrics}")
    if metrics["hermiticity_error"] > tolerance:
        raise AssertionError(f"{label}: state is not Hermitian: {metrics}")
    if metrics["minimum_eigenvalue"] < -tolerance:
        raise AssertionError(f"{label}: state is not positive: {metrics}")
    return metrics


def husimi_q_grid(rho, j, n_theta=80, n_phi=160):
    """
    Q(Omega) = (2j+1)/(4 pi) <Omega|rho|Omega>.

    The returned grid is also multiplied by sin(theta) when converted into a
    discrete sampling probability, because dOmega = sin(theta)dtheta dphi.
    """
    thetas = np.linspace(1e-5, np.pi - 1e-5, n_theta)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)

    Q = np.zeros((n_theta, n_phi), dtype=float)
    prefactor = (2 * j + 1) / (4 * np.pi)

    for i, theta in enumerate(thetas):
        for k, phi in enumerate(phis):
            psi = spin_coherent_state(j, float(theta), float(phi))
            val = torch.real(psi.conj() @ rho @ psi)
            Q[i, k] = prefactor * float(val)

    # Normalize numerically with spherical measure.
    weights = Q * np.sin(thetas)[:, None]
    minimum_weight = float(weights.min())
    if minimum_weight < -1e-12:
        raise ValueError(f"Husimi-Q grid has negative weight {minimum_weight}")
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()

    return thetas, phis, Q, weights


def sample_from_grid(thetas, phis, weights, n_samples, seed):
    rng = np.random.default_rng(seed)
    flat = weights.reshape(-1)
    indices = rng.choice(flat.size, size=n_samples, p=flat)

    n_phi = len(phis)
    ti = indices // n_phi
    pi = indices % n_phi

    # Small cell jitter for a less pixelated sample cloud.
    dtheta = thetas[1] - thetas[0]
    dphi = phis[1] - phis[0]

    theta_samples = np.clip(
        thetas[ti] + rng.uniform(-0.5, 0.5, size=n_samples) * dtheta,
        0.0,
        np.pi,
    )
    phi_samples = (
        phis[pi] + rng.uniform(-0.5, 0.5, size=n_samples) * dphi
    ) % (2 * np.pi)

    return np.stack([theta_samples, phi_samples], axis=1)


def plot_q_map(thetas, phis, Q, title, output_file):
    """
    Equirectangular diagnostic plot. For scientific spherical analysis,
    replace this with HEALPix or another equal-area representation.
    """
    lon = phis - np.pi
    lat = np.pi / 2 - thetas

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.roll(Q, shift=Q.shape[1] // 2, axis=1),
        shading="auto",
    )
    ax.set_xlabel("longitude [rad]")
    ax.set_ylabel("latitude [rad]")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Husimi Q")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training(history, output_file):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(np.arange(len(history)), history)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reverse path loss")
    ax.set_title("Training of the reverse quantum channel stack")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multipole_power(times, powers, output_file):
    """Plot total Hilbert--Schmidt power in each multipole rank."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ell, values in sorted(powers.items()):
        ax.semilogy(
            times,
            values.detach().cpu().numpy(),
            marker="o",
            label=fr"$\ell={ell}$",
        )
    ax.set_xlabel("diffusion time")
    ax.set_ylabel(r"$P_\ell(t)=\sum_m |c_{\ell m}(t)|^2$")
    ax.set_title("Forward diffusion: multipole power")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_multipole_decay_comparison(times, powers, D, output_file):
    """Compare normalized observed root-power with the theoretical decay."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    times_array = np.asarray(times, dtype=float)
    for ell, values in sorted(powers.items()):
        initial = float(values[0])
        if initial <= 1e-24:
            continue
        observed = np.sqrt(np.maximum(values.detach().cpu().numpy(), 0.0) / initial)
        theoretical = np.exp(-D * ell * (ell + 1) * times_array)
        (line,) = ax.semilogy(
            times_array,
            observed,
            "o",
            label=fr"observed $\ell={ell}$",
        )
        ax.semilogy(times_array, theoretical, "--", color=line.get_color())
    ax.set_xlabel("diffusion time")
    ax.set_ylabel(r"$\sqrt{P_\ell(t)/P_\ell(0)}$")
    ax.set_title(r"Observed multipole decay vs. $e^{-D\ell(\ell+1)t}$")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--j", type=float, default=2.0)
    parser.add_argument("--n-data", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=0.22)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--D", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.35)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--objective",
        choices=[
            "frobenius",
            "multipole",
            "fidelity",
            "trace-distance",
            "husimi-js",
        ],
        default="frobenius",
    )
    parser.add_argument(
        "--multipole-weighting",
        choices=["uniform", "rank-balanced", "high-rank"],
        default="rank-balanced",
    )
    parser.add_argument("--loss-grid-theta", type=int, default=16)
    parser.add_argument("--loss-grid-phi", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="spin_diffusion_output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    j = args.j
    N = int(round(2 * j))
    d = N + 1

    print(f"spin j={j}, symmetric-qubit count N={N}, spin dimension d={d}")

    # 1. Synthetic directional dataset.
    points = make_direction_dataset(args.n_data, args.sigma, args.seed)

    # 2. Encode the empirical distribution as a spin-j mixed state.
    rho_data = empirical_density(points, j)

    # 3. Spin operators and exact forward diffusion.
    Jx, Jy, Jz, I = spin_operators(j)
    forward_states = build_forward_trajectory(
        rho_data, Jx, Jy, Jz, args.D, args.dt, args.T
    )

    forward_metrics = [
        assert_physical_density(rho, f"forward state {step}")
        for step, rho in enumerate(forward_states)
    ]

    prior = I / d
    initial_distance = float(torch.linalg.norm(forward_states[0] - prior))
    terminal_distance = float(torch.linalg.norm(forward_states[-1] - prior))
    print(f"||rho_T - I/d||_F = {terminal_distance:.6e}")

    # 4. Resolve the exact forward trajectory into irreducible multipoles.
    times = [step * args.dt for step in range(args.T + 1)]
    tensors = irreducible_spherical_tensors(j, Jx, Jy)
    coefficients = multipole_trajectory(forward_states, tensors)
    powers = multipole_powers(coefficients)
    decay_error = max_multipole_decay_error(coefficients, times, args.D)
    if decay_error > 1e-10:
        raise AssertionError(
            f"multipole decay disagrees with theory: max error={decay_error}"
        )
    print(f"max multipole decay error = {decay_error:.6e}")

    # 5. Train the reverse quantum collision channels.
    generators = build_reverse_generators(Jx, Jy, Jz, I)
    loss_husimi_grid = None
    if args.objective == "husimi-js":
        loss_husimi_grid = build_husimi_grid(
            j,
            spin_coherent_state,
            n_theta=args.loss_grid_theta,
            n_phi=args.loss_grid_phi,
        )
    params, history = train_reverse(
        forward_states=forward_states,
        generators=generators,
        layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        objective=args.objective,
        multipole_tensors=tensors,
        multipole_weighting=args.multipole_weighting,
        husimi_grid=loss_husimi_grid,
    )

    # 6. Generate from the maximally mixed prior and validate every state.
    reverse_states = generate_trajectory(prior, params, generators)
    reverse_metrics = [
        assert_physical_density(rho, f"reverse state {step}")
        for step, rho in enumerate(reverse_states)
    ]
    rho_generated = reverse_states[-1]
    final_error = float(frobenius_loss(rho_generated, rho_data))
    final_fidelity = float(quantum_fidelity(rho_generated, rho_data))
    final_trace_distance = float(trace_distance(rho_generated, rho_data))
    final_rank_balanced_multipole_loss = float(
        multipole_loss(
            rho_generated,
            rho_data,
            tensors,
            weighting="rank-balanced",
        )
    )
    final_high_rank_multipole_loss = float(
        multipole_loss(
            rho_generated,
            rho_data,
            tensors,
            weighting="high-rank",
        )
    )
    print(f"final Frobenius state error = {final_error:.6e}")
    print(f"final quantum fidelity = {final_fidelity:.6e}")
    print(f"final trace distance = {final_trace_distance:.6e}")

    cptp_metrics = []
    for step, theta in enumerate(params):
        U = reverse_collision_unitary(theta, generators)
        unitary_error = float(
            torch.linalg.norm(U.conj().T @ U - torch.eye(2 * d, dtype=CDTYPE))
        )
        kraus = collision_kraus_operators(U, d)
        completeness = sum(K.conj().T @ K for K in kraus)
        trace_preservation_error = float(torch.linalg.norm(completeness - I))
        choi = choi_matrix_from_kraus(kraus)
        minimum_choi_eigenvalue = float(torch.linalg.eigvalsh(choi).min())
        if unitary_error > 1e-10 or trace_preservation_error > 1e-10:
            raise AssertionError(f"reverse channel {step} is not trace preserving")
        if minimum_choi_eigenvalue < -1e-10:
            raise AssertionError(f"reverse channel {step} is not completely positive")
        cptp_metrics.append(
            {
                "unitarity_error": unitary_error,
                "trace_preservation_error": trace_preservation_error,
                "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
            }
        )

    # 7. Read out the target and generated distributions via Husimi Q.
    th, ph, Q_target, W_target = husimi_q_grid(rho_data, j)
    _, _, Q_generated, W_generated = husimi_q_grid(rho_generated, j)

    eps = 1e-14
    p = W_target.reshape(-1) + eps
    q = W_generated.reshape(-1) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    print(f"grid Jensen-Shannon divergence = {js:.6e}")

    q_grid_correlation = float(
        np.corrcoef(Q_target.reshape(-1), Q_generated.reshape(-1))[0, 1]
    )
    target_peak = np.unravel_index(np.argmax(Q_target), Q_target.shape)
    generated_peak = np.unravel_index(np.argmax(Q_generated), Q_generated.shape)

    def unit_direction(index):
        theta = th[index[0]]
        phi = ph[index[1]]
        return np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

    peak_angular_shift = float(
        np.arccos(
            np.clip(
                unit_direction(target_peak) @ unit_direction(generated_peak),
                -1.0,
                1.0,
            )
        )
    )
    print(f"Q-grid correlation = {q_grid_correlation:.6e}")
    print(f"Q peak angular shift = {peak_angular_shift:.6e} rad")

    # 8. Draw classical generated directions from Q_generated and save results.
    generated_points = sample_from_grid(
        th, ph, W_generated, n_samples=args.n_data, seed=args.seed + 1
    )
    np.save(outdir / "generated_theta_phi.npy", generated_points)
    np.save(outdir / "rho_data.npy", rho_data.detach().cpu().numpy())
    np.save(outdir / "rho_generated.npy", rho_generated.detach().cpu().numpy())
    torch.save(params.cpu(), outdir / "reverse_parameters.pt")
    np.savez(
        outdir / "forward_multipoles.npz",
        times=np.asarray(times),
        **{
            f"ell{ell}_m{m:+d}": np.asarray(
                [complex(at_time[(ell, m)]) for at_time in coefficients]
            )
            for ell, m in tensors
        },
    )

    validation_metrics = {
        "forward": {
            "initial_distance_from_maximally_mixed": initial_distance,
            "terminal_distance_from_maximally_mixed": terminal_distance,
            "max_trace_error": max(
                abs(item["trace_real"] - 1.0) for item in forward_metrics
            ),
            "max_hermiticity_error": max(
                item["hermiticity_error"] for item in forward_metrics
            ),
            "minimum_eigenvalue": min(
                item["minimum_eigenvalue"] for item in forward_metrics
            ),
            "max_multipole_decay_error": decay_error,
        },
        "reverse": {
            "training_objective": args.objective,
            "multipole_weighting": args.multipole_weighting,
            "initial_training_loss": history[0],
            "final_training_loss": history[-1],
            "final_frobenius_state_error": final_error,
            "final_quantum_fidelity": final_fidelity,
            "final_trace_distance": final_trace_distance,
            "final_rank_balanced_multipole_loss": (
                final_rank_balanced_multipole_loss
            ),
            "final_high_rank_multipole_loss": final_high_rank_multipole_loss,
            "grid_jensen_shannon_divergence": float(js),
            "q_grid_correlation": q_grid_correlation,
            "q_peak_angular_shift_radians": peak_angular_shift,
            "max_trace_error": max(
                abs(item["trace_real"] - 1.0) for item in reverse_metrics
            ),
            "max_hermiticity_error": max(
                item["hermiticity_error"] for item in reverse_metrics
            ),
            "minimum_eigenvalue": min(
                item["minimum_eigenvalue"] for item in reverse_metrics
            ),
            "max_trace_preservation_error": max(
                item["trace_preservation_error"] for item in cptp_metrics
            ),
            "minimum_choi_eigenvalue": min(
                item["minimum_choi_eigenvalue"] for item in cptp_metrics
            ),
        },
    }
    with (outdir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, indent=2)

    plot_q_map(
        th,
        ph,
        Q_target,
        "Target spherical distribution (Husimi Q)",
        outdir / "target_Q.png",
    )
    plot_q_map(
        th,
        ph,
        Q_generated,
        "Generated spherical distribution (Husimi Q)",
        outdir / "generated_Q.png",
    )
    plot_training(history, outdir / "training_loss.png")
    plot_multipole_power(times, powers, outdir / "multipole_power.png")
    plot_multipole_decay_comparison(
        times,
        powers,
        args.D,
        outdir / "multipole_decay_comparison.png",
    )

    print(f"outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
