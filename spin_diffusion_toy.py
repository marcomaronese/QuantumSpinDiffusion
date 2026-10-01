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
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
import matplotlib.pyplot as plt


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

    rho = rho / len(points)
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / torch.trace(rho)
    return rho


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
    # Column-vectorization implemented as transpose + row-major flatten.
    vec = rho.T.contiguous().reshape(-1)
    out = S @ vec
    out = out.reshape(rho.shape).T.contiguous()
    out = 0.5 * (out + out.conj().T)
    out = out / torch.trace(out)
    return out


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
    out = joint[:, 0, :, 0] + joint[:, 1, :, 1]
    return 0.5 * (out + out.conj().T)


def reverse_collision_channel(rho, theta, generators):
    """
    theta shape: [layers, n_generators].

    Append |0><0|_a, apply U_theta, trace the ancilla.
    """
    d = rho.shape[0]
    dim = 2 * d
    U = torch.eye(dim, dtype=CDTYPE)

    for layer in range(theta.shape[0]):
        for k, G in enumerate(generators):
            Uk = torch.matrix_exp(-1j * theta[layer, k] * G)
            U = Uk @ U

    anc0 = torch.tensor([[1, 0], [0, 0]], dtype=CDTYPE)
    joint_in = torch.kron(rho.contiguous(), anc0)
    joint_out = U @ joint_in @ U.conj().T

    rho_out = partial_trace_ancilla(joint_out, d)
    rho_out = rho_out / torch.trace(rho_out)
    return rho_out


def frobenius_loss(a, b):
    diff = a - b
    return torch.real(torch.sum(diff.conj() * diff))


def train_reverse(
    forward_states,
    generators,
    layers,
    epochs,
    lr,
    seed,
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
            loss = loss + frobenius_loss(rho, forward_states[t - 1])

        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            final_err = float(frobenius_loss(rho, forward_states[0]).detach())
            print(
                f"epoch={epoch:5d}  "
                f"path_loss={float(loss.detach()):.6e}  "
                f"final_state_error={final_err:.6e}"
            )

    return params.detach(), history


@torch.no_grad()
def generate_density(prior, params, generators):
    rho = prior
    for step in range(params.shape[0]):
        rho = reverse_collision_channel(rho, params[step], generators)
    return rho


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
    weights = np.maximum(weights, 0.0)
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

    prior = I / d
    terminal_distance = float(torch.linalg.norm(forward_states[-1] - prior))
    print(f"||rho_T - I/d||_F = {terminal_distance:.6e}")

    # 4. Train the reverse quantum collision channels.
    generators = build_reverse_generators(Jx, Jy, Jz, I)
    params, history = train_reverse(
        forward_states=forward_states,
        generators=generators,
        layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    # 5. Generate from the maximally mixed prior.
    rho_generated = generate_density(prior, params, generators)
    final_error = float(frobenius_loss(rho_generated, rho_data))
    print(f"final Frobenius state error = {final_error:.6e}")

    # 6. Read out the target and generated distributions via Husimi Q.
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

    # 7. Draw classical generated directions from Q_generated.
    generated_points = sample_from_grid(
        th, ph, W_generated, n_samples=args.n_data, seed=args.seed + 1
    )
    np.save(outdir / "generated_theta_phi.npy", generated_points)
    np.save(outdir / "rho_data.npy", rho_data.detach().cpu().numpy())
    np.save(outdir / "rho_generated.npy", rho_generated.detach().cpu().numpy())
    torch.save(params.cpu(), outdir / "reverse_parameters.pt")

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

    print(f"outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
