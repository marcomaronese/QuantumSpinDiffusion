"""Independent NumPy/SciPy replay of the saved Euclidean benchmark."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right, label, tolerance=1e-10):
    if not np.allclose(left, right, rtol=0, atol=tolerance):
        raise AssertionError(f"{label}: maximum difference {np.max(np.abs(np.asarray(left)-right))}")


def chart_encode(values, description):
    centered = (values-np.asarray(description["location"])) / description["scale"]
    scaled = centered / description["radius"]
    radii = np.sqrt(np.sum(scaled**2, axis=1))
    planar = np.zeros((len(values), 2))
    planar[:, :values.shape[1]] = scaled / np.maximum(1, radii)[:, None]
    return np.c_[planar, np.sqrt(np.maximum(0, 1-np.sum(planar**2, axis=1)))], radii > 1


def kets(directions, two_j):
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    return np.column_stack([math.sqrt(math.comb(two_j, k))*np.cos(theta/2)**(two_j-k)
                            *np.sin(theta/2)**k*np.exp(1j*k*phi) for k in range(two_j+1)])


def density(directions, two_j):
    vectors = kets(directions, two_j)
    return np.einsum("ni,nj->ij", vectors, vectors.conj()) / len(vectors)


def povm(two_j, polar, azimuth):
    nodes, weights = np.polynomial.legendre.leggauss(polar)
    z = np.repeat(nodes, azimuth)
    phi = np.tile(2*np.pi*np.arange(azimuth)/azimuth, polar)
    directions = np.c_[np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z]
    frame = kets(directions, two_j)*np.sqrt(np.repeat(weights, azimuth)*(two_j+1)/(2*azimuth))[:, None]
    close(frame.T @ frame.conj(), np.eye(two_j+1), "POVM completeness")
    return directions, frame


def born(rho, frame):
    return np.einsum("ki,ij,kj->k", frame.conj(), rho, frame).real


def trace_distance(left, right):
    return .5*np.abs(np.linalg.eigvalsh(left-right)).sum()


def histogram(values, dimensions, weights=None):
    edges = np.r_[-np.inf, np.linspace(-3, 3, (24 if dimensions == 1 else 12)+1), np.inf]
    values = np.histogramdd(values, bins=[edges]*dimensions, weights=weights)[0]
    return values / values.sum()


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in manifest.items():
        if sha256(output/path) != expected:
            raise AssertionError(f"artifact changed: {path}")
    metadata = json.loads((output / "metadata.json").read_text())
    config = metadata["config"]
    rows = json.loads((output / "prepared.json").read_text())
    directions, frame = povm(config["two_j"], *config["grid"])
    fine_directions, fine_frame = povm(config["two_j"], *config["fine_grid"])
    maximum_replay_error = 0.
    minimum_eigenvalue = 1.
    for row in rows:
        directory = output / row["directory"]
        data = dict(np.load(directory / "data.npz"))
        fit = dict(np.load(directory / "fit.npz"))
        saved = dict(np.load(directory / "evaluation.npz"))
        report = json.loads((directory / "metrics.json").read_text())
        description = json.loads((directory / "chart.json").read_text())
        if sha256(directory / "data.npz") != row["data_sha256"]:
            raise AssertionError("dataset identity changed")
        if sha256(directory / "chart.json") != row["chart_sha256"]:
            raise AssertionError("chart changed after preparation")
        if sha256(directory / "fit.npz") != report["resources"]["fit_sha256"]:
            raise AssertionError("checkpoint changed before test evaluation")
        if sha256(directory / "classical.pt") != report["resources"]["classical_checkpoint_sha256"]:
            raise AssertionError("classical checkpoint changed before test evaluation")
        close(description["location"], data["train"].mean(0), "training-only location")
        close(description["scale"], data["train"].std(0), "training-only scale")
        states = {}
        for split in ("train", "validation", "test"):
            transformed, clipped = chart_encode(data[split], description)
            states[split] = density(transformed, config["two_j"])
            close(states[split], saved[f"rho_{split}"], f"encoded {split}")
            close(clipped.mean(), report["clipped_fractions"][split], "clipping fraction")
        close(fit["forward"][0], states["train"], "forward input")
        size = config["two_j"]+1
        rho = np.eye(size, dtype=complex)/size
        close(rho, fit["reverse"][0], "maximally mixed input")
        for index, controls in enumerate(fit["parameters"]):
            unitary = np.eye(2*size, dtype=complex)
            for layer in controls:
                for angle, generator in zip(layer, fit["generators"], strict=True):
                    unitary = expm(-1j*angle*generator) @ unitary
            kraus = [unitary.reshape(size, 2, size, 2)[:, k, :, 0] for k in range(2)]
            close(sum(k.conj().T @ k for k in kraus), np.eye(size), "Kraus completeness")
            rho = sum(k @ rho @ k.conj().T for k in kraus)
            error = float(np.max(np.abs(rho-fit["reverse"][index+1])))
            maximum_replay_error = max(maximum_replay_error, error)
            close(rho, fit["reverse"][index+1], "independent channel replay")
        # Independent spin operators and heat superoperator, not the training helper.
        j = config["two_j"]/2
        plus = np.zeros((size, size), dtype=complex)
        for k in range(1, size):
            m = j-k
            plus[k-1, k] = np.sqrt((j-m)*(j+m+1))
        operators = ((plus+plus.conj().T)/2, (plus-plus.conj().T)/(2j), np.diag(j-np.arange(size)))
        liouvillian = np.zeros((size*size, size*size), dtype=complex)
        for operator in operators:
            commutator = np.kron(np.eye(size), operator)-np.kron(operator.T, np.eye(size))
            liouvillian -= commutator @ commutator
        eigenvalues = sorted(-ell*(ell+1) for ell in range(config["two_j"]+1) for _ in range(2*ell+1))
        close(np.linalg.eigvalsh(liouvillian), eigenvalues, "analytic multipole heat rates")
        forward = states["train"]
        heat = expm(.35*liouvillian)
        for expected in fit["forward"][1:]:
            forward = (heat @ forward.flatten(order="F")).reshape((size, size), order="F")
            close(forward, expected, "independent heat replay")
        populations = {"model": rho, "encoded_train": states["train"], "encoded_test": states["test"],
                       "mixed": np.eye(size)/size,
                       "classical_common": density(fit["classical_directions"], config["two_j"])}
        close(populations["classical_common"], fit["classical_density"], "DDPM common readout input")
        for state in [*states.values(), *populations.values(), *fit["forward"], *fit["reverse"]]:
            close(state.trace(), 1., "state trace")
            close(state, state.conj().T, "state Hermiticity")
            eigenvalue = np.linalg.eigvalsh(state).min()
            minimum_eigenvalue = min(minimum_eigenvalue, float(eigenvalue))
            if eigenvalue < -1e-10:
                raise AssertionError("nonphysical state")
        decoded = np.asarray(description["location"]) + description["radius"]*np.asarray(description["scale"])*directions[:, :data["train"].shape[1]]
        fine_decoded = np.asarray(description["location"]) + description["radius"]*np.asarray(description["scale"])*fine_directions[:, :data["train"].shape[1]]
        for name, state in populations.items():
            probability = born(state, frame)
            close(probability, saved[f"born_{name}"], "Born probabilities")
            close(probability.sum(), 1., "Born normalization")
            close(decoded[saved[f"outcomes_{name}"]], saved[f"samples_{name}"], "POVM-to-sample decoding")
            index = config["conditions"].index(name)
            rng = np.random.default_rng(row["base"]+60000+index)
            close(rng.choice(len(probability), config["n_eval"], p=probability),
                  saved[f"outcomes_{name}"], "independent Born draws", tolerance=0)
            fine_probability = born(state, fine_frame)
            coarse_h = histogram((decoded-description["location"])/description["scale"],
                                 data["train"].shape[1], probability)
            fine_h = histogram((fine_decoded-description["location"])/description["scale"],
                               data["train"].shape[1], fine_probability)
            close(.5*np.abs(coarse_h-fine_h).sum(), report["grid_refinement"][name]["binned_tv"], "grid binned TV")
            close(np.max(np.abs(probability @ decoded-fine_probability @ fine_decoded)),
                  report["grid_refinement"][name]["mean_max_abs"], "grid moments")
            for split in ("validation", "test"):
                if name == "encoded_test":
                    continue
                target_born = born(states[split], frame)
                close(.5*np.abs(probability-target_born).sum(), report["state_metrics"][f"{name}_{split}"]["born_tv"], "Born TV")
                close(trace_distance(state, states[split]), report["state_metrics"][f"{name}_{split}"]["trace_distance"], "trace distance")
        reference = (data["test"]-description["location"]) / description["scale"]
        for name in config["conditions"]:
            generated = (saved[f"samples_{name}"]-description["location"]) / description["scale"]
            if name not in populations:
                rng = np.random.default_rng(row["base"]+60000+config["conditions"].index(name))
                if name == "real_reference":
                    expected = data["reference"]
                elif name == "classical_direct":
                    expected = np.asarray(description["location"])+description["radius"]*np.asarray(description["scale"])*fit["classical_directions"][:, :reference.shape[1]]
                else:
                    expected = data["train"][rng.integers(len(data["train"]), size=config["n_eval"])]
                    if name == "clipped_bootstrap":
                        transformed, _ = chart_encode(expected, description)
                        expected = np.asarray(description["location"])+description["radius"]*np.asarray(description["scale"])*transformed[:, :reference.shape[1]]
                close(expected, saved[f"samples_{name}"], "independent classical/reference samples")
            generated_h = (histogram((decoded-description["location"])/description["scale"], reference.shape[1], saved[f"born_{name}"])
                           if name in populations else histogram(generated, reference.shape[1]))
            close(.5*np.abs(histogram(reference, reference.shape[1])-generated_h).sum(),
                  report["metrics"][name]["raw_binned_tv"], "raw binned TV")
            if reference.shape[1] == 1:
                discrepancy = wasserstein_distance(reference[:, 0], generated[:, 0])
            else:
                discrepancy = np.mean([wasserstein_distance(reference @ np.array([np.cos(angle), np.sin(angle)]),
                                                            generated @ np.array([np.cos(angle), np.sin(angle)]))
                                       for angle in np.arange(32)*np.pi/32])
            close(discrepancy, report["metrics"][name]["sliced_w1"], "independent W1")
            x, y = reference[:config["energy_count"]], generated[:config["energy_count"]]
            energy = 2*np.sum(cdist(x, y))/(len(x)*len(y)) - np.sum(cdist(x, x))/len(x)**2 - np.sum(cdist(y, y))/len(y)**2
            close(energy, report["metrics"][name]["energy_v"], "independent energy")
    summary = json.loads((output / "summary.json").read_text())["targets"]
    for target, group in summary.items():
        reports = [json.loads((output / row["directory"] / "metrics.json").read_text()) for row in rows if row["target"] == target]
        for condition, metrics in group.items():
            for name, stats in metrics.items():
                values = [report["metrics"][condition][name] for report in reports]
                close(np.mean(values), stats["mean"], "aggregate mean")
                close(np.std(values, ddof=1) if len(values)>1 else 0, stats["sd"], "aggregate SD")
    result = {"passed": True, "artifact_hashes": len(manifest), "fits": len(rows),
              "maximum_channel_replay_error": maximum_replay_error,
              "minimum_state_eigenvalue": minimum_eigenvalue,
              "manifest_sha256": sha256(output / "manifest.json")}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.study_dir.resolve()), indent=2))
