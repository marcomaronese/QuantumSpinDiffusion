"""Independent array-only reconstruction of chart screening and confirmation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right, name, tolerance=1e-10):
    if not np.allclose(left, right, atol=tolerance, rtol=1e-11):
        raise AssertionError(f"{name}: {np.max(np.abs(np.asarray(left)-right))}")


def encode(z, description):
    radius = np.hypot(*z.T)
    s = description["length"]
    unit = np.divide(z, radius[:, None], out=np.zeros_like(z), where=radius[:, None]>0)
    if description["kind"] == "bounded":
        # Preserve the protocol's normalization order. Algebraically replacing
        # this with z/max(s,|z|) changes roundoff before sqrt(1-|u|^2) at the
        # clipped equator, creating O(sqrt(eps)) input-direction differences.
        scaled=z/s
        length=np.sqrt(np.sum(scaled*scaled,axis=1))
        clipped=scaled/np.maximum(1,length)[:,None]
        return np.c_[clipped, np.sqrt(np.maximum(0, 1-(clipped**2).sum(1)))]
    if description["kind"] == "stereographic":
        a = radius/s
        return np.c_[unit*(2*a/(1+a*a))[:, None], (1-a*a)/(1+a*a)]
    # Complementary angle formula independently verifies the stable encoder.
    delta = 4*np.arctan(np.exp(-radius/s))
    return np.c_[unit*np.sin(delta)[:, None], -np.cos(delta)]


def decode(directions, description):
    x, y, z = np.moveaxis(directions, -1, 0)
    if description["kind"] == "bounded":
        return directions[..., :2]*description["length"]
    planar = np.hypot(x, y)
    if description["kind"] == "stereographic":
        radius = description["length"]*(1-z)/planar
    else:
        delta = np.arctan2(planar, -z)
        radius = -description["length"]*np.log(np.tan(delta/4))
    return np.stack((x/planar*radius, y/planar*radius), axis=-1)


def independent_scores(reference, generated):
    angles = np.arange(32)*np.pi/32
    w1 = np.mean([wasserstein_distance(reference @ [np.cos(a), np.sin(a)],
                                       generated @ [np.cos(a), np.sin(a)]) for a in angles])
    x, y = reference[:1024], generated[:1024]
    energy = 2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean()
    return float(w1), float(energy)


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(output / name) != expected:
            raise AssertionError(f"changed artifact: {name}")
    metadata = json.loads((output / "metadata.json").read_text())
    config = metadata["config"]
    rows = json.loads((output / "results.json").read_text())
    worst_decode = 0.
    for row in rows:
        data = np.load(output / row["data_directory"] / "data.npz")
        noise = np.load(output / row["noise"])
        saved = np.load(output / row["artifact"])
        calibration, reference = data["calibration"], data["reference"]
        location, scale = calibration.mean(0), calibration.std(0)
        standardized = (calibration-location)/scale
        description = row["map"]
        incoming = encode(standardized, description)[noise["indices"]]
        axes = np.eye(3)[np.argmin(np.abs(incoming), axis=-1)]
        tangent = np.cross(incoming, axes)
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
        bitangent = np.cross(incoming, tangent)
        cosine = 2*np.exp(np.log(noise["radial"])/(row["two_j"]+1))-1
        sine = np.sqrt(np.maximum(0, 1-cosine**2))
        phi = 2*np.pi*noise["azimuth"]
        observed = cosine[...,None]*incoming+sine[...,None]*(np.cos(phi)[...,None]*tangent+np.sin(phi)[...,None]*bitangent)
        expected = location+scale*decode(observed, description)
        worst_decode = max(worst_decode, float(np.max(np.abs(expected-saved["decoded"]))))
        close(expected, saved["decoded"], "independent POVM and inverse", tolerance=1e-9)
        close(np.linalg.norm(observed, axis=-1), 1, "unit POVM outcomes")
        values = (saved["decoded"]-location)/scale
        real = (reference-location)/scale
        for batch, scores in zip(values, row["batch_scores"], strict=True):
            w1, energy = independent_scores(real, batch)
            close(w1, scores["sliced_w1"], "batch SW1")
            close(energy, scores["energy_v"], "batch energy")
        close(np.mean([s["sliced_w1"] for s in row["batch_scores"]]), row["sliced_w1"], "mean SW1")
        radii = np.linalg.norm(values.reshape(-1,2), axis=1)
        quantiles = np.quantile(radii,[.5,.95,.995])
        close(quantiles, row["radius_quantiles"], "radial quantiles")
        close(quantiles[-1]/np.quantile(np.linalg.norm(real,axis=1),.995), row["tail_ratio_995"], "tail guardrail")
        rho = saved["rho"]
        close(np.trace(rho), 1, "state trace")
        close(rho,rho.conj().T,"state Hermiticity")
        if np.linalg.eigvalsh(rho).min() < -1e-10:
            raise AssertionError("nonpositive encoded state")
        # The south-pole overlap is ((1-m_z)/2)^N for each input component.
        all_incoming = encode(standardized, description)
        n=row["two_j"]
        phase=np.arctan2(all_incoming[:,1],all_incoming[:,0])
        c=np.sqrt(np.maximum(0,(1+all_incoming[:,2])/2))
        s=np.sqrt(np.maximum(0,(1-all_incoming[:,2])/2))
        vectors=np.column_stack([math.sqrt(math.comb(n,k))*c**(n-k)*s**k*np.exp(1j*k*phase)
                                 for k in range(n+1)])
        close(vectors.T@vectors.conj()/len(vectors),rho,"independent encoded density")
        pole = (row["two_j"]+1)/(4*np.pi)*np.mean(((1-all_incoming[:,2])/2)**row["two_j"])
        close(pole,row["pole_husimi_density"],"pole density")
        close(pole,(row["two_j"]+1)/(4*np.pi)*rho[-1,-1].real,"encoded-state pole density")
        if row["population_second_moment_finite"] != (description["kind"]!="stereographic"):
            raise AssertionError("moment-existence declaration")
        dot = np.sum(observed*incoming,axis=-1)
        close(dot,cosine,"exact coherent sampling angle")
        close(dot.mean(),row["mean_dot"],"angular mean")
        close((dot**2).mean(),row["mean_dot_squared"],"angular second moment")
    selection = json.loads((output / "selection.json").read_text())
    if config["phase"] == "screen":
        angular=np.load(output/"angular_control.npz")
        calculated=[]
        for block in np.array_split(angular["grid"],32):
            calculated.extend(6/(4*np.pi)*np.mean(((1+block@angular["input_directions"].T)/2)**5,axis=1))
        close(np.asarray(calculated).reshape(angular["q"].shape),angular["q"],"angular reference kernel")
        for target, selected in selection.items():
            group = [r for r in rows if r["target"]==target and r["two_j"]==5]
            for kind in ("stereographic","sinh_stereographic"):
                available = {r["map"]["name"]:r["map"] for r in group if r["map"]["kind"]==kind}
                means = {name:np.mean([r["sliced_w1"] for r in group if r["map"]["name"]==name]) for name in available}
                if kind=="sinh_stereographic":
                    guarded={name:description for name,description in available.items()
                             if all(r["tail_ratio_995"]<=1.5 for r in group if r["map"]["name"]==name)}
                    available=guarded or available
                chosen = min(available,key=lambda n:(means[n],available[n]["length"]))
                if available[chosen] not in selected["selected"]:
                    raise AssertionError("validation-only scale selection")
            candidate = selected["selected"][-1]
            candidate_rows = [r for r in group if r["map"]==candidate]
            baseline = np.mean([r["sliced_w1"] for r in group if r["map"]["kind"]=="bounded"])
            improvement = 1-np.mean([r["sliced_w1"] for r in candidate_rows])/baseline
            eligible = improvement>=.1 and all(r["tail_ratio_995"]<=1.5 for r in candidate_rows)
            if bool(eligible)!=selected["candidate_eligible"]:
                raise AssertionError("screen eligibility")
    else:
        if digest(output / "selection.json") != metadata["selection_sha256"]:
            raise AssertionError("selection changed before confirmation")
        for row in rows:
            if row["map"] not in selection[row["target"]]["selected"]:
                raise AssertionError("unselected map reached confirmation")
    # Exact rotation covariance of the kernel, including both arguments.
    rng = np.random.default_rng(91831)
    first, second = rng.normal(size=(2,32,3))
    first/=np.linalg.norm(first,axis=1,keepdims=True)
    second/=np.linalg.norm(second,axis=1,keepdims=True)
    rotation=Rotation.from_rotvec([.4,-.7,.2]).as_matrix()
    kernel = lambda a,b: ((1+a@b.T)/2)**5
    rotation_residual=float(np.max(np.abs(kernel(first,second)-kernel(first@rotation.T,second@rotation.T))))
    if rotation_residual>1e-12:
        raise AssertionError("rotation covariance")
    result={"passed":True,"cases":len(rows),"hashed_artifacts":len(manifest),
            "maximum_decoding_replay_error":worst_decode,"kernel_rotation_residual":rotation_residual,
            "manifest_sha256":digest(output / "manifest.json")}
    (output / "audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.study_dir.resolve()),indent=2))
