"""Read-only checks of frozen inputs; writes derived audit.json and CSV.

State metrics are independently evaluated with NumPy. Tensor construction,
peak detection and channel replay reuse the exact archived implementation;
these are explicitly consistency checks, not independent implementations.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spin-paper-matplotlib')
import torch

PAPER = Path(__file__).resolve().parents[1]
RAW = PAPER / 'evidence/raw'
SOURCE = PAPER / 'evidence/source/diagnostic'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'experiments'))
import spin_diffusion_toy as toy
from spin_multipoles import irreducible_spherical_tensors
from spin_reverse import build_reverse_generators, collision_kraus_operators, parameter_schedule, reverse_collision_unitary
from validation_study import significant_q_modes


def read(path):
    return json.loads(path.read_text())


@lru_cache(None)
def grid(j):
    th = np.linspace(1e-5, np.pi - 1e-5, 80)
    ph = np.linspace(0, 2*np.pi, 160, endpoint=False)
    t, p = np.meshgrid(th, ph, indexing='ij')
    n = round(2*j)
    psi = np.stack([math.sqrt(math.comb(n,k))*np.cos(t/2)**(n-k)*np.sin(t/2)**k*np.exp(1j*k*p)
                    for k in range(n+1)], axis=-1)
    return th, ph, psi


def q_grid(rho, j):
    _, _, psi = grid(j)
    return (2*j+1)/(4*np.pi)*np.einsum('abi,ij,abj->ab', psi.conj(), rho, psi).real


def state_metrics(target, generated):
    w,v = np.linalg.eigh((target+target.conj().T)/2)
    root = (v*np.sqrt(np.maximum(w,0)))@v.conj().T
    sand = root@generated@root
    fidelity = np.sqrt(np.maximum(np.linalg.eigvalsh((sand+sand.conj().T)/2),0)).sum()**2
    delta = generated-target
    return {'frobenius_error': float(np.vdot(delta,delta).real),
            'trace_distance': float(np.abs(np.linalg.eigvalsh((delta+delta.conj().T)/2)).sum()/2),
            'infidelity': float(1-fidelity)}


def q_metrics(target, generated, j):
    th,ph,_ = grid(j)
    a,b = q_grid(target,j),q_grid(generated,j)
    p,q = a*np.sin(th)[:,None],b*np.sin(th)[:,None]
    p,q = p/p.sum()+1e-14,q/q.sum()+1e-14
    p,q = p/p.sum(),q/q.sum()
    mid = (p+q)/2
    js = float((p*np.log(p/mid)+q*np.log(q/mid)).sum()/2)
    modes_a,modes_b = significant_q_modes(th,ph,a),significant_q_modes(th,ph,b)
    def vec(m):
        t,p=m['theta'],m['phi']
        return np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    count = 0
    if modes_a and modes_b:
        dist = np.arccos(np.clip(np.array([vec(m) for m in modes_a])@np.array([vec(m) for m in modes_b]).T,-1,1))
        i,k=linear_sum_assignment(dist)
        count=int(np.sum(dist[i,k]<=.35))
    return {'q_js_divergence':js,'q_correlation':float(np.corrcoef(a.ravel(),b.ravel())[0,1]),
            'target_mode_count':len(modes_a),'generated_mode_count':len(modes_b),
            'all_target_modes_recovered':count==len(modes_a) and len(modes_a)>0}


def main():
    torch.set_num_threads(1)
    manifest=read(PAPER/'evidence/manifest.json')
    for f in manifest['files']:
        assert hashlib.sha256((PAPER/f['path']).read_bytes()).hexdigest()==f['sha256'], f['path']
    rows=[]
    largest_metric_delta=0.0
    max_trace=max_herm=max_decay=max_replay=max_kraus=0.0
    min_eig=1.0
    basis=irreducible_spherical_tensors(2.5,*toy.spin_operators(2.5)[:2])
    generators=build_reverse_generators(*toy.spin_operators(2.5))
    groups=[('diagnostic',RAW/'diagnostic_study'),('validation',RAW/'hybrid_study/validation'),('test',RAW/'hybrid_study/test')]
    for split,root in groups:
        declared=read(root/'results.json')
        for old in declared:
            # Recover paths portably, retaining only condition and seed suffix.
            suffix=Path(old['run_directory']).parts[-2:]
            run=root/'runs'/Path(*suffix)
            cfg=read(run/'config.json')
            target=np.load(run/'rho_data.npy'); generated=np.load(run/'rho_generated.npy')
            values=state_metrics(target,generated)|q_metrics(target,generated,2.5)
            for key in ('frobenius_error','trace_distance','infidelity','q_js_divergence','q_correlation'):
                diff=abs(values[key]-old[key]); largest_metric_delta=max(largest_metric_delta,diff)
                assert diff<1e-9,(run,key,diff)
            assert values['all_target_modes_recovered']==old['all_target_modes_recovered'],run
            assert values['target_mode_count']==old['target_mode_count'],run
            assert values['generated_mode_count']==old['generated_mode_count'],run
            rank_errors={}
            for ell in range(6):
                rank_errors[ell]=sum(abs(np.trace((generated-target)@t.numpy().conj().T))**2
                                     for (rank,m),t in basis.items() if rank==ell)
                assert abs(rank_errors[ell]-old['ranks'][str(ell)]['squared_coefficient_error'])<1e-12
            assert abs(sum(rank_errors.values())-values['frobenius_error'])<1e-12
            for filename in ('forward_states.npy','reverse_states.npy','rho_generated_mixed.npy'):
                for rho in np.load(run/filename).reshape(-1,6,6):
                    max_trace=max(max_trace,float(abs(np.trace(rho)-1)))
                    max_herm=max(max_herm,float(np.linalg.norm(rho-rho.conj().T)))
                    min_eig=min(min_eig,float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()))
            states=np.load(run/'forward_states.npy')
            for (ell,m),tensor in basis.items():
                t=tensor.numpy()
                c=np.einsum('tij,ji->t',states,t.conj().T)
                expected=c[0]*np.exp(-cfg['diffusion_rate']*ell*(ell+1)*np.arange(7)*cfg['time_step'])
                max_decay=max(max_decay,float(np.max(np.abs(c-expected))))
            params=torch.load(run/'reverse_parameters.pt',map_location='cpu',weights_only=True)
            prior=torch.tensor(states[-1]) if cfg['starting_state']=='forward' else torch.eye(6,dtype=torch.complex128)/6
            replay=toy.generate_density(prior,params,generators).numpy()
            max_replay=max(max_replay,float(np.linalg.norm(replay-generated)))
            for step in parameter_schedule(params,n_steps=6):
                ks=collision_kraus_operators(reverse_collision_unitary(step,generators),6)
                residual=torch.linalg.norm(sum(k.conj().T@k for k in ks)-torch.eye(6,dtype=torch.complex128))
                max_kraus=max(max_kraus,float(residual))
            rows.append({'split':split,'seed':cfg['seed'],'starting_state':cfg['starting_state'],
                         'supervision':cfg['supervision'],'epochs':cfg['epochs'],'hybrid_lambda':cfg['hybrid_lambda'],
                         **values,'rank45_squared_error':float(rank_errors[4]+rank_errors[5]),
                         'run':str(run.relative_to(PAPER))})
    assert len(rows)==70
    assert max(max_trace,max_herm,max_decay,max_replay,max_kraus)<1e-10
    assert min_eig>=-1e-10
    # Independently implemented two-spin metrics, isolated from training code.
    spec=importlib.util.spec_from_file_location('two_audit',PAPER/'evidence/source/current/experiments/validate_two_spin.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    states=np.load(RAW/'two_spin_pilot/density_matrices.npz')
    names=['factorized','riemannian_heat_kernel','tensor_mixture','quantum_no_direct','quantum_direct','neural']
    two_delta=0.0
    for name,declared in zip(names,read(RAW/'two_spin_pilot/summary.json')['models'],strict=True):
        for k,v in module.metrics(states[name],states['target_test']).items():
            two_delta=max(two_delta,abs(v-declared[k]))
    assert two_delta<1e-9
    out=PAPER/'results';out.mkdir(exist_ok=True)
    with (out/'audited_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'passed':True,'hashed_files':len(manifest['files']),'single_spin_runs':len(rows),
            'two_spin_models':len(names),'maximum_state_metric_discrepancy':largest_metric_delta,
            'maximum_two_spin_metric_discrepancy':two_delta,'maximum_trace_residual':max_trace,
            'maximum_hermiticity_residual':max_herm,'minimum_state_eigenvalue':min_eig,
            'maximum_forward_decay_error':max_decay,'maximum_saved_state_replay_error':max_replay,
            'maximum_kraus_completeness_error':max_kraus,
            'mode_recovery_and_rank_errors_agree':True,
            'scope_notes':['NumPy state metrics are independent of training losses.',
                           'Tensor basis, peak finding and channel replay reuse historical source.',
                           'Historical instrument ensemble summaries were preserved, not rerun.',
                           'Historical j=2 objective sweep is retained as supplementary evidence, not independently re-audited here.']}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
