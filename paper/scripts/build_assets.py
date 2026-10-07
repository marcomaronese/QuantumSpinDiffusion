"""Generate all paper plots and LaTeX tables from frozen evidence.

No experiment training, source mutation, or hand-entered numerical outcomes.
"""
from pathlib import Path
import csv
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/spin-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

PAPER=Path(__file__).resolve().parents[1]
RAW=PAPER/'evidence/raw'
FIG=PAPER/'figures'; TAB=PAPER/'tables'; RESULT=PAPER/'results'
for p in (FIG,TAB,RESULT): p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'axes.titleweight':'bold',
                    'savefig.bbox':'tight','figure.dpi':130})
TEAL='#147d83'; ORANGE='#c26335'; NAVY='#193954'; GRAY='#78828b'


def read(path): return json.loads(path.read_text())
def mean(rows,key): return float(np.mean([r[key] for r in rows]))
def sci(value,digits=2):
    if value==0: return '0'
    mant,exp=f'{value:.{digits}e}'.split('e')
    return rf'{mant}\times10^{{{int(exp)}}}'
def stats(rows,key):
    a=np.array([r[key] for r in rows]); return a.mean(),a.std(ddof=1)
def pm(rows,key):
    a,b=stats(rows,key)
    exp=int(math.floor(math.log10(a))) if a else 0
    if exp < -2:
        return rf'$({a/10**exp:.2f}\pm{b/10**exp:.2f})\,10^{{{exp}}}$'
    return rf'${a:.4f}\pm{b:.4f}$'
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf');fig.savefig(FIG/f'{name}.png',dpi=180);plt.close(fig)
def table(name,columns,header,rows):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(r)+' \\\\' for r in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (TAB/f'{name}.tex').write_text(text)


def main():
    audit=read(RESULT/'audit.json');assert audit['passed']
    with (RESULT/'audited_runs.csv').open() as f: raw_rows=list(csv.DictReader(f))
    rows=[]
    for r in raw_rows:
        for k in ('seed','epochs'): r[k]=int(r[k])
        for k in ('hybrid_lambda','frobenius_error','trace_distance','infidelity','q_js_divergence','q_correlation','rank45_squared_error'): r[k]=float(r[k])
        r['all_target_modes_recovered']=r['all_target_modes_recovered']=='True'
        rows.append(r)
    diag=[r for r in rows if r['split']=='diagnostic']
    def condition(start,sup,epochs):
        return sorted([r for r in diag if r['starting_state']==start and r['supervision']==sup and r['epochs']==epochs],key=lambda r:r['seed'])
    output=[];summary=[]
    for epochs in (100,500):
        for start in ('mixed','forward'):
            for sup in ('path','final'):
                group=condition(start,sup,epochs)
                count=sum(r['all_target_modes_recovered'] for r in group)
                output.append([r'$\id/d$' if start=='mixed' else r'$\rho_6$',sup,str(epochs),f'{count}/5',pm(group,'infidelity'),pm(group,'trace_distance')])
                summary.append({'start':start,'supervision':sup,'epochs':epochs,'mode_successes':count,
                                **{k:{'mean':stats(group,k)[0],'sample_sd':stats(group,k)[1]} for k in ('infidelity','trace_distance','q_js_divergence')}})
    table('diagnostic','lllcrr',r'Start & Loss & Epochs & Modes & Infidelity & Trace distance',output)
    output=[]
    for split in ('validation','test'):
        for lam in sorted({r['hybrid_lambda'] for r in rows if r['split']==split}):
            group=[r for r in rows if r['split']==split and r['hybrid_lambda']==lam]
            output.append(['Validation' if split=='validation' else 'Confirmation',f'{lam:g}']+
                          ['$'+sci(mean(group,k))+'$' for k in ('infidelity','trace_distance','q_js_divergence','rank45_squared_error')])
    table('hybrid','llrrrr',r'Split & $\lambda$ & Infidelity & Trace distance & JS (nats) & $E_{45}$',output)
    models=read(RAW/'two_spin_pilot/summary.json')['models']
    short=['Factorized marginals','Spherical kernel','Separable mixture','Quantum, no direct','Quantum, direct','Neural generator']
    params=['16 e','0 t','33 e','33 t','39 t','39 t']
    table('two_spin','lrrrr',r'Model & Parameters & Trace dist. & MI error & Corr. error',
          [[name,p]+[f'{m[k]:.4f}' for k in ('trace_distance','mutual_information_absolute_error','connected_correlation_error')]
           for name,p,m in zip(short,params,models,strict=True)])

    test0=sorted([r for r in rows if r['split']=='test' and r['hybrid_lambda']==0],key=lambda r:r['seed'])
    testh=sorted([r for r in rows if r['split']=='test' and r['hybrid_lambda']==.75],key=lambda r:r['seed'])
    reductions={k:100*(1-mean(testh,k)/mean(test0,k)) for k in ('rank45_squared_error','trace_distance')}
    numbers={'HybridRankReduction':f'{reductions["rank45_squared_error"]:.1f}',
             'HybridTraceReduction':f'{reductions["trace_distance"]:.1f}'}
    for macro,key in {'AuditDecay':'maximum_forward_decay_error','AuditTrace':'maximum_trace_residual',
                      'AuditHerm':'maximum_hermiticity_residual','AuditMinEigen':'minimum_state_eigenvalue',
                      'AuditKraus':'maximum_kraus_completeness_error','AuditTwoDelta':'maximum_two_spin_metric_discrepancy',
                      'AuditMetricDelta':'maximum_state_metric_discrepancy'}.items():
        numbers[macro]='$'+sci(audit[key])+'$'
    (TAB/'numbers.tex').write_text('% Generated by scripts/build_assets.py; do not edit.\n'+
                                  '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n')

    fig,axes=plt.subplots(1,2,figsize=(10,3.35),layout='constrained')
    with (RAW/'validation_study/representation_sweep.csv').open() as f: rep=list(csv.DictReader(f))
    axes[0].plot([float(r['j']) for r in rep],[int(r['significant_mode_count']) for r in rep],'-o',color=TEAL)
    axes[0].set(xlabel='Spin j',ylabel='Significant target maxima',yticks=[1,2],ylim=(.8,2.2),title='A  Encoded resolution')
    axes[0].axvline(2.5,color=GRAY,ls=':',lw=1)
    a=np.load(RAW/'diagnostic_study/runs/mixed_final_e100_fidelity_l0.0/seed_7/forward_multipoles.npz')
    colors=plt.get_cmap('viridis')(np.linspace(.08,.87,5))
    for ell,col in zip(range(1,6),colors):
        power=sum(np.abs(a[f'ell{ell}_m{m:+d}'])**2 for m in range(-ell,ell+1))
        expected=power[0]*np.exp(-2*ell*(ell+1)*a['times'])
        keep=expected>1e-28
        axes[1].plot(a['times'][keep],expected[keep],color=col,lw=1.3,label=rf'$\ell={ell}$')
        axes[1].scatter(a['times'][keep],power[keep],color=col,s=18)
    axes[1].set(yscale='log',xlabel='Diffusion time (D = 1)',ylabel=r'Multipole power $P_\ell$',title='B  Exact forward spectrum')
    axes[1].legend(fontsize=8,ncol=2,loc='lower left');save(fig,'physics')

    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    groups=[condition('mixed',s,e) for e in (100,500) for s in ('path','final')]
    for i,group in enumerate(groups):
        col=ORANGE if i%2==0 else TEAL
        axes[0].scatter(i+np.linspace(-.12,.12,5),[r['infidelity'] for r in group],color=col,s=25)
        axes[0].plot([i-.2,i+.2],[mean(group,'infidelity')]*2,color=NAVY,lw=2)
        axes[0].text(i,.13,f'{sum(r["all_target_modes_recovered"] for r in group)}/5',ha='center',fontsize=9)
    axes[0].set(yscale='log',ylim=(2e-6,.22),xticks=range(4),xticklabels=['Path\n100','Final\n100','Path\n500','Final\n500'],
                ylabel='Infidelity',xlabel='Supervision / epochs',title='A  Paired seed outcomes')
    original=read(RAW/'diagnostic_study/results.json')
    for sup,epochs,col,style in [('path',100,ORANGE,'-'),('final',100,TEAL,'-'),('path',500,ORANGE,'--'),('final',500,TEAL,'--')]:
        group=[r for r in original if r['starting_state']=='mixed' and r['supervision']==sup and r['epochs']==epochs]
        vals=[np.mean([r['ranks'][str(ell)]['relative_coefficient_error'] for r in group]) for ell in range(1,6)]
        axes[1].plot(range(1,6),vals,style,color=col,marker='o',ms=4,label=f'{sup.capitalize()}, {epochs}')
    axes[1].set(yscale='log',xticks=range(1,6),xlabel=r'Multipole rank $\ell$',ylabel='Mean relative coefficient error',title='B  Rank-resolved reconstruction')
    axes[1].legend(fontsize=8);save(fig,'diagnostic')

    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    paired={}
    for ax,key,title,ylabel in zip(axes,('rank45_squared_error','trace_distance'),('A  High-rank accuracy','B  State distinguishability'),
                                 (r'Rank-4 + rank-5 squared error $E_{45}$','Trace distance')):
        differences=[]
        for a,b in zip(test0,testh,strict=True):
            assert a['seed']==b['seed']
            ax.plot([0,1],[a[key],b[key]],'-o',color=TEAL,alpha=.5,lw=1,ms=4)
            differences.append(b[key]-a[key])
        ax.plot([0,1],[mean(test0,key),mean(testh,key)],'-D',color=NAVY,lw=2,ms=6,label='Mean (5 seeds)')
        ax.set(yscale='log',xticks=[0,1],xticklabels=['Fidelity\nλ = 0','Hybrid\nλ = 0.75'],xlim=(-.3,1.3),ylabel=ylabel,title=title)
        ax.legend(fontsize=8)
        d=np.array(differences);half=float(student_t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
        paired[key]={'mean_hybrid_minus_fidelity':float(d.mean()),'descriptive_t95_interval':[float(d.mean()-half),float(d.mean()+half)],
                     'seeds_improved':int(np.sum(d<0)),
                     'relative_reduction_of_means_percent':reductions[key]}
    save(fig,'hybrid')
    (RESULT/'paper_summary.json').write_text(json.dumps({'diagnostic':summary,'hybrid_paired_confirmation':paired},indent=2)+'\n')
    print('Generated 3 figures, 3 tables, numerical macros, and paired summaries.')


if __name__=='__main__': main()
