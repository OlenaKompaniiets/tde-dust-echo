"""Geometry/normalization CONTROL, not a self-shielding radiative-transfer model.
Run: python scripts/run_v43c_geometry_audit.py --mode control
Then: python scripts/run_v43c_geometry_audit.py --mode geometry --workers 4
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, hashlib, itertools, json, sys, traceback
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_v43b_extended_geometry_test as v
from src.v4_dust_population import surviving_absorbed_fraction

def fingerprint():
    files=sorted(list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+
                 [p for p in (ROOT/'data').rglob('*') if p.is_file()]+
                 [v.g.STARS_ROOT/s for s in v.DRIVERS])
    hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

@lru_cache(maxsize=48)
def echo(s,tau,zone,theta,inc):
    v.THETA=theta; v.INC=inc
    return v.build_echo(s,tau,zone)

def fit(cp,ep,obs,cap):
    v.FMAX=v.FTOT=float(cap)
    t=obs.mjd.to_numpy(); y1=obs.F_W1_Jy.to_numpy();y2=obs.F_W2_Jy.to_numpy()
    s1=obs.sigma_W1_Jy.to_numpy();s2=obs.sigma_W2_Jy.to_numpy()
    b1=obs.baseline_W1_mJy.iloc[0]/1000;b2=obs.baseline_W2_mJy.iloc[0]/1000
    def obj(t0,payload=False):
        c1,c2=v.ip(cp,t-t0);e1,e2=v.ip(ep,t-t0)
        z=v.solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2)
        return z if payload else z[3]+z[4]
    grid=np.arange(56800.,57400.1,10.); values=np.array([obj(t0) for t0 in grid])
    candidates=[(float(values[i]),float(grid[i])) for i in range(len(grid))]
    # Refine EVERY local minimum and retain all coarse points and endpoints.
    for i in range(len(grid)):
        if values[i]<=values[max(0,i-1)] and values[i]<=values[min(len(grid)-1,i+1)]:
            lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
            if hi>lo:
                z=minimize_scalar(obj,bounds=(lo,hi),method='bounded',options={'xatol':.02})
                if z.success and np.isfinite(z.fun): candidates.append((float(z.fun),float(z.x)))
    chi,t0=min(candidates); z=obj(t0,True)
    return dict(best_t0_mjd=t0,f_abs_compact=float(z[0][0]),f_abs_extended=float(z[0][1]),
                f_abs_total=float(sum(z[0])),chi2_w1=z[3],chi2_w2=z[4],chi2_total=chi,
                t0_boundary=bool(t0<=grid[0]+.1 or t0>=grid[-1]-.1)),z

def tasks(mode):
    s=v.DRIVERS[0]; c=(.15,.5,0.);e=(.75,3.,1.)
    if mode=='control':
        return [dict(stars=s,tau=350.,compact=c,extended=e,theta=60.,inc=55.,cap=f,
                     label=label) for f,label in [(.8,'legacy_reference'),(.5,'geometric_cap_control')]]
    out=[]
    # A targeted 720-block search, not a full combinatorial science grid.
    for s,tau,c,e,theta,inc in itertools.product(v.DRIVERS,(300.,350.,400.),
        ((.10,.40,0.),(.15,.50,0.),(.20,.50,0.),(.15,.40,1.),(.15,.60,2.)),
        ((.75,3.,1.),(1.,1.5,2.)),(30.,40.,50.,60.),(40.,55.)):
        out.append(dict(stars=s,tau=tau,compact=c,extended=e,theta=theta,inc=inc,
                        cap=min(.8,float(np.cos(np.deg2rad(theta)))),label='geometry_control'))
    return out

def taskid(task):
    return hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()[:20]

def worker(task):
    obs=v.g.load_observations()
    cp=echo(task['stars'],task['tau'],tuple(task['compact']),task['theta'],task['inc'])
    ep=echo(task['stars'],task['tau'],tuple(task['extended']),task['theta'],task['inc'])
    r,_=fit(cp,ep,obs,task['cap']); cover=float(np.cos(np.deg2rad(task['theta'])))
    r.update(task);r.update(block_id=taskid(task),solid_angle_fraction=cover,
        mean_linear_absorption_column=r['f_abs_total']/cover,
        exceeds_solid_angle=bool(r['f_abs_total']>cover+1e-8),
        attenuation_review_required=bool(r['f_abs_total']/cover>.1),
        at_amplitude_cap=bool(r['f_abs_total']>=task['cap']-1e-6),
        compact_alive_final=cp['alive'],extended_alive_final=ep['alive'])
    return r

def energy_check(row):
    """Source-clock energy integral and equilibrium-lookup closure; NOT full RT."""
    g=v.g
    driver,_=g.build_physical_driver(stars_file=g.STARS_ROOT/row['stars'],M_BH_Msun=g.M_BH_MSUN,
        eta_rad=v.ETA,tau_visc_days=row['tau'],t_circ_days=g.T_CIRC_DAYS,
        dt_days=g.DRIVER_DT_DAYS,pre_peak_days=g.PRE_PEAK_DAYS,post_fallback_days=g.POST_FALLBACK_DAYS)
    td=driver.t_days.to_numpy();L=driver.L_bol_erg_s.to_numpy()
    result={'E_source_erg':float(np.trapezoid(L,td)*86400),
            'interpretation':'independent-absorber energy bookkeeping; no shielding or IR self-absorption'}
    _,th,a,lookup=g._cached_dust_static('graphite',v.GRAIN[0],v.GRAIN[1])
    for key,amp in [('compact',row['f_abs_compact']),('extended',row['f_abs_extended'])]:
        r,rw=g.radial_grid_and_weights(*row[key],g.N_RADII)
        d=g._build_dust_response_cached(td,L,r,a,lookup,'graphite',v.DENS)
        pop=g.build_population_normalization(th,a,r,1.,q=v.GRAIN[2],clump_weight=rw)
        frac=surviving_absorbed_fraction(pop,r,d.alive,clump_weight=rw)
        result['E_abs_'+key+'_erg']=float(amp*np.trapezoid(L*frac,td)*86400)
        # Reconstruct emitted power using the forward equilibrium table F(T).
        emitted=np.zeros(len(td))
        for ia,aa in enumerate(a):
            T=d.temperature_K[:,:,ia];valid=d.alive[:,:,ia]&np.isfinite(T)&(T>0)
            flux=np.zeros_like(T)
            flux[valid]=10**np.interp(np.log10(T[valid]),np.log10(lookup.temperature_grid_K),
                                     lookup.log10_flux_grid_W_m2[ia,:])
            cross=np.pi*(aa*1e-6)**2*pop.source_mean_qabs[ia]
            emitted+=pop.total_grain_number*pop.grain_number_weights[ia]*cross*(flux@rw)*1e7
        result['E_emit_lookup_'+key+'_erg']=float(amp*np.trapezoid(emitted,td)*86400)
    result['E_abs_total_erg']=sum(result['E_abs_'+k+'_erg'] for k in ('compact','extended'))
    result['E_emit_lookup_total_erg']=sum(result['E_emit_lookup_'+k+'_erg'] for k in ('compact','extended'))
    result['geometric_source_energy_ceiling_erg']=row['solid_angle_fraction']*result['E_source_erg']
    result['passes_geometric_energy_ceiling']=bool(result['E_abs_total_erg']<=result['geometric_source_energy_ceiling_erg']*(1+1e-6))
    result['closure_relative']=result['E_emit_lookup_total_erg']/max(result['E_abs_total_erg'],1e-300)-1
    return result

def products(row,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    obs=v.g.load_observations();cp=echo(row['stars'],row['tau'],tuple(row['compact']),row['theta'],row['inc'])
    ep=echo(row['stars'],row['tau'],tuple(row['extended']),row['theta'],row['inc'])
    fitted,z=fit(cp,ep,obs,row['cap'])
    if not np.isclose(fitted['chi2_total'],row['chi2_total'],atol=1e-5,rtol=1e-7):
        raise RuntimeError('Best-model reconstruction differs from checkpoint')
    tt=np.linspace(obs.mjd.min()-50,obs.mjd.max()+50,1800)
    c=v.ip(cp,tt-row['best_t0_mjd']);e=v.ip(ep,tt-row['best_t0_mjd'])
    fig,ax=plt.subplots(2,2,figsize=(12,7),sharex='col',gridspec_kw={'height_ratios':[3,1]})
    tab={'mjd':obs.mjd.to_numpy()};full={'mjd':tt}
    for j,band in enumerate(('W1','W2')):
        base=obs['baseline_'+band+'_mJy'].iloc[0];y=1000*obs['F_'+band+'_Jy'].to_numpy();sig=1000*obs['sigma_'+band+'_Jy'].to_numpy()
        cc=1000*row['f_abs_compact']*c[j];ee=1000*row['f_abs_extended']*e[j]
        pred=1000*z[j+1];res=(y-pred)/sig
        ax[0,j].errorbar(obs.mjd,y,yerr=sig,fmt='o',ms=4,color='black',label='WISE '+band)
        ax[0,j].plot(tt,base+cc+ee,label='Total');ax[0,j].plot(tt,base+cc,'--',label='Host + compact')
        ax[0,j].plot(tt,base+ee,':',label='Host + extended');ax[0,j].axhline(base,color='grey',lw=.7)
        ax[0,j].set_ylabel('Flux density (mJy)');ax[0,j].legend(fontsize=8)
        ax[1,j].axhline(0,color='grey');ax[1,j].plot(obs.mjd,res,'o',ms=4);ax[1,j].set(xlabel='MJD',ylabel='Residual / sigma')
        for name,value in [('data',y),('model',pred),('sigma',sig),('residual_sigma',res)]:tab[band+'_'+name]=value
        full[band+'_compact_mJy']=cc;full[band+'_extended_mJy']=ee;full[band+'_total_mJy']=base+cc+ee
    fig.suptitle('Geometry control: chi2=%.2f; shielding not included'%row['chi2_total'])
    fig.tight_layout();fig.savefig(out/'best_components.png',dpi=200);plt.close(fig)
    pd.DataFrame(tab).to_csv(out/'best_epochs.csv',index=False);pd.DataFrame(full).to_csv(out/'best_dense_curves.csv',index=False)
    (out/'best_energy.json').write_text(json.dumps(energy_check(row),indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['control','geometry'],default='control')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--max-blocks',type=int)
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    digest,hashes=fingerprint();out=ROOT/'results'/('v43c_'+args.mode+'_'+digest[:12]);out.mkdir(parents=True,exist_ok=True)
    todo=tasks(args.mode); manifest={'code_and_input_sha256':digest,'files':hashes,'tasks':todo,
        'model':'independent absorbers; geometric cap is necessary, not sufficient for RT consistency',
        'thin_column_flag_threshold':.1,'threshold_is_heuristic':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pending=[t for t in todo if not (out/(taskid(t)+'.json')).exists()]
    if args.max_blocks is not None:pending=pending[:max(0,args.max_blocks)]
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(worker,t):t for t in pending}
        for f in as_completed(fs):
            task=fs[f]
            try:
                r=f.result();tmp=out/(taskid(task)+'.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(out/(taskid(task)+'.json'))
                print(r['block_id'],'chi2=%.3f'%r['chi2_total'],'f=%.3f'%r['f_abs_total'],flush=True)
            except Exception:
                failures.append({'task':task,'traceback':traceback.format_exc()});print(failures[-1]['traceback'],flush=True)
    if failures:(out/'errors.json').write_text(json.dumps(failures,indent=2))
    rows=[json.loads((out/(taskid(t)+'.json')).read_text()) for t in todo if (out/(taskid(t)+'.json')).exists()]
    if not rows:raise RuntimeError('No successful blocks')
    rows.sort(key=lambda r:r['chi2_total']);pd.DataFrame(rows).to_csv(out/'blocks.csv',index=False)
    (out/'best.json').write_text(json.dumps(rows[0],indent=2));products(rows[0],out)
    # Always export the physically capped control separately from legacy reference.
    if args.mode=='control':
        for r in rows:
            sub=out/r['label'];sub.mkdir(exist_ok=True);products(r,sub)
    print('Completed',len(rows),'/',len(todo),'RESULTS',out,flush=True)
    if failures:raise SystemExit('Some blocks failed; see errors.json. Successful checkpoints retained.')
if __name__=='__main__':main()
