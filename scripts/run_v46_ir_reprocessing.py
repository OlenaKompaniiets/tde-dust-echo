"""UGC11487 V4.6 outward diffuse-IR reprocessing CONTROL.

First run:
  python scripts/test_v46_ir_reprocessing.py
Then:
  python scripts/run_v46_ir_reprocessing.py --workers 2

This is a small fixed-geometry control, not a geometry refit.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,argparse,traceback,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_v45_direct_absorption as v45
import scripts.run_v4_model_grid as g
from src.v46_ir_reprocessing import solve_direct_plus_ir
from src.v4_wise_emission import grain_size_number_weights

# Start at the V4.5 best direct-absorption point. No geometry freedom here.
BASE=dict(v45.BASE,tau_C=.4,tau_E=2.2,attenuation=True,label='v46_ir_control')


def simulate(c):
    t,L,fullE=v45.driver(c['stars'],c['tau_visc'],c['eta'],c['dt'],c['source_end_days'])
    op,th,a,lookup=g._cached_dust_static(c['composition'],c['amin'],c['amax'])
    wa=grain_size_number_weights(a,q=c['q']);qs=th.source_mean_qabs(a)
    lam=th.wavelength_m;C=(np.pi*(a*1e-6)**2)[:,None]*op.q_abs(th.wavelength_micron[None,:],a[:,None])
    f=float(np.cos(np.deg2rad(c['theta'])))
    radii=[];nums=[];weights=[];totals=[]
    for key,tau in [('compact',c['tau_C']),('extended',c['tau_E'])]:
        r,rw=g.radial_grid_and_weights(*c[key],c['n_radii'])
        pop=g.build_population_normalization(th,a,r,1.,q=c['q'],clump_weight=rw)
        N=pop.total_grain_number*f*tau
        radii.append(r);weights.append(rw);totals.append(N);nums.append(N*rw[:,None]*wa[None,:])
    radii_all=np.concatenate(radii);numbers=np.concatenate(nums)
    sub=float(g.sublimation_temperature_K(c['density'],c['composition']))
    z=solve_direct_plus_ir(t,L,lam,th.source_sed,C,qs,lookup,radii_all,numbers,f,sub,
                           attenuation=True,ir_reprocess_fraction=c['ir_fraction'])
    w1,w2,vw,vf=g._cached_wise_static();dl=g.D_L_MPC*1e6*g.PC_M
    per=[g.absolute_population_band_cube(op,b,a,z['T'],z['alive'],c['q'],g.Z,dl,vw,vf,g.WISE_TGRID) for b in (w1,w2)]
    mu0,muw=np.polynomial.legendre.leggauss(c['n_mu'])
    mu=np.repeat(f*mu0,c['n_phi']);phi=np.tile((np.arange(c['n_phi'])+.5)*2*np.pi/c['n_phi'],c['n_mu'])
    angle_w=np.repeat(muw/(2*c['n_phi']),c['n_phi']);nang=len(mu);inc=np.deg2rad(c['inc'])
    zfactor=mu*np.cos(inc)+np.sqrt(1-mu**2)*np.cos(phi)*np.sin(inc)
    comps=[]
    for iz,r in enumerate(radii):
        sl=slice(iz*c['n_radii'],(iz+1)*c['n_radii']);rc=np.repeat(r,nang);zc=(r[:,None]*zfactor).ravel();ww=(weights[iz][:,None]*angle_w[None,:]).ravel()
        echoes=g.integrate_radial_responses_over_clumps(t,r,totals[iz]*np.stack([x[:,sl] for x in per]),rc,zc,clump_weight=ww,redshift=g.Z,source_time_input=True)
        comps.append(dict(time=echoes[0].time_obs_days,w1=echoes[0].response,w2=echoes[1].response))
    Esource=float(np.trapezoid(L,t)*86400); Edir=np.trapezoid(z['absorbed_direct_erg_s'],t,axis=0)*86400; Eir=np.trapezoid(z['absorbed_ir_erg_s'],t,axis=0)*86400
    en=dict(E_source_window_erg=Esource,E_source_full_driver_erg=fullE,source_energy_window_fraction=Esource/fullE,
            E_direct_absorbed_erg=float(Edir.sum()),E_secondary_IR_absorbed_erg=float(Eir.sum()),
            secondary_to_direct_absorbed_ratio=float(Eir.sum()/max(Edir.sum(),1e-300)),
            energy_budget_relative_max=z['energy_budget_relative_max'],ir_reprocess_fraction=c['ir_fraction'],
            initial_radial_IR_absorption_depth={str(w):float(np.interp(w*1e-6,lam,z['initial_tau_lambda'])) for w in (3.4,4.6,10.)},
            model_scope='radial outward diffuse-IR reprocessing control; isotropic hemisphere split; no scattering/non-radial diffuse paths')
    return comps,en


def evaluate(c):
    comps,en=simulate(c);obs=g.load_observations();tt=obs.mjd.to_numpy();y=np.r_[obs.F_W1_Jy,obs.F_W2_Jy];sig=np.r_[obs.sigma_W1_Jy,obs.sigma_W2_Jy]
    b=np.r_[np.full(len(tt),obs.baseline_W1_mJy.iloc[0]/1000),np.full(len(tt),obs.baseline_W2_mJy.iloc[0]/1000)]
    def model(t0):return np.concatenate([sum(np.interp(tt-t0,x['time'],x[band],left=0,right=0) for x in comps) for band in ('w1','w2')])
    def fun(t0):return float(np.sum(((y-b-model(t0))/sig)**2))
    grid=np.arange(56800,57400.1,10.);vals=np.array([fun(x) for x in grid]);cand=list(zip(vals,grid))
    for i in range(1,len(grid)-1):
        if vals[i]<=vals[i-1] and vals[i]<=vals[i+1]:
            o=minimize_scalar(fun,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':.03})
            if o.success:cand.append((float(o.fun),float(o.x)))
    chi,t0=min(cand);res=(y-b-model(t0))/sig;n=len(tt)
    return dict(config=c,chi2_total=chi,chi2_w1=float(res[:n]@res[:n]),chi2_w2=float(res[n:]@res[n:]),t0_mjd=t0,energy=en)


def tasks():
    # 0 is the mandatory exact V4.5 regression point; 0.5 is fiducial isotropic split.
    return [dict(BASE,ir_fraction=x) for x in (0.,.25,.5,.75,1.)]

def worker(c):return evaluate(c)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args(); ts=tasks()
    out=ROOT/'results'/'v46_ir_reprocessing_control';out.mkdir(parents=True,exist_ok=True)
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,c):c for c in ts}
        for f in as_completed(fs):
            try:
                r=f.result();rows.append(r);print('ir_fraction=',r['config']['ir_fraction'],'chi2=',r['chi2_total'],'W1=',r['chi2_w1'],'W2=',r['chi2_w2'],flush=True)
            except Exception:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    if errors:(out/'errors.json').write_text(json.dumps(errors,indent=2))
    if not rows:raise RuntimeError('No successful V4.6 controls')
    rows=sorted(rows,key=lambda x:x['config']['ir_fraction']);pd.json_normalize(rows).to_csv(out/'blocks.csv',index=False)
    (out/'results.json').write_text(json.dumps(rows,indent=2));best=min(rows,key=lambda x:x['chi2_total']);(out/'best.json').write_text(json.dumps(best,indent=2))
    # Mandatory scientific gate: ir_fraction=0 must recover V4.5 best chi2=174.564 within small numerical tolerance.
    z0=next(x for x in rows if x['config']['ir_fraction']==0.)
    gate=abs(z0['chi2_total']-174.564)<0.25
    (out/'regression_gate.json').write_text(json.dumps(dict(expected_v45_chi2=174.564,actual=z0['chi2_total'],pass_gate=gate),indent=2))
    print('V4.5 regression gate:',gate,'actual=',z0['chi2_total'],flush=True)
    print('BEST',best['config']['ir_fraction'],best['chi2_total'],'saved to',out,flush=True)
    if not gate:raise SystemExit('STOP: V4.6 does not reproduce V4.5 at ir_fraction=0; do not interpret nonzero controls')
if __name__=='__main__':main()
