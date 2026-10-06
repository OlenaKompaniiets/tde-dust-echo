"""UGC11487 V4.5 direct spectral absorption control. Not full diffuse-IR RT.
python scripts/run_v45_direct_absorption.py --mode control
python scripts/run_v45_direct_absorption.py --mode grid --workers 2
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,itertools,argparse,traceback
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_v4_model_grid as g
from src.v45_direct_absorption import solve_direct,quadrature_weights
from src.v4_wise_emission import grain_size_number_weights
from src.v4_thermal import planck_lambda_si

BASE=dict(stars='input/m3.0_t0.0/0.850.dat',tau_visc=350.,eta=.1,
          compact=[.2,.5,0.],extended=[1.,1.5,2.],theta=40.,inc=40.,
          composition='graphite',amin=.1,amax=1.,q=3.5,density=1e9,
          n_radii=32,n_mu=24,n_phi=64,dt=10.,source_end_days=6500.,seed=12345)

@lru_cache(maxsize=8)
def driver(stars,tau,eta,dt,end):
    d,diag=g.build_physical_driver(stars_file=g.STARS_ROOT/stars,M_BH_Msun=g.M_BH_MSUN,
        eta_rad=eta,tau_visc_days=tau,t_circ_days=g.T_CIRC_DAYS,dt_days=dt,
        pre_peak_days=g.PRE_PEAK_DAYS,post_fallback_days=g.POST_FALLBACK_DAYS)
    fullE=float(np.trapezoid(d.L_bol_erg_s,d.t_days)*86400)
    d=d[d.t_days<=end];return d.t_days.to_numpy(),d.L_bol_erg_s.to_numpy(),fullE


def simulate(c):
    t,L,fullE=driver(c['stars'],c['tau_visc'],c['eta'],c['dt'],c['source_end_days'])
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
    if radii[0][-1]>=radii[1][0]:raise ValueError('This control requires disjoint radially ordered populations')
    radii_all=np.concatenate(radii);numbers=np.concatenate(nums)
    sub=float(g.sublimation_temperature_K(c['density'],c['composition']))
    result=solve_direct(t,L,lam,th.source_sed,C,qs,lookup,radii_all,numbers,f,sub,
                        attenuation=c.get('attenuation',True))
    w1,w2,vw,vf=g._cached_wise_static();dl=g.D_L_MPC*1e6*g.PC_M
    per=[g.absolute_population_band_cube(op,b,a,result['T'],result['alive'],c['q'],g.Z,dl,vw,vf,g.WISE_TGRID) for b in (w1,w2)]
    # Same angular rays at every radial cell: uniform solid angle within wedge.
    # Grain numbers are radial quadrature weights, not random clump counts.
    # Deterministic quadrature prevents small grids from being driven by clump noise.
    mu0,muw=np.polynomial.legendre.leggauss(c['n_mu'])
    mu=np.repeat(f*mu0,c['n_phi']);phi=np.tile((np.arange(c['n_phi'])+.5)*2*np.pi/c['n_phi'],c['n_mu'])
    angle_w=np.repeat(muw/(2*c['n_phi']),c['n_phi']);nang=len(mu)
    inc=np.deg2rad(c['inc']);zfactor=mu*np.cos(inc)+np.sqrt(1-mu**2)*np.cos(phi)*np.sin(inc)
    components=[]
    for iz,r in enumerate(radii):
        sl=slice(iz*c['n_radii'],(iz+1)*c['n_radii'])
        rc=np.repeat(r,nang);zc=(r[:,None]*zfactor).ravel()
        ww=(weights[iz][:,None]*angle_w[None,:]).ravel()
        echoes=g.integrate_radial_responses_over_clumps(t,r,
            totals[iz]*np.stack([x[:,sl] for x in per]),rc,zc,clump_weight=ww,redshift=g.Z,source_time_input=True)
        components.append(dict(time=echoes[0].time_obs_days,w1=echoes[0].response,w2=echoes[1].response))
    Esource=float(np.trapezoid(L,t)*86400)
    Eabs=np.trapezoid(result['absorbed_erg_s'],t,axis=0)*86400
    Eescape=float(np.trapezoid(result['escaped_direct_erg_s'],t)*86400)
    # Independently evaluate Planck/Q emission at the source peak, all radial cells and sizes.
    it=int(np.argmax(L));emit=0.;abs_peak=float(result['absorbed_erg_s'][it].sum())
    for ir in range(len(radii_all)):
        for ia,aa in enumerate(a):
            temp=result['T'][it,ir,ia]
            if result['alive'][it,ir,ia] and temp>0:
                B=planck_lambda_si(lam,temp)
                emit+=numbers[ir,ia]*4*np.pi*np.trapezoid(C[ia]*B,lam)*1e7
    ir_tau={str(w):float(np.interp(w*1e-6,lam,result['initial_tau_lambda'])) for w in (3.4,4.6,10.)}
    energy=dict(E_source_window_erg=Esource,E_source_full_driver_erg=fullE,
        source_energy_window_fraction=Esource/fullE,E_abs_compact_erg=float(sum(Eabs[:c['n_radii']])),
        E_abs_extended_erg=float(sum(Eabs[c['n_radii']:])),E_abs_total_erg=float(sum(Eabs)),
        E_direct_escape_erg=Eescape,absorbed_fraction_window=float(sum(Eabs))/Esource,
        geometric_absorption_ceiling_erg=f*Esource,
        direct_budget_relative_max=result['energy_budget_relative_max'],
        grain_heating_partition_relative_max=result['heating_partition_relative_max'],
        independent_Planck_closure_at_source_peak=emit/max(abs_peak,1e-300)-1,
        initial_radial_IR_absorption_depth=ir_tau,
        IR_transfer_review_required=bool(max(ir_tau.values())>.1),
        max_cell_tau_lambda=result['max_cell_tau_lambda'],
        max_sublimation_iterations=result['max_sublimation_iterations'],
        final_alive_number_fraction=[float(np.sum(nums[j]*result['alive'][-1,j*c['n_radii']:(j+1)*c['n_radii']])/max(totals[j],1e-300)) for j in range(2)],
        model_scope='direct absorption only; isotropic dust emission escapes freely; no diffuse IR/scattering')
    return components,energy


def evaluate(c):
    comps,en=simulate(c);obs=g.load_observations();t=obs.mjd.to_numpy()
    y=np.r_[obs.F_W1_Jy,obs.F_W2_Jy];sig=np.r_[obs.sigma_W1_Jy,obs.sigma_W2_Jy]
    b=np.r_[np.full(len(t),obs.baseline_W1_mJy.iloc[0]/1000),np.full(len(t),obs.baseline_W2_mJy.iloc[0]/1000)]
    def model(tt,t0):return np.concatenate([sum(np.interp(tt-t0,x['time'],x[band],left=0,right=0) for x in comps) for band in ('w1','w2')])
    def fun(t0):return float(np.sum(((y-b-model(t,t0))/sig)**2))
    grid=np.arange(56800,57400.1,10.);vals=np.array([fun(x) for x in grid]);cand=list(zip(vals,grid))
    for i in range(len(grid)):
        if vals[i]<=vals[max(0,i-1)] and vals[i]<=vals[min(i+1,len(grid)-1)]:
            lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
            if hi>lo:
                opt=minimize_scalar(fun,bounds=(lo,hi),method='bounded',options={'xatol':.03})
                if opt.success:cand.append((float(opt.fun),float(opt.x)))
    chi,t0=min(cand);m=b+model(t,t0);res=(y-m)/sig;n=len(t)
    row=dict(config=c,chi2_total=chi,chi2_w1=float(res[:n]@res[:n]),chi2_w2=float(res[n:]@res[n:]),
             t0_mjd=float(t0),t0_boundary=bool(t0<56800.1 or t0>57399.9),energy=en)
    return row,comps


def task_grid(mode):
    f=np.cos(np.deg2rad(BASE['theta']));tc=.29058526074811747/f;te=.330863050715465/f
    if mode=='control':return [dict(BASE,tau_C=tc,tau_E=te,attenuation=b,label=label) for b,label in [(False,'unattenuated_reference'),(True,'same_dust_with_absorption')]]
    return [dict(BASE,tau_C=tc,tau_E=te,attenuation=True,label='direct_absorption_grid') for tc,te in itertools.product((.2,.4,.7,1.),(.4,.8,1.4,2.2))]


def identity(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:20]

def fingerprint():
    files=[Path(__file__),ROOT/'scripts/run_v4_model_grid.py']+list((ROOT/'src').glob('*.py'))+[p for p in (ROOT/'data').rglob('*') if p.is_file()]+[g.STARS_ROOT/BASE['stars']]
    h={p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    return hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest(),h


def worker(c):return evaluate(c)[0]


def products(row,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check,comps=evaluate(row['config'])
    if not np.isclose(check['chi2_total'],row['chi2_total'],atol=1e-6,rtol=1e-8):raise RuntimeError('Reconstruction mismatch')
    obs=g.load_observations();tt=np.linspace(obs.mjd.min()-50,obs.mjd.max()+50,1600)
    full={'mjd':tt};tab={'mjd':obs.mjd.to_numpy()}
    fig,ax=plt.subplots(2,2,figsize=(12,7),sharex='col',gridspec_kw={'height_ratios':[3,1]})
    for j,b in enumerate(('W1','W2')):
        base=obs['baseline_'+b+'_mJy'].iloc[0];parts=[];model=np.full(len(obs),base)
        for k,x in enumerate(comps):
            yy=1000*np.interp(tt-row['t0_mjd'],x['time'],x[b.lower()],left=0,right=0);parts.append(yy)
            full[b+('_compact_mJy' if k==0 else '_extended_mJy')]=yy
            model+=1000*np.interp(obs.mjd-row['t0_mjd'],x['time'],x[b.lower()],left=0,right=0)
            ax[0,j].plot(tt,base+yy,'--' if k==0 else ':',label='Host + '+('compact' if k==0 else 'extended'))
        total=base+sum(parts);full[b+'_total_mJy']=total
        y=obs['F_'+b+'_Jy'].to_numpy()*1000;sig=obs['sigma_'+b+'_Jy'].to_numpy()*1000;res=(y-model)/sig
        ax[0,j].plot(tt,total,label='Total');ax[0,j].errorbar(obs.mjd,y,yerr=sig,fmt='o',color='black',ms=4,label=b)
        ax[0,j].legend(fontsize=8);ax[0,j].set_ylabel('Flux density (mJy)')
        ax[1,j].plot(obs.mjd,res,'o');ax[1,j].axhline(0,color='grey');ax[1,j].set(xlabel='MJD',ylabel='Residual / sigma')
        for name,value in [('data_mJy',y),('model_mJy',model),('sigma_mJy',sig),('residual_sigma',res)]:tab[b+'_'+name]=value
    fig.suptitle('Direct absorption control; chi2=%.2f; diffuse IR omitted'%row['chi2_total']);fig.tight_layout()
    fig.savefig(out/'best_components.png',dpi=180);plt.close(fig)
    pd.DataFrame(full).to_csv(out/'best_dense_curves.csv',index=False);pd.DataFrame(tab).to_csv(out/'best_epochs.csv',index=False)
    (out/'best_energy.json').write_text(json.dumps(row['energy'],indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['control','grid'],default='control');ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--max-blocks',type=int);args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    tasks=task_grid(args.mode);digest,hashes=fingerprint();out=ROOT/'results'/('v45_direct_'+args.mode+'_'+digest[:12]);out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(dict(fingerprint=digest,files=hashes,tasks=tasks,scope='direct-only finite-volume effective-medium control'),indent=2))
    pending=[c for c in tasks if not (out/(identity(c)+'.json')).exists()]
    if args.max_blocks is not None:pending=pending[:max(0,args.max_blocks)]
    print('Completed',len(tasks)-sum(not (out/(identity(c)+'.json')).exists() for c in tasks),'/',len(tasks),'new blocks',len(pending),flush=True)
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(worker,c):c for c in pending}
        for future in as_completed(fs):
            c=fs[future]
            try:
                r=future.result();p=out/(identity(c)+'.tmp');p.write_text(json.dumps(r,indent=2));p.replace(out/(identity(c)+'.json'))
                print(c['label'],'tau_C=',c['tau_C'],'tau_E=',c['tau_E'],'chi2=',r['chi2_total'],flush=True)
            except Exception:errors.append(dict(config=c,traceback=traceback.format_exc()));print(errors[-1]['traceback'],flush=True)
    rows=[json.loads((out/(identity(c)+'.json')).read_text()) for c in tasks if (out/(identity(c)+'.json')).exists()]
    if errors:(out/'errors.json').write_text(json.dumps(errors,indent=2))
    if not rows:raise RuntimeError('No successful blocks')
    pd.json_normalize(rows).to_csv(out/'blocks.csv',index=False)
    # Select among attenuated models; reference is never scientific best.
    eligible=[r for r in rows if r['config']['attenuation']]
    if eligible:
        best=min(eligible,key=lambda r:r['chi2_total']);(out/'best.json').write_text(json.dumps(best,indent=2));products(best,out)
    print('Saved',len(rows),'/',len(tasks),'to',out,flush=True)
    if errors:raise SystemExit('Some blocks failed; successful checkpoints retained')
if __name__=='__main__':main()
