"""One configured V4.8 grid model, with atomic checkpoints and shared RT cache.
Effective axisymmetric medium; static opacity; no scattering.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib,time,argparse,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
# NumPy 1.26 compatibility for the existing physical modules.
if not hasattr(np, "trapezoid"): np.trapezoid=np.trapz
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import make_interp_spline
import scripts.run_v45_direct_absorption as v45
from src import v48_angular_rt as rt
from src.v4_wise_emission import grain_size_number_weights,wise_vega_scale_jy
from src.v4_thermal import planck_lambda_si
g=v45.g


def write_json(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def write_npz(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def profile(time_obs,flux,obs):
    tt=obs.mjd.to_numpy()
    y=np.stack([obs['F_'+b+'_Jy'] for b in ('W1','W2')])
    sig=np.stack([obs['sigma_'+b+'_Jy'] for b in ('W1','W2')])
    host=np.array([obs['baseline_'+b+'_mJy'].iloc[0]/1000 for b in ('W1','W2')])[:,None]
    def model(t,t0):return np.stack([np.interp(np.asarray(t)-t0,time_obs,x,left=0,right=0) for x in flux])
    def loss(t0):return float(np.sum(((y-host-model(tt,t0))/sig)**2))
    grid=np.arange(56500,57500.1,10.);vals=np.array([loss(t0) for t0 in grid]);candidates=list(zip(vals,grid))
    for i in range(len(grid)):
        if vals[i]<=vals[max(0,i-1)] and vals[i]<=vals[min(len(grid)-1,i+1)]:
            opt=minimize_scalar(loss,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded',options={'xatol':.02})
            if opt.success:candidates.append((float(opt.fun),float(opt.x)))
    chi,t0=min(candidates);res=(y-host-model(tt,t0))/sig
    result=dict(chi2_total=chi,chi2_w1=float(res[0]@res[0]),chi2_w2=float(res[1]@res[1]),t0_mjd=t0,
                t0_boundary=bool(t0<56500.1 or t0>57499.9))
    return result,model,res


def products(out,tobs,flux,result,state,mesh,obs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,7),sharex='col',gridspec_kw={'height_ratios':[3,1]})
    best,model,res=profile(tobs,flux,obs)
    rows={'mjd':obs.mjd.to_numpy()};tt=np.linspace(obs.mjd.min()-70,obs.mjd.max()+70,1800)
    dense={'mjd':tt}
    for i,b in enumerate(('W1','W2')):
        host=obs['baseline_'+b+'_mJy'].iloc[0]
        y=obs['F_'+b+'_Jy'].to_numpy()*1000;sig=obs['sigma_'+b+'_Jy'].to_numpy()*1000
        m=host+1000*model(obs.mjd,best['t0_mjd'])[i]
        curve=host+1000*model(tt,best['t0_mjd'])[i]
        ax[0,i].errorbar(obs.mjd,y,yerr=sig,fmt='o',color='black',ms=4,label='WISE data')
        ax[0,i].plot(tt,curve,color='#0072b2',label='Angular RT, fixed physical parameters')
        ax[0,i].set(title=b,ylabel='Flux density (mJy)');ax[0,i].legend(fontsize=8)
        ax[1,i].plot(obs.mjd,res[i],'o-',ms=3,color='#0072b2');ax[1,i].axhline(0,color='0.5',lw=.8)
        ax[1,i].set(xlabel='MJD',ylabel='Residual / sigma')
        rows.update({b+'_data_mJy':y,b+'_model_mJy':m,b+'_sigma_mJy':sig,b+'_residual_sigma':res[i]})
        dense[b+'_model_mJy']=curve
    fig.suptitle('UGC 11487: absorption + re-emission along 3-D rays\nAxisymmetric effective medium; static-opacity fixed-geometry calculation',fontsize=12)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'angular_rt_lightcurves.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'epochs.csv',index=False);pd.DataFrame(dense).to_csv(out/'dense.csv',index=False)
    # r-mu temperature map: use the time of largest intrinsic MIR luminosity.
    phase=int(np.argmax(state['emitted'].sum(axis=(1,2))))
    mapping=np.full(mesh.lookup.shape,np.nan)
    for ic,(ir,im) in enumerate(zip(mesh.cell_radial,mesh.cell_mu)):
        mapping[ir,im]=np.average(state['T'][phase,ic],weights=mesh.numbers[ic])
    fig,ax=plt.subplots(figsize=(8,4.7))
    artist=ax.pcolormesh(mesh.radial_edges,mesh.mu_edges,mapping.T,shading='flat',cmap='inferno')
    ax.set(xlabel='Radius (pc)',ylabel='cos(polar angle)',title='Number-weighted grain temperature at peak source-retarded phase')
    fig.colorbar(artist,ax=ax,label='Temperature (K)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'temperature_map.{ext}',dpi=180)
    plt.close(fig)

    # Actual simultaneous rest-frame snapshots: T(r,t)=T_cell(u=t-r/c).
    # These are sampling points of an effective medium, not resolved clouds.
    from scipy.stats import qmc
    xyz=[];cell=[]
    for ic in range(mesh.nc):
        u=qmc.Sobol(3,scramble=True,seed=3100+ic).random_base2(5)
        points=rt.sample_positions(mesh,ic,u);xyz.extend(points);cell.extend([ic]*len(points))
    xyz=np.asarray(xyz);cell=np.asarray(cell);r=np.linalg.norm(xyz,axis=1)
    meanT=np.sum(state['T']*mesh.numbers[None,:,:],axis=2)/mesh.numbers.sum(axis=1)[None,:]
    phases=[250.,750.,2000.];snap={}
    fig=plt.figure(figsize=(14,4.7));norm=plt.Normalize(0,max(1100.,float(meanT.max())))
    for j,phase in enumerate(phases):
        temp=np.array([np.interp(phase-rr*rt.PC_DAYS,state['time'],meanT[:,ic],left=0,right=0) for rr,ic in zip(r,cell)])
        snap[f'T_number_mean_{int(phase)}d_K']=temp
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        p=ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=temp,cmap='inferno',norm=norm,s=3,alpha=.65,rasterized=True)
        ax.scatter([0],[0],[0],marker='*',s=45,color='cyan')
        ax.set(xlim=(-mesh.radius,mesh.radius),ylim=(-mesh.radius,mesh.radius),zlim=(-mesh.radius,mesh.radius),
               xlabel='x (pc)',ylabel='y (pc)',zlabel='z (pc)',title=f't_local = {int(phase)} rest-frame days')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=50,azim=0)
    fig.subplots_adjust(left=.02,right=.90,bottom=.03,top=.83,wspace=.05)
    cax=fig.add_axes([.93,.22,.013,.5]);fig.colorbar(p,cax=cax,label='Number-weighted grain temperature (K)')
    fig.suptitle('Simultaneous thermal evolution of the effective torus\nSampling points show the temperature field; they are not individual physical clumps',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(out/f'thermal_evolution_3d.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(dict(x_pc=xyz[:,0],y_pc=xyz[:,1],z_pc=xyz[:,2],cell=cell,**snap)).to_csv(out/'thermal_snapshots_3d.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resolution',choices=['coarse','fine'],default='coarse')
    ap.add_argument('--seed',type=int,default=481)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plots',action='store_true')
    ap.add_argument('--angular-check',action='store_true')
    args=ap.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    numerics=dict(nr=4,nmu=4,nwave=96,dt=40.,nrays=128,nobserver=256,stop_days=15000.,seed=args.seed)
    if args.resolution=='fine':numerics.update(nr=6,nmu=6,nwave=144,dt=30.,nrays=256,nobserver=512)
    if args.angular_check: numerics.update(nmu=12,nrays=512,nobserver=1024)
    # Keep the established 10-day physical driver identical across RT grids.
    # Integrate it over RT time bins; never renormalize to a target energy.
    config.update(dt=10.)
    paths=[Path(__file__),ROOT/'src/v48_angular_rt.py',ROOT/'scripts/run_v45_direct_absorption.py',ROOT/'scripts/run_v4_model_grid.py']
    paths+=list((ROOT/'src').glob('*.py'))+[p for p in (ROOT/'data').rglob('*') if p.is_file()]+[g.STARS_ROOT/config['stars']]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    prov=dict(config=config,numerics=numerics,hashes=hashes,python=platform.python_version(),numpy=np.__version__,
              scope='axisymmetric effective medium; 3-D angular ray transfer; no scattering; sublimation threshold gate')
    digest=hashlib.sha256(json.dumps(prov,sort_keys=True).encode()).hexdigest()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() and (out/'response.csv').exists():
        old=json.loads((out/'summary.json').read_text())
        if old.get('provenance_sha256')==digest and (not args.plots or (out/'thermal_solution.npz').exists()):
            print('Already complete with identical fingerprint',flush=True);return
        if old.get('provenance_sha256')!=digest:
            raise RuntimeError('Existing output fingerprint differs; use a new output folder')
    write_json(out/'provenance.json',dict(sha256=digest,**prov));print('OUTPUT',out,flush=True)
    started=time.monotonic()
    op,th,grains,lookup=g._cached_dust_static(config['composition'],config['amin'],config['amax'])
    wa=grain_size_number_weights(grains,q=config['q'])
    lam=np.geomspace(.01,1000,numerics['nwave'])*1e-6
    cross=np.pi*(grains[:,None]*1e-6)**2*op.q_abs(lam[None,:]*1e6,grains[:,None])
    source=planck_lambda_si(lam,g.SOURCE_T_K)*rt.quadrature_weights(lam);source/=source.sum()
    sigma=wa@cross;sigma_source=float(sigma@source)
    mesh=rt.build_mesh(config['compact'],config['extended'],config['tau_C'],config['tau_E'],config['theta'],numerics['nr'],numerics['nmu'],wa,sigma_source)
    D,direct_escape=rt.direct_kernels(mesh,sigma,source,numerics['dt'])
    # Reuse transfer kernels across drivers; opacity, geometry and numerics stay identical.
    kernel_signature=dict(mesh=[config[k] for k in ('compact','extended','tau_C','tau_E','theta','composition','amin','amax','q')],numerics=numerics,hashes=hashes)
    kernel_id=hashlib.sha256(json.dumps(kernel_signature,sort_keys=True).encode()).hexdigest()
    cache=out.parent.parent/'kernel_cache';cache.mkdir(parents=True,exist_ok=True)
    kernel_path=cache/(kernel_id+'.npz')
    if kernel_path.exists():
        with np.load(kernel_path,allow_pickle=False) as f:K=f['K'];E=f['E'];prob_error=float(f['prob_error'])
        print('Restored kernels with identical fingerprint',flush=True)
    else:
        K,E,prob_error=rt.build_transport(mesh,sigma,numerics['dt'],numerics['nrays'],args.seed)
        write_npz(kernel_path,K=K,E=E,prob_error=np.array(prob_error))
    ts,Ls,fullE=v45.driver(config['stars'],config['tau_visc'],config['eta'],config['dt'],config['source_end_days'])
    t=ts[0]+np.arange(int(np.ceil((numerics['stop_days']-ts[0])/numerics['dt']))+1)*numerics['dt']
    primitive=make_interp_spline(ts,Ls,k=1).antiderivative()
    left=np.clip(t-numerics['dt']/2,ts[0],ts[-1])
    right=np.clip(t+numerics['dt']/2,ts[0],ts[-1])
    L=(primitive(right)-primitive(left))/numerics['dt']*1e-7
    L=np.maximum(L,0.)
    expected_window=float(np.trapezoid(Ls,ts)*86400)
    if abs(float(L.sum()*numerics['dt']*86400*1e7)/expected_window-1)>1e-10:
        raise RuntimeError('Conservative driver time-bin integration failed')
    sub=float(g.sublimation_temperature_K(config['density'],config['composition']))
    thermal=rt.Thermal(lam,cross,wa,mesh.numbers,sub)
    solved=rt.solve_time_rt(L,D,K,E,thermal,numerics['dt'])
    solved['time']=t
    solved['energy'].update(E_source_window_erg=float(L.sum()*numerics['dt']*86400*1e7),
        E_source_full_driver_erg=fullE,E_direct_eventual_escape_erg=float(L.sum()*direct_escape*numerics['dt']*86400*1e7))
    del K
    if args.plots: write_npz(out/'thermal_solution.npz',time=t,temperature=solved['T'],emitted_bin_power_W=solved['emitted'],
        wavelength_m=lam,numbers=mesh.numbers,radial_edges=mesh.radial_edges,mu_edges=mesh.mu_edges,
        cell_radial=mesh.cell_radial,cell_mu=mesh.cell_mu)
    write_json(out/'iterations.json',solved['iterations'])
    columns,delays=rt.observer_columns(mesh,config['inc'],numerics['nobserver'],seed=args.seed+9000)
    w1,w2,vw,vf=g._cached_wise_static();flux=[];tobs=None;source_MIR_peak={}
    nobs=np.array([np.sin(np.deg2rad(config['inc'])),0.,np.cos(np.deg2rad(config['inc']))])
    source_paths,_=rt.path_segments(mesh,np.zeros(3),nobs)
    source_column=sum(mesh.radial_density[mesh.cell_radial[d]]*(b-a)*rt.PC for d,a,b in source_paths)
    source_sed_norm=float(np.sum(planck_lambda_si(lam,g.SOURCE_T_K)*rt.quadrature_weights(lam)))
    for bp in (w1,w2):
        print('Observer transfer:',bp.name,flush=True)
        spectral,offset=rt.observer_spectrum(solved['T'],thermal,op,grains,bp.wavelength_micron*1e-6,g.Z,columns,delays,numerics['dt'],mesh.radius)
        projection=(rt.quadrature_weights(bp.wavelength_micron)*bp.response*wise_vega_scale_jy(bp,vw,vf)
                    *1e-7/(4*np.pi*(g.D_L_MPC*1e6*rt.PC)**2*(1+g.Z)))
        band=spectral@projection
        rest_lam=bp.wavelength_micron*1e-6/(1+g.Z)
        source_cross=np.pi*(grains[:,None]*1e-6)**2*op.q_abs(rest_lam[None,:]*1e6,grains[:,None])
        source_spec=planck_lambda_si(rest_lam,g.SOURCE_T_K)/source_sed_norm
        source_band=L*float((source_spec*np.exp(-source_column*(wa@source_cross)))@projection)
        band[:len(source_band)]+=source_band
        source_MIR_peak[bp.name]=float(source_band.max()*1000)
        flux.append(band)
        tobs=(t[0]+np.arange(len(band))*numerics['dt']-offset)*(1+g.Z)
    flux=np.stack(flux);obs=g.load_observations();fit,_,_=profile(tobs,flux,obs)
    summary=dict(fit=fit,energy=solved['energy'],max_temperature_K=float(solved['T'].max()),
        sublimation_threshold_K=sub,static_opacity_gate_passed=bool(solved['T'].max()<sub),
        photon_probability_error=prob_error,iterations=len(solved['iterations']),
        final_iteration=solved['iterations'][-1],elapsed_seconds=time.monotonic()-started,
        direct_source_MIR_peak_mJy=source_MIR_peak,
        config=config,numerics=numerics,provenance_sha256=digest,
        status='grid candidate; static-opacity domain only; requires fine-resolution confirmation')
    pd.DataFrame(dict(time_obs_days=tobs,W1_Jy=flux[0],W2_Jy=flux[1])).to_csv(out/'response.csv',index=False)
    if args.plots: products(out,tobs,flux,fit,solved,mesh,obs)
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
