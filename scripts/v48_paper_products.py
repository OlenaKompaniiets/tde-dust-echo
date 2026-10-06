"""Post-process saved fine solutions. Never runs the RT solver or alters fitting."""
from pathlib import Path
import json,sys
import numpy as np
if not hasattr(np,'trapezoid'):np.trapezoid=np.trapz
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v48_staged_common import atomic,records,identity
from v48_staged_plan import AXES,levels,valid_rows

def save(fig,out,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)

def make_products(out):
    out=Path(out);summary=json.loads((out/'summary.json').read_text());config=summary['config']
    from src.v4_opacity import make_grain_size_grid
    obs=pd.read_csv(ROOT/'data/UGC11487_WISE_TDE_model_input_22epochs.csv').sort_values('mjd').reset_index(drop=True);e=pd.read_csv(out/'epochs.csv');response=pd.read_csv(out/'response.csv')
    h=np.array([float(obs['baseline_'+b+'_mJy'].iloc[0]) for b in ('W1','W2')])
    f=np.stack([e[b+'_data_mJy'].to_numpy()-h[i] for i,b in enumerate(('W1','W2'))])
    sig=np.stack([e[b+'_sigma_mJy'].to_numpy() for b in ('W1','W2')])
    model=np.stack([e[b+'_model_mJy'].to_numpy()-h[i] for i,b in enumerate(('W1','W2'))])
    fig,ax=plt.subplots(figsize=(6.8,5.8))
    ax.errorbar(f[0],f[1],xerr=sig[0],yerr=sig[1],fmt='none',ecolor='0.6',alpha=.8,zorder=1)
    sc=ax.scatter(f[0],f[1],c=e.mjd,cmap='viridis',s=30,zorder=3)
    t=response.time_obs_days.to_numpy()+summary['fit']['t0_mjd'];mask=(t>=e.mjd.min())&(t<=e.mjd.max())
    ax.plot(response.W1_Jy[mask]*1000,response.W2_Jy[mask]*1000,color='#c06123',label='Physical model')
    ax.scatter(model[0],model[1],marker='x',s=22,color='#c06123',label='Model at WISE epochs')
    ax.set(xlabel='W1 excess (mJy)',ylabel='W2 excess (mJy)',title='Fixed-host excess; adopted photometric errors')
    ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='MJD');save(fig,out,'paper_flux_flux_errors')
    fig,ax=plt.subplots(figsize=(8,4.6))
    y1=e.W1_data_mJy.to_numpy();y2=e.W2_data_mJy.to_numpy()
    color=-2.5*np.log10((y1/309.540)/(y2/171.787))
    cerr=2.5/np.log(10)*np.sqrt((sig[0]/y1)**2+(sig[1]/y2)**2)
    mc=-2.5*np.log10((e.W1_model_mJy/309.540)/(e.W2_model_mJy/171.787))
    ax.errorbar(e.mjd,color,yerr=cerr,fmt='o',color='black',label='WISE (conditional independent errors)')
    ax.plot(e.mjd,mc,'s-',color='#087f8c',label='Model at WISE epochs');ax.set(xlabel='MJD',ylabel='W1-W2 (Vega mag)');ax.legend(fontsize=8);save(fig,out,'paper_colour_evolution')
    # Saved actual thermal solution, not reconstructed from MIR colour.
    with np.load(out/'thermal_solution.npz',allow_pickle=False) as z:
        u=z['time'];T=z['temperature'];N=z['numbers'];edges=z['radial_edges'];cr=z['cell_radial'];cm=z['cell_mu'];em=z['emitted_bin_power_W'].sum(axis=(1,2))
    a=make_grain_size_grid(float(config['amin']),float(config['amax']),T.shape[2])
    if len(a)!=T.shape[2]:raise ValueError('Grain axis mismatch')
    pd.DataFrame({'grain_index':np.arange(len(a)),'a_micron':a}).to_csv(out/'grain_sizes.csv',index=False)
    from src.v48_angular_rt import PC_DAYS
    radii=.75*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
    used=sorted(set(cr));times=[250.,750.,2000.];ais=[0,len(a)//2,len(a)-1]
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.1),sharey=True);tab=[]
    for ax,ai in zip(axs,ais):
        for phase in times:
            vals=[]
            for ir in used:
                cells=np.where(cr==ir)[0];r=radii[ir]
                v=np.array([np.interp(phase-r*PC_DAYS,u,T[:,ic,ai],left=0,right=0) for ic in cells])
                avg=np.average(v,weights=N[cells,ai]);vals.append(avg)
                tab.append(dict(t_local_rest_days=phase,r_pc=r,a_micron=a[ai],T_K=avg))
            # Break vacuum gap rather than draw temperature through empty space.
            for pop,sel in enumerate(([i for i,ir in enumerate(used) if radii[ir]<config['compact'][1]], [i for i,ir in enumerate(used) if radii[ir]>config['extended'][0]])):
                ax.plot([radii[used[i]] for i in sel],[vals[i] for i in sel],'o-',ms=3,color={250.:'C0',750.:'C1',2000.:'C2'}[phase],label=f'{phase:g} d' if pop==0 else None)
        ax.axhline(summary['sublimation_threshold_K'],ls=':',color='gray',lw=1)
        ax.set(xlabel='Radius (pc)',title=f'a = {a[ai]:.3g} micron');ax.legend(fontsize=8)
    axs[0].set_ylabel('Angular number-weighted T (K)');save(fig,out,'paper_temperature_radial_evolution')
    pd.DataFrame(tab).to_csv(out/'temperature_radial_evolution.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.8));time_local=np.linspace(0,5000,501);tab=[]
    chosen=[used[0],used[len(used)//2],used[-1]]
    for ir in chosen:
        cells=np.where(cr==ir)[0];mean=np.sum(T[:,cells,:]*N[None,cells,:],axis=(1,2))/N[cells,:].sum()
        vals=np.interp(time_local-radii[ir]*PC_DAYS,u,mean,left=0,right=0)
        ax.plot(time_local,vals,label=f'r = {radii[ir]:.3f} pc')
        for t,v in zip(time_local,vals):tab.append(dict(t_local_rest_days=t,r_pc=radii[ir],T_number_mean_K=v))
    ax.set(xlabel='Local rest-frame time from fallback peak (days)',ylabel='Number-weighted temperature (K)');ax.legend();save(fig,out,'paper_temperature_time')
    pd.DataFrame(tab).to_csv(out/'temperature_time.csv',index=False)
    pd.DataFrame([{'quantity':k,'value':v} for k,v in summary['energy'].items()]).to_csv(out/'energy_ledger.csv',index=False)
    diagnostics={}
    t=e.mjd.to_numpy()
    for i,b in enumerate(('W1','W2')):
        for name,flux in [('data',f[i]),('model',model[i])]:
            pos=np.maximum(flux,0);area=float(np.trapezoid(pos,t));cen=float(np.trapezoid(t*pos,t)/area) if area>0 else None
            width=float(np.sqrt(np.trapezoid((t-cen)**2*pos,t)/area)) if area>0 else None
            diagnostics[b+'_'+name]=dict(positive_fluence_mJy_observer_day=area,centroid_MJD=cen,width_observer_day=width)
    atomic(out/'paper_products.json',dict(status='complete',chi2=summary['fit'],diagnostics=diagnostics,
        conventions=['Temperatures are actual saved equilibrium solutions; radial profiles use volume-mean representative radius and angular number weights.',
        '3D thermal_evolution_3d is a sampling of an effective smooth medium, not discrete physical clumps.',
        'Flux-flux errors are adopted photometric sigma with fixed host. Host uncertainty/covariance is not included in the likelihood.',
        'Colour error uses first-order independent-band propagation.',
        'Fluence moments use only WISE epochs and positive excess; not bolometric energy or published estimator uncertainties.',
        'No diagnostics in this file enter model selection; only photometric chi-square.',
        'Static opacity: unsupported_sublimation means solver domain limitation, not zero MIR.']))

def selection_report(folder,spec):
    folder=Path(folder);rows=records(folder);fine=records(folder/'fine');lv=levels(spec);by={r['model_id']:r for r in rows};table=[]
    for star in spec['axes']['stars']:
        rr=[r for r in rows if r['config']['stars']==star];vv=valid_rows(rr)
        table.append(dict(stars=star,n_completed=sum(r['status']=='completed' for r in rr),n_unsupported=sum(r['status']=='unsupported_sublimation' for r in rr),n_error=sum(r['status']=='error' for r in rr),best_chi2=vv[0]['fit']['chi2_total'] if vv else None))
    pd.DataFrame(table).to_csv(folder/'search_coverage.csv',index=False)
    tab=[]
    for r in fine:
        c=r['config'];coarse=by.get(r['model_id'],{}).get('fit',{}).get('chi2_total');fc=r.get('fit',{}).get('chi2_total')
        boundaries=[k for k in AXES if c[k]==lv[k][0] or c[k]==lv[k][-1]]
        tab.append(dict(model_id=r['model_id'],stars=c['stars'],fine_status=r['status'],coarse_chi2=coarse,fine_chi2=fc,delta_chi2=fc-coarse if fc is not None and coarse is not None else None,edge_levels=','.join(boundaries),t0_boundary=r.get('fit',{}).get('t0_boundary')))
    pd.DataFrame(tab).to_csv(folder/'coarse_fine_comparison.csv',index=False)
    pd.DataFrame([dict(parameter=k,values=json.dumps(v),note='Discrete enumerated values; geometry entries are [Rin_pc,Rout_pc,p]') for k,v in spec['axes'].items() if k!='stars']).to_csv(folder/'parameter_domain.csv',index=False)
    (folder/'STARS_used.txt').write_text('\n'.join(spec['axes']['stars'])+'\n')
    atomic(folder/'search_summary.json',dict(scope='Budgeted search of a finite conditional domain, not exhaustive posterior or global minimum proof.',
        all_stars=len(spec['axes']['stars']),total_checkpoints=len(rows),fine_candidates=len(fine),
        review_required=['Check search_coverage for unsupported-only sources.', 'Check coarse/fine ranking and boundary levels; expand the domain if minima approach edges.', 'Inspect early-rise residuals and energy ledger.', 'Check numerical convergence and independent ray seed before final physical parameter claims.']))

if __name__=='__main__':make_products(Path(sys.argv[1]))
