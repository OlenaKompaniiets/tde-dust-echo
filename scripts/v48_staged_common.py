"""Restartable, serial V4.8 grid; each model runs in an isolated subprocess."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, itertools, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def expand(spec):
    base=spec['base']; axes=spec['axes']
    allowed=set(base)
    if not set(axes)<=allowed: raise ValueError('Unknown grid parameter')
    if not axes or any(not isinstance(v,list) or not v for v in axes.values()):
        raise ValueError('Axes must be nonempty lists')
    result=[];seen=set()
    for values in itertools.product(*axes.values()):
        c=dict(base,**dict(zip(axes,values)))
        if not (0<c['eta']<1 and c['tau_visc']>=0 and c['tau_C']>0 and c['tau_E']>0):
            raise ValueError('Invalid efficiency/time/optical depth')
        if not (0<c['theta']<90 and 0<=c['inc']<=90): raise ValueError('Invalid angles')
        for k in ('compact','extended'):
            if not isinstance(c[k],list) or len(c[k])!=3:
                raise ValueError(k+' must be [Rin_pc, Rout_pc, p]; grid axes require a list of such triples')
        if not (0<c['compact'][0]<c['compact'][1]<=c['extended'][0]<c['extended'][1]):
            raise ValueError('Invalid radial domains')
        key=json.dumps(c,sort_keys=True)
        if key not in seen:result.append(c);seen.add(key)
    return result

def identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]

@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b')
    if path.stat().st_size==0:f.write(b'0');f.flush()
    f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise RuntimeError('Another grid process is using this folder')
    try: yield
    finally: f.close()  # OS releases lock, including after a crash.

def records(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('models/*/status.json'))]

def report(folder, rows=None):
    if rows is None:rows=records(folder)
    flat=[]
    for row in rows:
        r={k:row.get(k) for k in ('model_id','status','elapsed_seconds','output')}
        r.update(row.get('fit',{}));r.update({'parameter_'+k:json.dumps(v) for k,v in row['config'].items()})
        flat.append(r)
    if not flat:return
    keys=sorted(set().union(*(r.keys() for r in flat)))
    temp=folder/'grid_results.tmp'
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    os.replace(temp,folder/'grid_results.csv')
    valid=sorted((r for r in rows if r['status']=='completed'),key=lambda r:r['fit']['chi2_total'])
    if valid:
        atomic(folder/'best.json',valid[0])
        # Plot and epoch diagnostics use saved response only, never rerun RT.
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sys.path.insert(0,str(ROOT))
        from scripts.run_v4_model_grid import load_observations
        best=valid[0];obs=load_observations();resp=pd.read_csv(folder/best['output']/'response.csv')
        fig,ax=plt.subplots(2,2,figsize=(11,7),sharex='col',gridspec_kw={'height_ratios':[3,1]})
        table={'mjd':obs.mjd.to_numpy()}
        for i,b in enumerate(('W1','W2')):
            t=resp.time_obs_days.to_numpy()+best['fit']['t0_mjd']
            host=float(obs['baseline_'+b+'_mJy'].iloc[0]);f=resp[b+'_Jy'].to_numpy()*1000
            m=np.interp(obs.mjd,t,f,left=0,right=0)+host
            y=obs['F_'+b+'_Jy'].to_numpy()*1000;err=obs['sigma_'+b+'_Jy'].to_numpy()*1000
            residual=(y-m)/err
            ax[0,i].errorbar(obs.mjd,y,yerr=err,fmt='o',color='black',ms=4)
            ax[0,i].plot(t,f+host);ax[0,i].set(title=b,ylabel='Flux (mJy)',xlim=(obs.mjd.min()-100,obs.mjd.max()+100))
            ax[1,i].plot(obs.mjd,residual,'o');ax[1,i].axhline(0,color='grey');ax[1,i].set(xlabel='MJD',ylabel='Residual / sigma')
            table.update({b+'_model_mJy':m,b+'_data_mJy':y,b+'_sigma_mJy':err,b+'_residual_sigma':residual})
        fig.suptitle('V4.8 grid best so far: chi2 = %.3f'%best['fit']['chi2_total']);fig.tight_layout()
        fig.savefig(folder/'best_lightcurves.png',dpi=160);plt.close(fig)
        pd.DataFrame(table).to_csv(folder/'best_epochs.csv',index=False)
    print('STATUS:',{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},flush=True)
    if valid:print('BEST chi2:',valid[0]['fit']['chi2_total'],flush=True)
