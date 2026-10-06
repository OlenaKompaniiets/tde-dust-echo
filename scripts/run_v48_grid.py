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
    result=[]
    for values in itertools.product(*axes.values()):
        c=dict(base,**dict(zip(axes,values)))
        if not (0<c['eta']<1 and c['tau_visc']>=0 and c['tau_C']>0 and c['tau_E']>0):
            raise ValueError('Invalid efficiency/time/optical depth')
        if not (0<c['theta']<90 and 0<=c['inc']<=90): raise ValueError('Invalid angles')
        if not (0<c['compact'][0]<c['compact'][1]<=c['extended'][0]<c['extended'][1]):
            raise ValueError('Invalid radial domains')
        if c not in result: result.append(c)
    return result

def identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]

@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
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

def report(folder):
    rows=records(folder)
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

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'configs/v48_grid.json')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--status',action='store_true')
    ap.add_argument('--retry-errors',action='store_true')
    ap.add_argument('--refine-top',type=int,default=0,help='Recalculate N best completed coarse models at fine resolution')
    args=ap.parse_args();spec=json.loads(args.config.read_text(encoding='utf-8'));configs=expand(spec)
    stars=Path(os.environ.get('UGC_STARS_ROOT',ROOT/'STARS_library-master'))
    files=set((ROOT/'src').glob('*.py'))|set((ROOT/'scripts').glob('*.py'))|{p for p in (ROOT/'data').rglob('*') if p.is_file()}
    files|={stars/c['stars'] for c in configs}
    missing=[str(p) for p in files if not p.exists()]
    if missing:raise FileNotFoundError('Missing inputs: '+str(missing))
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    import importlib.metadata as md
    versions={k:md.version(k) for k in ('numpy','scipy','pandas','astropy','matplotlib')}
    provenance=dict(spec=spec,hashes=hashes,python=platform.python_version(),packages=versions)
    run_id=identity(provenance);folder=ROOT/'results'/('v48_grid_'+run_id)
    if args.refine_top<0:raise ValueError('--refine-top must be nonnegative')
    resolution=spec.get('resolution','coarse')
    if resolution not in ('coarse','fine'):raise ValueError('Invalid resolution')
    if args.refine_top:
        completed=sorted((r for r in records(folder) if r['status']=='completed'),key=lambda r:r['fit']['chi2_total'])
        if not completed:raise RuntimeError('No completed coarse models to refine')
        configs=[r['config'] for r in completed[:args.refine_top]];folder=folder/'fine';resolution='fine'
    print('MODELS:',len(configs),'RESOLUTION:',resolution,'\nOUTPUT:',folder,flush=True)
    if args.dry_run:return
    if args.status:
        rows=records(folder)
        print('Saved models:',len(rows))
        for state in ('completed','unsupported_sublimation','error'):
            print(state, sum(r['status']==state for r in rows))
        valid=[r for r in rows if r['status']=='completed']
        if valid:print('Best chi2:',min(r['fit']['chi2_total'] for r in valid))
        return
    with lock(folder/'grid.lock'):
        atomic(folder/'manifest.json',provenance)
        for n,c in enumerate(configs,1):
            mid=identity(c);out=folder/'models'/mid;out.mkdir(parents=True,exist_ok=True)
            status=out/'status.json'
            if status.exists():
                prior=json.loads(status.read_text())
                if prior['status'] in ('completed','unsupported_sublimation') or not args.retry_errors:continue
            atomic(out/'config.json',c)
            command=[sys.executable,'-u',str(ROOT/'scripts/run_v48_grid_model.py'),'--config',str(out/'config.json'),'--output',str(out),'--resolution',resolution,'--seed',str(spec.get('seed',481))]
            print(f'[{n}/{len(configs)}] {mid} eta={c["eta"]} visc={c["tau_visc"]} tau={c["tau_C"]},{c["tau_E"]}',flush=True)
            start=time.monotonic()
            with (out/'run.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:code=process.wait()
                except KeyboardInterrupt:
                    process.terminate();process.wait();print('Stopped; rerun the same command to resume.');return
            result=dict(model_id=mid,config=c,output=str(out.relative_to(folder)),elapsed_seconds=time.monotonic()-start)
            if code==0:
                summary=json.loads((out/'summary.json').read_text());result.update(status='completed',fit=summary['fit'])
            else:
                log=(out/'run.log').read_text(encoding='utf-8',errors='replace')
                result['status']='unsupported_sublimation' if 'RuntimeError: SUBLIMATION:' in log else 'error'
                result['log']='run.log'
            atomic(status,result);report(folder)
        report(folder)
        print('Finished. Sublimation cases are outside this solver, NOT excluded physical models.',flush=True)

if __name__=='__main__':main()
