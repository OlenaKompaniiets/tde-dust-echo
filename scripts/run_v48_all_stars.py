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

def main():
    import importlib.metadata as md
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    ap=argparse.ArgumentParser(description='Full 141-STARS V4.8 grid; no change to thermal physics')
    ap.add_argument('--config',type=Path,default=ROOT/'configs/v48_all_stars.json')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--status',action='store_true')
    ap.add_argument('--retry-errors',action='store_true')
    ap.add_argument('--refine-top',type=int,default=0)
    args=ap.parse_args()
    if args.workers<1 or args.refine_top<0:ap.error('workers must be positive; refine-top nonnegative')
    spec=json.loads(args.config.read_text(encoding='utf-8-sig'));configs=expand(spec)
    nstars=len({c['stars'] for c in configs})
    if nstars!=spec.get('expected_stars_count',nstars):raise ValueError('STARS count differs from manifest')
    print('STARS curves:',nstars,flush=True)
    resolution=spec.get('resolution','coarse')
    if resolution not in ('coarse','fine'):raise ValueError('Invalid resolution')
    stars=Path(os.environ.get('UGC_STARS_ROOT',ROOT/'STARS_library-master'))
    required=['scripts/run_v48_all_stars.py','scripts/run_v48_all_stars_model.py','scripts/v48_large_lock.py',
              'scripts/run_v45_direct_absorption.py','scripts/run_v4_model_grid.py','src/v48_angular_rt.py',
              'data/UGC11487_WISE_TDE_model_input_22epochs.csv']
    files={ROOT/x for x in required}|set((ROOT/'src').glob('*.py'))|{p for p in (ROOT/'data').rglob('*') if p.is_file()}
    files|={stars/c['stars'] for c in configs}
    missing=[str(p) for p in files if not p.is_file()]
    if missing:raise FileNotFoundError('Extract this update INSIDE your working project. Missing: '+str(missing))
    hashes={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'external_STARS/'+p.relative_to(stars).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    try: versions={k:md.version(k) for k in ('numpy','scipy','pandas','astropy','matplotlib')}
    except md.PackageNotFoundError as exc:
        raise RuntimeError('Required package missing in this Python environment: '+str(exc)+'. Activate v4_dust.') from exc
    provenance=dict(spec=spec,hashes=hashes,python=platform.python_version(),packages=versions)
    folder=ROOT/'results'/('v48_all_stars_'+identity(provenance))
    if args.refine_top:
        valid=sorted((r for r in records(folder) if r['status']=='completed'),key=lambda r:r['fit']['chi2_total'])
        if not valid:raise RuntimeError('No completed LARGE grid models; do not change config/code before refinement')
        configs=[r['config'] for r in valid[:args.refine_top]];folder=folder/'fine';resolution='fine'
    print('MODELS:',len(configs),'RESOLUTION:',resolution,'WORKERS:',args.workers,'\nOUTPUT:',folder,flush=True)
    if args.dry_run:return
    if args.status:
        rows=records(folder)
        print({s:sum(r['status']==s for r in rows) for s in ('completed','unsupported_sublimation','error')})
        valid=[r for r in rows if r['status']=='completed']
        if valid:print('BEST chi2:',min(r['fit']['chi2_total'] for r in valid))
        return
    import threading
    stopped=threading.Event();processes={};mutex=threading.Lock()
    def evaluate(c):
        mid=identity(c);out=folder/'models'/mid;out.mkdir(parents=True,exist_ok=True)
        atomic(out/'config.json',c);start=time.monotonic()
        command=[sys.executable,'-u',str(ROOT/'scripts/run_v48_all_stars_model.py'),'--config',str(out/'config.json'),'--output',str(out),'--resolution',resolution,'--seed',str(spec.get('seed',481))]
        result=dict(model_id=mid,config=c,output=str(out.relative_to(folder)))
        try:
            with (out/'run.log').open('w',encoding='utf-8') as log:
                with mutex:
                    if stopped.is_set():return None
                    p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes[mid]=p
                code=p.wait()
                with mutex:processes.pop(mid,None)
            if stopped.is_set():return None
            if code==0:
                summary=json.loads((out/'summary.json').read_text(encoding='utf-8-sig'))
                import math
                if not (out/'response.csv').is_file() or not math.isfinite(summary['fit']['chi2_total']):
                    raise ValueError('Missing response or nonfinite chi2')
                result.update(status='completed',fit=summary['fit'])
            else:
                logtext=(out/'run.log').read_text(encoding='utf-8',errors='replace')
                result.update(status='unsupported_sublimation' if 'RuntimeError: SUBLIMATION:' in logtext else 'error',error_tail=logtext[-2500:])
        except Exception as exc:result.update(status='error',error_tail=repr(exc))
        result['elapsed_seconds']=time.monotonic()-start;atomic(out/'status.json',result);return result
    with lock(folder/'grid.lock'):
        atomic(folder/'manifest.json',provenance)
        pending=[]
        for c in configs:
            out=folder/'models'/identity(c);status=out/'status.json'
            if status.exists():
                prior=json.loads(status.read_text())
                if prior['status'] in ('completed','unsupported_sublimation'):
                    if prior['status']!='completed' or ((out/'summary.json').exists() and (out/'response.csv').exists()):continue
                elif not args.retry_errors:continue
            pending.append(c)
        print('Remaining:',len(pending),flush=True)
        saved={r['model_id']:r for r in records(folder)}
        best_chi=min((r['fit']['chi2_total'] for r in saved.values() if r['status']=='completed'),default=float('inf'))
        last_report=time.monotonic();since_report=0
        pool=ThreadPoolExecutor(max_workers=args.workers);active={};queue=iter(pending);consecutive_errors=0
        def fill():
            while len(active)<args.workers:
                c=next(queue,None)
                if c is None:break
                print('START',identity(c),'eta',c['eta'],'visc',c['tau_visc'],'r',c['compact'],c['extended'],flush=True)
                active[pool.submit(evaluate,c)]=c
        try:
            fill()
            while active:
                done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                if not done:
                    print('Running:',len(active),'models; details in models/<id>/run.log',flush=True);continue
                improved=False
                for future in done:
                    active.pop(future);result=future.result()
                    if result is None:continue
                    saved[result['model_id']]=result;since_report+=1
                    chi=result.get('fit',{}).get('chi2_total',float('inf'))
                    if chi<best_chi:best_chi=chi;improved=True
                    print('PROGRESS',len(saved),'/',len(configs),'BEST',best_chi,flush=True)
                    print('DONE',result['model_id'],result['status'],result.get('fit',{}).get('chi2_total',''),flush=True)
                    consecutive_errors=consecutive_errors+1 if result['status']=='error' else 0
                    if result['status']=='error':print(result.get('error_tail','')[-1500:],flush=True)
                if improved or since_report>=1000 or time.monotonic()-last_report>=600:
                    try:report(folder,list(saved.values()))
                    except Exception as exc:print('Report warning (checkpoints retained):',repr(exc),flush=True)
                    last_report=time.monotonic();since_report=0
                if consecutive_errors>=3:raise RuntimeError('Stopped after 3 technical errors. Fix the shown error and resume with --retry-errors.')
                fill()
        except (KeyboardInterrupt,RuntimeError):
            stopped.set()
            with mutex:
                for p in processes.values():
                    if p.poll() is None:p.terminate()
            print('Stopped. Completed checkpoints retained; unfinished models restart on next launch.',flush=True)
            raise
        finally:pool.shutdown(wait=True)
        try:report(folder,list(saved.values()))
        except Exception as exc:print('Report warning:',repr(exc))
        print('Finished. Sublimation statuses are outside solver scope, not physical exclusions.',flush=True)
if __name__=='__main__':main()
