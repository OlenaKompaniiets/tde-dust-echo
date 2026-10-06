"""Parallel bounded coordinate polling for the two existing V4.8 candidates.
One global pool, deterministic complete polls, checkpointed accepted states.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,threading,time,sys,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import run_final as rf
P=Path(__file__).resolve().parent
ORDER=['theta','tau_C','tau_E','eta','tau_visc','inc']
INITIAL={'eta':.005,'tau_visc':15.,'tau_C':.015,'tau_E':.10,'theta':1.5,'inc':1.5}
_guard=threading.Lock();_locks={}
def kernel_lock(c):
 # These parameters determine the shared kernel signature; serialize matching
 # groups because the unchanged solver writes a fixed .tmp file per kernel.
 keys=('stars','compact','extended','tau_C','tau_E','theta','composition','amin','amax','q')
 key=json.dumps({k:c[k] for k in keys},sort_keys=True)
 with _guard:return _locks.setdefault(key,threading.Lock())
def evaluate(c):
 with kernel_lock(c):s,folder=rf.evaluate(c)
 return dict(config=c,chi2=s['fit']['chi2_total'],t0_mjd=s['fit']['t0_mjd'],path=str(folder))
def poll(c,steps):
 points=[];seen=set()
 for k in ORDER:
  for sign in (-1,1):
   q=copy.deepcopy(c);lo,hi=rf.BOUNDS[k];q[k]=float(max(lo,min(hi,c[k]+sign*steps[k])))
   signature=json.dumps(q,sort_keys=True)
   if q[k]!=c[k] and signature not in seen:points.append(q);seen.add(signature)
 return points
def choose(best,results,steps,tol):
 winner=min([best]+results,key=lambda r:r['chi2']);gain=best['chi2']-winner['chi2']
 # Preserve even a small improvement, but shrink after an insufficient poll.
 newsteps=steps.copy()
 if gain<=tol:newsteps={k:v/2 for k,v in steps.items()}
 done=gain<=tol and all(steps[k]<=INITIAL[k]/16*(1+1e-10) for k in INITIAL)
 return winner,newsteps,done,gain
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workers',type=int,default=12,choices=range(1,17));ap.add_argument('--max-evals',type=int,default=300);ap.add_argument('--branch',choices=['both','local','grid'],default='both');ap.add_argument('--plots',action='store_true')
 args=ap.parse_args();rf.prepare()
 if args.max_evals<12:ap.error('--max-evals must be at least 12')
 out=rf.ROOT/'results/v48_parallel_final';out.mkdir(parents=True,exist_ok=True)
 # Lock stops two parallel coordinators from altering the same checkpoints.
 # OS lock is released on exit/crash; no persistent stale-lock deletion needed.
 lock=(out/'coordinator.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:
  if os.name=='nt':
   import msvcrt;msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except OSError:raise RuntimeError('Another parallel coordinator is running.')
 branches=['local','grid'] if args.branch=='both' else [args.branch];states={};tol=.005
 for b in branches:
  checkpoint=out/b/'checkpoint.json'
  if checkpoint.exists():st=rf.read(checkpoint)
  else:
   source=rf.ROOT/'results/v48_paper_final'/b/'baseline.json'
   if not source.exists():raise RuntimeError(f'Missing {source}; keep existing baseline results.')
   old=rf.read(source);st=dict(best={k:old[k] for k in ('config','chi2','t0_mjd','path')},steps=INITIAL.copy(),evaluations=0,round=0,poll_converged=False,history=[],initial_chi2=old['chi2'])
  checked=evaluate(st['best']['config'])
  if abs(checked['chi2']-st['best']['chi2'])>1e-6:raise RuntimeError('Baseline reproduction changed; inspect provenance before continuing.')
  st['best']=checked;states[b]=st
  print('START',b,'chi2',checked['chi2'],'completed polls',st['round'],flush=True)
 try:
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
   while True:
    jobs={};buffers={};active=[]
    for b,st in states.items():
     if st['poll_converged']:continue
     points=poll(st['best']['config'],st['steps'])
     if st['evaluations']+len(points)>args.max_evals:continue
     buffers[b]=[None]*len(points);active.append(b)
     for i,cfg in enumerate(points):jobs[pool.submit(evaluate,cfg)]=(b,i)
    if not jobs:break
    started=time.monotonic()
    # A complete poll is committed only after all its evaluations succeeded.
    # Ctrl+C waits for the current batch. Completed RT files remain reusable.
    for fut in as_completed(jobs):
     b,i=jobs[fut];r=fut.result();buffers[b][i]=r
     print('DONE',b,'chi2',round(r['chi2'],6),'round',states[b]['round']+1,flush=True)
    for b in active:
     st=states[b];st['best'],st['steps'],st['poll_converged'],gain=choose(st['best'],buffers[b],st['steps'],tol)
     st['evaluations']+=len(buffers[b]);st['round']+=1
     st['history'].append(dict(round=st['round'],best_chi2=st['best']['chi2'],gain=gain,steps=st['steps'],models=buffers[b]))
     rf.write(out/b/'checkpoint.json',st)
     rf.write(out/b/'best.json',dict(**st['best'],poll_converged=st['poll_converged'],evaluations=st['evaluations'],steps=st['steps'],scope='bounded coordinate-poll stationarity only; not Powell convergence or a global minimum'))
     print('BEST',b,round(st['best']['chi2'],6),'evals',st['evaluations'],'poll_converged',st['poll_converged'],flush=True)
    print('BATCH minutes',round((time.monotonic()-started)/60,2),flush=True)
 except KeyboardInterrupt:
  print('Interrupted: finished model files preserved. Repeat the same command to resume.',flush=True);return
 if args.plots:
  # Matplotlib post-processing is deliberately serial.
  import subprocess
  from plot_all import all_plots
  for b,st in states.items():
   _,folder=rf.evaluate(st['best']['config'],plots=True)
   subprocess.run([sys.executable,str(rf.ROOT/'scripts/v48_paper_products.py'),str(folder)],cwd=rf.ROOT,check=True)
   if all_plots(folder):raise RuntimeError('Missing inputs for full figure set.')
 for b,st in states.items():
  print(b,'STOP',('coordinate-poll criterion' if st['poll_converged'] else 'evaluation budget'), 'BEST',st['best']['chi2'],flush=True)
 with zipfile.ZipFile(out/'SEND_PARALLEL_FOR_REVIEW.zip','w',zipfile.ZIP_DEFLATED) as z:
  for b,st in states.items():
   for name in ('checkpoint.json','best.json'):
    f=out/b/name
    if f.exists():z.write(f,f'{b}/{name}')
   model=Path(st['best']['path'])
   for name in ('summary.json','config.json','provenance.json','response.csv'):
    if (model/name).exists():z.write(model/name,f'{b}/model/{name}')
 print('FINISHED',out/'SEND_PARALLEL_FOR_REVIEW.zip',flush=True)
if __name__=='__main__':main()
