"""Two equal-budget local searches and the original grid paper products."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,threading,zipfile,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
LOCK=threading.Lock()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(p)
def log(*s):
 with LOCK:print(*s,flush=True)
def products(folder):
 script=PROJECT/'scripts/v48_paper_products.py'
 subprocess.run([sys.executable,str(script),str(folder)],cwd=PROJECT,check=True)
def evaluate(cfg,root,angular=False,plots=False):
 ident=hashlib.sha256(json.dumps([cfg,angular],sort_keys=True).encode()).hexdigest()[:20]
 folder=root/'models'/ident;folder.mkdir(parents=True,exist_ok=True);cp=folder/'config.json'
 write(cp,cfg)
 if (folder/'summary.json').exists() and (not plots or (folder/'thermal_solution.npz').exists()):return folder,read(folder/'summary.json')['fit']['chi2_total']
 command=[sys.executable,'-u',str(ROOT/'solver.py'),'--config',str(cp),'--output',str(folder),'--resolution','fine','--seed','481']
 if angular:command+=['--angular-check']
 if plots:command+=['--plots']
 with (folder/'run.log').open('w',encoding='utf-8') as out:r=subprocess.run(command,cwd=PROJECT,stdout=out,stderr=subprocess.STDOUT)
 if r.returncode:
  # Numerical failures must not be silently treated as bad physical fits.
  raise RuntimeError('Solver failed: '+str(folder/'run.log'))
 return folder,read(folder/'summary.json')['fit']['chi2_total']
def search(name,root,plan):
 cfg=read(ROOT/(name+'_start.json'));history=[];bestfolder,best=evaluate(cfg,root)
 history.append(dict(evaluation=1,chi2=best,path=str(bestfolder),accepted=True))
 for sweep in range(plan['sweeps']):
  for key,initial in plan['steps'].items():
   current=cfg.copy();options=[(best,cfg,bestfolder)]
   for sign in [-1,1]:
    trial=current.copy();lo,hi=plan['bounds'][key];trial[key]=max(lo,min(hi,current[key]+sign*initial/(2**sweep)))
    if trial==current:continue
    folder,chi=evaluate(trial,root);options.append((chi,trial,folder))
    history.append(dict(evaluation=len(history)+1,sweep=sweep+1,parameter=key,chi2=chi,path=str(folder)))
    write(root/'history.json',history);log(name,'eval',len(history),'chi2',round(chi,5),'best',round(min(best,chi),5))
   best,cfg,bestfolder=min(options,key=lambda x:x[0])
  write(root/'best.json',dict(config=cfg,chi2=best,path=str(bestfolder),sweep=sweep+1))
 bestfolder,best=evaluate(cfg,root,plots=True);products(bestfolder)
 write(root/'best.json',dict(config=cfg,chi2=best,path=str(bestfolder),boundaries=[k for k,(lo,hi) in plan['bounds'].items() if min(abs(cfg[k]-lo),abs(cfg[k]-hi))<1e-8]))
 return bestfolder

def package(out,folders):
 with zipfile.ZipFile(out/'SEND_PAIR_FOR_REVIEW.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p.suffix in ['.json','.csv'] and 'kernel_cache' not in p.parts:z.write(p,p.relative_to(out))
  for folder in folders:
   for p in folder.iterdir():
    if p.suffix in ['.pdf','.png']:z.write(p,p.relative_to(out))
 log('SEND',out/'SEND_PAIR_FOR_REVIEW.zip')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['figures','angular','refine'],default='angular');ap.add_argument('--solution',type=Path);ap.add_argument('--workers',type=int,choices=[1,2],default=2);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
 for rel in ['src/v48_angular_rt.py','scripts/v48_paper_products.py','scripts/v48_staged_common.py','scripts/v48_staged_plan.py']:
  if not (PROJECT/rel).exists():raise SystemExit('Missing '+str(PROJECT/rel))
 if args.mode=='figures':
  if args.solution:folder=args.solution.resolve()
  else:
   candidates=sorted((PROJECT/'results').glob('v48_local_reproduce_*/candidate/thermal_solution.npz'))
   if not candidates:raise SystemExit('Use --solution PATH to existing candidate with thermal_solution.npz')
   folder=candidates[-1].parent
  log('FIGURES FOR',folder)
  if not args.check_only:products(folder)
  return
 plan=read(ROOT/'search_plan.json')
 # New cache namespace whenever code, data or plan changes; resume only identical inputs.
 h=hashlib.sha256()
 paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.json'))+list((PROJECT/'src').glob('*.py'))+list((PROJECT/'scripts').glob('*.py'))+[p for p in (PROJECT/'data').rglob('*') if p.is_file()]
 for p in sorted(paths):h.update(str(p.relative_to(PROJECT)).encode());h.update(p.read_bytes())
 # The solver also validates its full source/data/STARS fingerprint on execution.
 out=PROJECT/'results'/('v48_pair_'+args.mode+'_'+h.hexdigest()[:12]);out.mkdir(parents=True,exist_ok=True)
 log('OUTPUT',out,'MODE',args.mode,'WORKERS',args.workers)
 if args.check_only:return
 def task(name):
  root=out/name;root.mkdir(exist_ok=True)
  if args.mode=='refine':return search(name,root,plan)
  cfg=read(ROOT/(name+'_start.json'));folders=[];results=[]
  for angular in [False,True]:
   folder,chi=evaluate(cfg,root,angular=angular,plots=True);products(folder);folders.append(folder)
   results.append(dict(angular_check=angular,chi2=chi,path=str(folder)))
   log(name,'angular',angular,'chi2',chi)
  write(root/'angular_comparison.json',dict(results=results,delta_chi2=results[1]['chi2']-results[0]['chi2'],note='Angular sensitivity only; radial, spectral and time convergence not tested.'))
  return folders
 with ThreadPoolExecutor(max_workers=args.workers) as pool:result=list(pool.map(task,['local','grid']))
 folders=[p for r in result for p in (r if isinstance(r,list) else [r])]
 package(out,folders)
if __name__=='__main__':main()
