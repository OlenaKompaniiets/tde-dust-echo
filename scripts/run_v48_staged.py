"""V4.8 staged search: 141 sources, 64 balanced starts/source, adaptive follow-up."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,math,platform,shutil,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import importlib.metadata as md
from v48_staged_common import atomic,identity,lock,records,report
from v48_staged_plan import stage1,stage2,fine_selection,levels,valid_rows
ROOT=Path(__file__).resolve().parents[1]
RUNNER='scripts/run_v48_all_stars_model.py'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def provenance(spec):
    stars=Path(os.environ.get('UGC_STARS_ROOT',ROOT/'STARS_library-master'))
    paths={ROOT/RUNNER,ROOT/'scripts/v48_large_lock.py',ROOT/'scripts/run_v45_direct_absorption.py',ROOT/'scripts/run_v4_model_grid.py'}
    paths|=set((ROOT/'src').glob('*.py'))|{p for p in (ROOT/'data').rglob('*') if p.is_file()}
    paths|={stars/s for s in spec['axes']['stars']}
    if not (ROOT/'src/v48_angular_rt.py').is_file():raise FileNotFoundError('Extract update INSIDE UGC11487_V4_FIXED (src, data, STARS_library-master must exist).')
    hashes={}
    for p in sorted(paths):
        if not p.is_file():raise FileNotFoundError(str(p))
        name=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'external_STARS/'+p.relative_to(stars).as_posix()
        hashes[name]=sha(p)
    return dict(hashes=hashes,python=platform.python_version(),packages={k:md.version(k) for k in ('numpy','scipy','pandas','astropy','matplotlib')},seed=481,resolution='coarse')

def compatible(old,prov):
    # Fail closed: exact numerical environment and ALL physical/input hashes required.
    return (old.get('python')==prov['python'] and old.get('packages')==prov['packages']
        and old.get('spec',{}).get('seed',481)==481
        and old.get('spec',{}).get('resolution','coarse')=='coarse'
        and all(old.get('hashes',{}).get(k)==v for k,v in prov['hashes'].items()))

def in_domain(c,spec):
    return set(c)==set(spec['base']) and all(c[k] in vals for k,vals in spec['axes'].items()) and all(c[k]==v for k,v in spec['base'].items() if k not in spec['axes'])

def import_legacy(folder,prov,spec):
    audit=[];count=0
    for src in sorted((ROOT/'results').glob('v48_all_stars_*')):
        if not (src/'manifest.json').exists():continue
        old=read(src/'manifest.json')
        if not compatible(old,prov):audit.append(dict(folder=str(src),status='not_reused_fingerprint_mismatch'));continue
        # Also prevents copying while the old scheduler is still writing.
        with lock(src/'grid.lock'):
            for status in sorted(src.glob('models/*/status.json')):
                r=read(status);c=r.get('config',{});mid=identity(c)
                if r.get('model_id')!=mid or not in_domain(c,spec):continue
                if r.get('status') not in ('completed','unsupported_sublimation'):continue
                dst=folder/'models'/mid
                if (dst/'status.json').exists():continue
                if r['status']=='completed':
                    if not all((status.parent/x).is_file() for x in ('summary.json','response.csv','provenance.json')):continue
                    summ=read(status.parent/'summary.json');p=read(status.parent/'provenance.json')
                    if summ.get('provenance_sha256')!=p.get('sha256'):continue
                    if summ.get('config')!=c or summ.get('numerics',{}).get('seed')!=481 or summ['numerics'].get('dt')!=40.:continue
                    if summ.get('fit')!=r.get('fit') or not math.isfinite(r['fit']['chi2_total']):continue
                    if not all(p.get('hashes',{}).get(k)==v for k,v in prov['hashes'].items() if k!='scripts/v48_large_lock.py' and not ('STARS_library-master/input/' in k or k.startswith('external_STARS/'))):continue
                    star_keys=[k for k in prov['hashes'] if k.endswith('/'+c['stars']) or k=='external_STARS/'+c['stars']]
                    if not star_keys or any(p.get('hashes',{}).get(k)!=prov['hashes'][k] for k in star_keys):continue
                dst.mkdir(parents=True,exist_ok=True)
                for f in status.parent.iterdir():
                    if f.is_file() and f.name!='status.json':shutil.copy2(f,dst/f.name)
                r.update(output='models/'+mid,reused_from=str(status.parent));atomic(dst/'status.json',r);count+=1
            # Runner fingerprint is unchanged; kernel names remain valid.
            for f in (src/'kernel_cache').glob('*.npz'):
                dst=folder/'kernel_cache'/f.name;dst.parent.mkdir(exist_ok=True)
                if not dst.exists():shutil.copy2(f,dst)
        audit.append(dict(folder=str(src),status='compatible'))
    atomic(folder/'reuse_audit.json',dict(imported_this_launch=count,sources=audit))
    print('REUSED completed/unsupported checkpoints:',count,flush=True)

def execute(configs,folder,resolution,args,plots=False):
    folder.mkdir(parents=True,exist_ok=True);stop=threading.Event();mutex=threading.Lock();processes={}
    wanted={identity(c) for c in configs}
    saved={r['model_id']:r for r in records(folder)}
    def complete(r):
        if r['status']=='unsupported_sublimation':return True
        out=folder/'models'/r['model_id']
        return r['status']=='completed' and all((out/x).is_file() for x in ['summary.json','response.csv']+(['thermal_solution.npz','paper_products.json'] if plots else []))
    finished={mid for mid in wanted if mid in saved and complete(saved[mid])}
    pending=[c for c in configs if identity(c) not in finished]
    if any(saved.get(identity(c),{}).get('status')=='error' for c in pending) and not args.retry_errors:
        raise RuntimeError('Technical errors remain. Read per-model run.log, fix cause, then add --retry-errors.')
    print('SCHEDULED',len(configs),'CACHED',len(configs)-len(pending),'TO RUN',len(pending),resolution,flush=True)
    def evaluate(c):
        mid=identity(c);out=folder/'models'/mid;out.mkdir(parents=True,exist_ok=True);atomic(out/'config.json',c)
        r=dict(model_id=mid,config=c,output='models/'+mid);start=time.monotonic()
        cmd=[sys.executable,'-u',str(ROOT/RUNNER),'--config',str(out/'config.json'),'--output',str(out),'--resolution',resolution,'--seed','481']
        if plots:cmd.append('--plots')
        try:
            with (out/'run.log').open('a',encoding='utf-8') as log:
                with mutex:
                    if stop.is_set():return None
                    p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes[mid]=p
                code=p.wait()
                with mutex:processes.pop(mid,None)
            if stop.is_set():return None
            if code:
                txt=(out/'run.log').read_text(encoding='utf-8',errors='replace')
                r.update(status='unsupported_sublimation' if 'RuntimeError: SUBLIMATION:' in txt else 'error',error_tail=txt[-2500:])
            else:
                summ=read(out/'summary.json')
                if not math.isfinite(summ['fit']['chi2_total']) or not (out/'response.csv').exists():raise ValueError('Invalid model output')
                r.update(status='completed',fit=summ['fit'])
                if plots:
                    with (out/'run.log').open('a',encoding='utf-8') as log:
                        with mutex:
                            if stop.is_set():return None
                            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/v48_paper_products.py'),str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes[mid]=p
                        plot_code=p.wait()
                        with mutex:processes.pop(mid,None)
                    if plot_code:raise RuntimeError('Paper products failed; see run.log')
        except Exception as exc:r.update(status='error',error_tail=repr(exc))
        r['elapsed_seconds']=time.monotonic()-start;atomic(out/'status.json',r);return r
    best=min((r['fit']['chi2_total'] for r in saved.values() if r['status']=='completed'),default=float('inf'))
    started=time.monotonic();done_new=0;last_report=started;errors=0
    pool=ThreadPoolExecutor(max_workers=args.workers);active={};queue=iter(pending)
    def fill():
        while len(active)<args.workers:
            c=next(queue,None)
            if c is None:break
            print('START',identity(c),'STARS',c['stars'],'eta',c['eta'],'visc',c['tau_visc'],'tau',c['tau_C'],c['tau_E'],'r',c['compact'],c['extended'],flush=True)
            active[pool.submit(evaluate,c)]=c
    try:
        fill()
        while active:
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            if not done:print('RUNNING',len(active),'models; logs in',folder/'models',flush=True);continue
            for fut in done:
                active.pop(fut);r=fut.result()
                if r is None:continue
                saved[r['model_id']]=r;done_new+=1
                chi=r.get('fit',{}).get('chi2_total',float('inf'));best=min(best,chi)
                if complete(r):finished.add(r['model_id'])
                settled=len(finished)
                rate=done_new/max(time.monotonic()-started,1)*3600
                remaining=max(0,len(pending)-done_new);eta=remaining/rate if rate else float('inf')
                print('DONE',r['model_id'],r['status'],'chi2_current',chi,'BEST',best,'STAGE',str(settled)+'/'+str(len(configs)),'rate_per_hour',round(rate,1),'stage_ETA_hours',round(eta,2),flush=True)
                errors=errors+1 if r['status']=='error' else 0
                if r['status']=='error':print(r.get('error_tail'),flush=True)
            if time.monotonic()-last_report>600:
                report(folder,list(saved.values()));last_report=time.monotonic()
            if errors>=3:raise RuntimeError('Three consecutive technical errors; stopped. See run.log.')
            fill()
    except (KeyboardInterrupt,Exception):
        stop.set()
        with mutex:
            for p in processes.values():
                if p.poll() is None:p.terminate()
        print('STOPPED. Completed checkpoints retained.',flush=True);raise
    finally:pool.shutdown(wait=True)
    report(folder,list(saved.values()))
    if not wanted<=finished:raise RuntimeError('Stage has missing/error outputs; next stage NOT selected.')

def freeze(path,configs,extra=None):
    if path.exists():return read(path)['configs']
    atomic(path,dict(configs=configs,**(extra or {})));return configs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--stage',choices=['all','1','2','fine'],default='all')
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--status',action='store_true');ap.add_argument('--retry-errors',action='store_true')
    args=ap.parse_args()
    if args.workers<1:ap.error('workers must be >= 1')
    spec=read(ROOT/'configs/v48_staged_domain.json')
    if len(set(spec['axes']['stars']))!=141:raise ValueError('Expected exactly 141 distinct STARS files')
    prov=provenance(spec);strategy=dict(version=1,domain=spec,stage1_per_star=64,stage2_explore_per_star=16,seed_design=[48101,48102],search_code={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('v48_staged_plan.py'),Path(__file__).with_name('v48_staged_common.py'),Path(__file__).with_name('v48_paper_products.py'))})
    folder=ROOT/'results'/('v48_staged_'+identity(dict(provenance=prov,strategy=strategy)))
    first=stage1(spec)
    print('OUTPUT',folder,'\nSTAGE 1:',len(first),'models, all 141 STARS. STAGE 2 <= 5640 new scheduled points; FINE <=20.',flush=True)
    if args.dry_run:return
    if args.status:
        rows=records(folder) if folder.exists() else [];print('CHECKPOINTS',len(rows))
        for s in ('completed','unsupported_sublimation','error'):print(s,sum(r['status']==s for r in rows))
        if valid_rows(rows):print('BEST',valid_rows(rows)[0]['fit']['chi2_total'])
        return
    with lock(folder/'grid.lock'):
        atomic(folder/'manifest.json',dict(**prov,strategy=strategy))
        if not (folder/'import_done.json').exists():
            import_legacy(folder,prov,spec);atomic(folder/'import_done.json',{'finished':True})
        first=freeze(folder/'stage1_plan.json',first)
        if args.stage in ('all','1'):
            execute(first,folder,'coarse',args);atomic(folder/'stage1_done.json',{'finished':True})
        if args.stage in ('all','2'):
            if not (folder/'stage1_done.json').exists():raise RuntimeError('Complete stage 1 first.')
            if (folder/'stage2_plan.json').exists():second=read(folder/'stage2_plan.json')['configs']
            else:
                second,seeds=stage2(spec,records(folder));second=freeze(folder/'stage2_plan.json',second,{'seed_model_ids':seeds})
            execute(second,folder,'coarse',args);atomic(folder/'stage2_done.json',{'finished':True})
        if args.stage in ('all','fine'):
            if not (folder/'stage2_done.json').exists():raise RuntimeError('Complete stage 2 first.')
            final=freeze(folder/'fine_plan.json',fine_selection(records(folder)))
            if not final:raise RuntimeError('No supported models; do not treat sublimation cases as physical exclusions.')
            execute(final,folder/'fine','fine',args,plots=True)
            from v48_paper_products import selection_report
            selection_report(folder,spec)
    print('DONE. See fine/grid_results.csv, search_coverage.csv and fine/models/<id>/paper_products.json. No global-minimum guarantee; read boundary and convergence diagnostics.',flush=True)
if __name__=='__main__':main()
