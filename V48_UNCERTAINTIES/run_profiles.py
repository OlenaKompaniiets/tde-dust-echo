"""Conditional likelihood profiles using the archived parallel best fits.
Requires existing V48_PAPER_FINAL and V48_LOCAL_PAIR. Physics unchanged.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys, copy, hashlib, zipfile, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
P=Path(__file__).resolve().parent
ROOT=P.parent
sys.path.insert(0,str(ROOT/'V48_PAPER_FINAL'))
import run_final as rf
import run_parallel as rp
_original_evaluate=rf.evaluate

def locked_evaluate(config,*args,**kwargs):
    with rp.kernel_lock(config):
        return _original_evaluate(config,*args,**kwargs)
rf.evaluate=locked_evaluate
PARAMS=list(rf.STEPS)

def write(path,obj):
    rf.write(path,obj)

def one_side(branch,param,sign,base,args,out):
    center=base['t0_mjd'] if param=='t0_mjd' else base['config'][param]
    bounds=(56500.,57500.) if param=='t0_mjd' else rf.BOUNDS[param]
    dest=out/branch/param/('minus.json' if sign<0 else 'plus.json')
    identity=hashlib.sha256(json.dumps(dict(base=base,parameter=param,sign=sign,maxfev=args.maxfev,points=args.points,refine=args.refine),sort_keys=True).encode()).hexdigest()
    state=rf.read(dest) if dest.exists() else dict(identity=identity,branch=branch,parameter=param,sign=sign,center=center,baseline_chi2=base['chi2'],points=[],complete=False)
    if state['identity']!=identity: raise RuntimeError('Settings changed: select a new --output directory.')
    if state['complete']: return state
    def solve(value,start):
        previous=next((p for p in state['points'] if p['value']==value),None)
        if previous is not None: return previous
        print('PROFILE',branch,param,sign,'value',value,flush=True)
        r=rf.optimize(start,{param:value},args.maxfev)
        r.update(value=value,delta_chi2=r['chi2']-base['chi2'])
        state['points'].append(r);write(dest,state)
        return r
    start=base['config'];inner=(center,0.,start);outer=None
    for i in range(args.points):
        value=max(bounds[0],min(bounds[1],center+sign*rf.STEPS[param]*2**i))
        r=solve(value,start)
        if r['delta_chi2'] < -0.01:
            state.update(complete=True,status='lower_minimum_found_no_intervals');write(dest,state);return state
        if not r['optimizer_success']:
            state.update(complete=True,status='nuisance_optimizer_budget_no_intervals');write(dest,state);return state
        start=r['config']
        if r['delta_chi2']>=3.841459: outer=(value,r['delta_chi2'],start);break
        inner=(value,r['delta_chi2'],start)
        if value in bounds: break
    # Refine brackets for each one-parameter likelihood threshold.
    for target in (1.,3.841459):
        for _ in range(args.refine):
            ordered=[dict(value=center,delta_chi2=0.,config=base['config'])]+sorted(state['points'],key=lambda r:abs(r['value']-center))
            pair=next(((a,b) for a,b in zip(ordered,ordered[1:]) if a['delta_chi2']<=target<=b['delta_chi2']),None)
            if pair is None: break
            a,b=pair;value=(a['value']+b['value'])/2
            r=solve(value,a['config'])
            if r['delta_chi2'] < -0.01 or not r['optimizer_success']:
                state.update(complete=True,status='profile_requires_review_no_intervals');write(dest,state);return state
    state.update(complete=True,status='sampled_profile_review_required')
    write(dest,state);return state

def products(out,bases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':18,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16})
    report=[]
    for branch,base in bases.items():
        for param in PARAMS:
            files=list((out/branch/param).glob('*.json'))
            if not files: continue
            states=[rf.read(f) for f in files];points=[p for s in states for p in s['points']]
            center=base['t0_mjd'] if param=='t0_mjd' else base['config'][param]
            fig,ax=plt.subplots(figsize=(8,6))
            good=sorted([dict(value=center,delta_chi2=0.)]+[p for p in points if p['optimizer_success']],key=lambda p:p['value'])
            ax.plot([p['value'] for p in good],[p['delta_chi2'] for p in good],'o-')
            bad=[p for p in points if not p['optimizer_success']]
            if bad: ax.scatter([p['value'] for p in bad],[p['delta_chi2'] for p in bad],marker='x',color='red',label='Unconverged');ax.legend()
            for y in (1.,3.841459):ax.axhline(y,color='grey',ls='--')
            ax.set(xlabel=param,ylabel=r'$\Delta\chi^2$',title=branch+' conditional profile')
            fig.tight_layout();fig.savefig(out/branch/(param+'.pdf'));plt.close(fig)
            report.append(dict(branch=branch,parameter=param,statuses=[s['status'] for s in states],minimum_sampled_chi2=min([base['chi2']]+[p['chi2'] for p in points]),note='Thresholds are nominal one-parameter 68.27% and 95%; numerical/optimization validation required. No automatic certified interval.'))
    write(out/'profile_summary.json',report)
    with zipfile.ZipFile(out/'SEND_UNCERTAINTIES_FOR_REVIEW.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.suffix in ('.json','.pdf'):z.write(f,f.relative_to(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,choices=range(1,17),default=12)
    ap.add_argument('--branch',choices=['local','grid','both'],default='both')
    ap.add_argument('--parameter',choices=['all']+PARAMS,default='all')
    ap.add_argument('--maxfev',type=int,default=240)
    ap.add_argument('--points',type=int,default=5)
    ap.add_argument('--refine',type=int,default=3)
    ap.add_argument('--output',default='results/v48_uncertainties')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.maxfev<20 or args.points<1 or args.refine<0:ap.error('Invalid calculation budget')
    rf.prepare();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    # Single coordinator, including on Windows. Release automatically on exit.
    lock=(out/'coordinator.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    branches=['local','grid'] if args.branch=='both' else [args.branch]
    bases={}
    for b in branches:
        path=ROOT/'results/v48_parallel_final'/b/'best.json'
        base=rf.read(path)
        s,folder=rf.evaluate(base['config'])
        if abs(s['fit']['chi2_total']-base['chi2'])>1e-6:raise RuntimeError('Baseline reproduction mismatch')
        bases[b]=base;write(out/b/'reference.json',base)
        print('REFERENCE',b,base['chi2'],'poll_converged',base['poll_converged'],flush=True)
    # Load the fixed-t0 evaluator once before any threads use it.
    if args.parameter in ('all','t0_mjd'):
        for b,base in bases.items():
            s,folder=rf.evaluate(base['config']);value=rf.fixed_time_loss(folder,base['t0_mjd'])
            if abs(value-base['chi2'])>1e-6:raise RuntimeError('Fixed-time reconstruction mismatch')
    if args.check_only:print('CHECK OK: no profile calculations launched');return
    params=PARAMS if args.parameter=='all' else [args.parameter]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(one_side,b,p,sign,base,args,out) for b,base in bases.items() for p in params for sign in (-1,1)]
            for future in as_completed(jobs):
                s=future.result();print('SIDE FINISHED',s['branch'],s['parameter'],s['sign'],s['status'],flush=True)
    finally:products(out,bases)
    print('FINISHED',out/'SEND_UNCERTAINTIES_FOR_REVIEW.zip',flush=True)
if __name__=='__main__':main()
