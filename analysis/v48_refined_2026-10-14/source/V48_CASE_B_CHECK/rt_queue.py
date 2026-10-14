"""Bounded RT validation; unchanged V4.8 physics. Default: 4 workers / 45 min."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse, hashlib, json, sys, time, subprocess, zipfile
from pathlib import Path
from cache_utils import read,write
HERE=Path(__file__).resolve().parent

def acquire(p):
 p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
 try:
  if os.name=='nt':
   import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except OSError:f.close();raise RuntimeError('Another coordinator is active: '+str(p))
 return f

def kernel(c):return json.dumps({k:c[k] for k in ('stars','compact','extended','tau_C','tau_E','theta','composition','amin','amax','q')},sort_keys=True)

def stop_children(active):
 # Only children started by THIS invocation. No taskkill /IM python.exe.
 for item in active:
  if item['p'].poll() is None:item['p'].terminate()
 deadline=time.monotonic()+3
 for item in active:
  try:item['p'].wait(timeout=max(.01,deadline-time.monotonic()))
  except subprocess.TimeoutExpired:item['p'].kill();item['p'].wait(timeout=3)
  item['log'].close()

def schedule(points,out,command,collect,workers,seconds,max_models,stop_path=None,budget=None):
 start=time.monotonic();deadline=start+seconds;active=[];finished={};checkpoint=out/'checkpoint.json'
 if checkpoint.exists():finished=read(checkpoint).get('finished',{})
 todo=[p for p in points if p['id'] not in finished][:max_models];launched=0;last=0;reason='completed'
 def save():write(checkpoint,{'finished':finished,'launched_this_session':launched,'elapsed_seconds':time.monotonic()-start,'stop_reason':reason})
 try:
  while todo or active:
   now=time.monotonic()
   if now>=deadline or ((out/'STOP').exists() or (stop_path is not None and stop_path.exists())):reason='time_limit' if now>=deadline else 'stop_requested';break
   for item in active[:]:
    rc=item['p'].poll()
    if rc is not None:
     item['log'].close();active.remove(item)
     if rc:reason='solver_error';raise RuntimeError('Solver failed; see '+str(item['path']))
     result=collect(item['point']);finished[item['point']['id']]=result;save()
     print('DONE',len(finished),'/',len(points),item['point']['name'],'chi2',result.get('chi2'),flush=True)
   for point in todo[:]:
    if len(active)>=workers:break
    key=kernel(point['config'])
    if any(v['key']==key for v in active):continue
    if budget is not None and not budget.reserve(point['id']):
     reason='total_budget_exhausted';todo=[];break
    path=out/(point['name']+'.log');log=path.open('a',encoding='utf-8')
    try:p=subprocess.Popen(command(point),stdout=log,stderr=subprocess.STDOUT)
    except Exception:log.close();raise
    active.append({'p':p,'log':log,'path':path,'point':point,'key':key});todo.remove(point);launched+=1;save()
    print('START',point['name'],'PID',p.pid,flush=True)
   if now-last>=10:
    print('HEARTBEAT elapsed_min',round((now-start)/60,1),'active',len(active),'saved',len(finished),'remaining_budget_min',round(max(0,deadline-now)/60,1),flush=True);last=now
   time.sleep(.25)
 except KeyboardInterrupt:reason='keyboard_interrupt'
 finally:
  # Collect jobs that finished just before the deadline before terminating others.
  for item in active[:]:
   if item['p'].poll()==0:
    item['log'].close();active.remove(item)
    try:finished[item['point']['id']]=collect(item['point'])
    except Exception:pass
  stop_children(active);save()
 if reason=='completed' and len(finished)<len(points):
  reason='model_call_limit';save()
 return finished,reason
