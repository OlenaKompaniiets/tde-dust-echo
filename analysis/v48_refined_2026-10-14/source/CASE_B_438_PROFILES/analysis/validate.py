import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import zipfile,json,io,hashlib,argparse,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import RBFInterpolator
K=['eta','tau_visc','tau_C','tau_E','theta','inc'];SCALE=np.array([.01,25,.025,.15,2.5,2.5])
def profile(t,f,obs):
 tt=obs['mjd'];y=np.array([obs['F_'+b+'_Jy'] for b in ('W1','W2')]);sig=np.array([obs['sigma_'+b+'_Jy'] for b in ('W1','W2')]);host=np.array([obs['baseline_'+b+'_mJy'][0]/1000 for b in ('W1','W2')])[:,None]
 def loss(t0):
  residual=(y-host-np.array([np.interp(tt-t0,t,v,left=0,right=0) for v in f]))/sig
  return float(np.sum(residual**2))
 grid=np.arange(56500,57500.1,10.);val=np.array([loss(x) for x in grid]);candidates=list(zip(val,grid))
 for i in range(len(grid)):
  if val[i]<=val[max(0,i-1)] and val[i]<=val[min(len(grid)-1,i+1)]:
   opt=minimize_scalar(loss,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded',options={'xatol':.02})
   if opt.success:candidates.append((float(opt.fun),float(opt.x)))
 return min(candidates)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('--out',default=str(Path(__file__).parent));a=ap.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
 z=zipfile.ZipFile(a.archive);m=json.loads(z.read('manifest.json'));obs=np.genfromtxt(io.BytesIO(z.read('project/data/UGC11487_WISE_TDE_model_input_22epochs.csv')),delimiter=',',names=True)
 records=[];curves=[];ts=[];errs=[];grids=set()
 for i,r in enumerate(m['models']):
  raw=z.read('models/'+r['id']+'/response.csv');assert hashlib.sha256(raw).hexdigest()==r['response_sha256'];c=np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True);s=json.loads(z.read('models/'+r['id']+'/summary.json'));t=c['time_obs_days'];f=np.array([c['W1_Jy'],c['W2_Jy']]);chi,t0=profile(t,f,obs);errs.append(chi-s['fit']['chi2_total']);records.append(dict(id=r['id'],config=s['config'],chi2=chi,t0=t0));curves.append(f);ts.append(t);grids.add(hashlib.sha256(t.tobytes()).hexdigest())
  if (i+1)%500==0:print('REPRODUCED',i+1,flush=True)
 print('MAX CHI2 REPRO ERROR',max(abs(np.array(errs))),'TIME GRIDS',len(grids),flush=True)
 # Exact common time sampling only; do not silently add time interpolation.
 assert len(grids)==1
 t=ts[0];F=np.array(curves);X=np.array([[r['config'][k] for k in K] for r in records]);ids=np.array([r['id'] for r in records]);chi=np.array([r['chi2'] for r in records]);hold=set()
 # New profile batch models are held out as whole configurations.
 hold.update(json.loads((Path(__file__).parent/'holdout_ids.json').read_text()))
 test=np.array([i for i,r in enumerate(records) if r['id'] in hold]);train=np.array([i for i,r in enumerate(records) if r['id'] not in hold]);x0=X[chi.argmin()];U=(X-x0)/SCALE
 np.savez_compressed(out/'response_cache.npz',X=X,F=F,t=t,ids=ids,chi2=chi,obs=obs,test=test,train=train)
 (out/'records.json').write_text(json.dumps(records))
 report={'n_models':len(records),'n_epochs':len(obs),'max_chi2_reproduction_error':float(max(abs(np.array(errs)))),'n_time_grids':len(grids),'holdout_size':len(test),'training_size':len(train),'reference':records[int(chi.argmin())],'validation':[]}
 # Compress curves with SVD of training curves only, then interpolate coefficients.
 Y=F.reshape(len(F),-1);mean=Y[train].mean(axis=0);_,sv,vt=np.linalg.svd(Y[train]-mean,full_matrices=False);rank=min(40,len(sv));basis=vt[:rank];coeff=(Y[train]-mean)@basis.T
 trunc=np.max(abs((Y[train]-mean)@basis.T@basis+mean-Y[train]));print('SVD',rank,'max trunc Jy',trunc,flush=True)
 for neighbors in [40,80,160]:
  for smoothing in [0.,1e-6,.001]:
   try:
    emulator=RBFInterpolator(U[train],coeff,neighbors=neighbors,kernel='thin_plate_spline',smoothing=smoothing,degree=1);pred=(emulator(U[test])@basis+mean).reshape(len(test),2,-1)
    cp=np.array([profile(t,v,obs)[0] for v in pred]);errors=cp-chi[test]
    r=dict(neighbors=neighbors,smoothing=smoothing,rank=rank,max_training_truncation_Jy=float(trunc),rmse_chi2=float(np.sqrt(np.mean(errors**2))),max_abs_chi2_error=float(max(abs(errors))),q95_abs_chi2_error=float(np.quantile(abs(errors),.95)),n_negative_curves=int(np.sum(np.min(pred,axis=(1,2)) < -1e-8)))
    report['validation'].append(r);print(r,flush=True)
   except Exception as e:report['validation'].append(dict(neighbors=neighbors,smoothing=smoothing,error=str(e)))
 (out/'validation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
