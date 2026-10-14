import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,zipfile
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import profile,SCALE
p=Path(__file__).resolve().parent;a=np.load(p/'response_cache.npz');X,F,t,chi,obs=[a[k] for k in ['X','F','t','chi2','obs']];U=(X-X[chi.argmin()])/SCALE;Y=F.reshape(len(F),-1)
rng=np.random.default_rng(481);o=rng.permutation(len(F));test=o[:len(F)//5];train=o[len(test):];mean=Y[train].mean(0);_,_,v=np.linalg.svd(Y[train]-mean,full_matrices=False);b=v[:60];rb=RBFInterpolator(U[train],(Y[train]-mean)@b.T,kernel='thin_plate_spline',degree=1,smoothing=1e-6);pred=(rb(U[test])@b+mean).reshape(len(test),2,-1);err=np.array([profile(t,f,obs)[0] for f in pred])-chi[test]
result={'random_holdout':[]}
for d in [1,4,10,100000]:
 sel=chi[test]<min(chi)+d
 r=dict(delta=d,n=int(sel.sum()),rmse=float(np.sqrt(np.mean(err[sel]**2))),max_abs=float(max(abs(err[sel]))));result['random_holdout'].append(r);print(r,flush=True)
result['new_points_leave_one_out']=[]
for i in range(len(F)-12,len(F)):
 tr=np.arange(len(F))!=i;mean=Y[tr].mean(0);_,_,v=np.linalg.svd(Y[tr]-mean,full_matrices=False);b=v[:60];rb=RBFInterpolator(U[tr],(Y[tr]-mean)@b.T,kernel='thin_plate_spline',degree=1,smoothing=1e-6);f=(rb(U[i:i+1])[0]@b+mean).reshape(2,-1);cp=profile(t,f,obs)[0];result['new_points_leave_one_out'].append(dict(index=i,chi2=float(chi[i]),predicted=cp,error=float(cp-chi[i])));print('LOO',i,cp-chi[i],flush=True)
(p/'local_validation.json').write_text(json.dumps(result,indent=2))
