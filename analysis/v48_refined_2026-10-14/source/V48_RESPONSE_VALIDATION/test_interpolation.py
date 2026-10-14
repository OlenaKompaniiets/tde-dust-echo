import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from validate import profile,K,SCALE
import numpy as np,json
from scipy.interpolate import RBFInterpolator
from pathlib import Path
P=Path(__file__).parent;a=np.load(P/'response_cache.npz');X,F,t,chi,obs,train,test=[a[k] for k in ['X','F','t','chi2','obs','train','test']];U=(X-X[chi.argmin()])/SCALE;Y=F.reshape(len(F),-1);mean=Y[train].mean(0);_,sv,vt=np.linalg.svd(Y[train]-mean,full_matrices=False);basis=vt[:60];coeff=(Y[train]-mean)@basis.T;results=[]
settings=[dict(neighbors=300,kernel='thin_plate_spline',degree=1,smoothing=1e-6),dict(neighbors=600,kernel='thin_plate_spline',degree=1,smoothing=1e-6),dict(neighbors=None,kernel='thin_plate_spline',degree=1,smoothing=1e-6)]+[dict(neighbors=100,kernel='gaussian',degree=0,epsilon=e,smoothing=s) for e in [.5,1.] for s in [1e-6,1e-4]]
for kw in settings:
 try:
  rbf=RBFInterpolator(U[train],coeff,**kw);pred=(rbf(U[test])@basis+mean).reshape(len(test),2,-1);cp=np.array([profile(t,v,obs)[0] for v in pred]);err=cp-chi[test];r=dict(settings=kw,rmse=float(np.sqrt(np.mean(err**2))),q95=float(np.quantile(abs(err),.95)),maximum=float(max(abs(err))),errors=err.tolist());print({k:v for k,v in r.items() if k!='errors'},flush=True);results.append(r)
 except Exception as e:results.append(dict(settings=kw,error=str(e)));print('ERROR',kw,str(e),flush=True)
(P/'interpolation_tests.json').write_text(json.dumps(results,indent=2))
