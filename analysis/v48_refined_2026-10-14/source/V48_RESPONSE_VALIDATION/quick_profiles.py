"""Exploratory response-emulator profiles; NOT certified confidence intervals."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,json,time
from pathlib import Path
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares,linprog
from validate import K,SCALE
P=Path(__file__).parent;a=np.load(P/'response_cache.npz');X,F,t,chi,obs=[a[k] for k in ['X','F','t','chi2','obs']];records=json.loads((P/'records.json').read_text());x0=X[chi.argmin()];U=(X-x0)/SCALE;Y=F.reshape(len(F),-1);mean=Y.mean(0);_,sv,vt=np.linalg.svd(Y-mean,full_matrices=False);basis=vt[:60];coef=(Y-mean)@basis.T;rbf=RBFInterpolator(U,coef,neighbors=300,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
tt=obs['mjd'];y=np.array([obs['F_'+b+'_Jy'] for b in ('W1','W2')]);sig=np.array([obs['sigma_'+b+'_Jy'] for b in ('W1','W2')]);host=np.array([obs['baseline_'+b+'_mJy'][0]/1000 for b in ('W1','W2')])[:,None];ref_t0=records[int(chi.argmin())]['t0']
lo=np.r_[U.min(0), (56500-ref_t0)/20];hi=np.r_[U.max(0),(57500-ref_t0)/20];evals=0

def residual(v):
 global evals
 evals+=1
 f=(rbf(v[:6][None,:])[0]@basis+mean).reshape(2,-1)
 model=np.array([np.interp(tt-(ref_t0+20*v[6]),t,fk,left=0,right=0) for fk in f]);return ((y-host-model)/sig).ravel()
def optimize(start,fixed=None):
 fixed=fixed or {};free=[i for i in range(7) if i not in fixed];v=np.clip(start,lo+1e-10,hi-1e-10).copy()
 for i,q in fixed.items():v[i]=q
 def fun(z):v[free]=z;return residual(v)
 r=least_squares(fun,v[free],bounds=(lo[free],hi[free]),diff_step=1e-4,xtol=2e-5,ftol=2e-6,gtol=1e-5,max_nfev=60)
 v[free]=r.x;return dict(v=v.tolist(),chi2=float(r.fun@r.fun),success=bool(r.success),nfev=r.nfev,optimality=float(r.optimality))
base=optimize(np.zeros(7));print('EMULATOR BASE',base,flush=True);target=base['chi2']+1;results={'scope':'exploratory emulator profiles, conditional on fixed physics; not validated RT confidence limits','baseline':base,'source_reference_chi2':float(min(chi)),'sides':{},'model_evaluations':0}
for j,k in enumerate(K):
 for sign in [-1,1]:
  center=base['v'][j];warm=np.array(base['v']);inside=dict(base);inside['fixed_value']=center;step=.7;outside=None;nodes=[]
  for n in range(6):
   value=float(np.clip(center+sign*step,lo[j]+1e-9,hi[j]-1e-9));point=optimize(warm,{j:value});point['fixed_value']=value;nodes.append(point);warm=np.array(point['v'])
   if point['chi2']>=target:outside=point;break
   inside=point
   if abs(value-(lo[j] if sign<0 else hi[j]))<1e-7:break
   step*=1.6
  if outside:
   for n in range(6):
    frac=float(np.clip((target-inside['chi2'])/(outside['chi2']-inside['chi2']),.2,.8));value=inside['fixed_value']+frac*(outside['fixed_value']-inside['fixed_value']);start=np.array((inside if frac<.5 else outside)['v']);point=optimize(start,{j:value});point['fixed_value']=value;nodes.append(point)
    if abs(point['chi2']-target)<.015:break
    if point['chi2']<target:inside=point
    else:outside=point
   endpoint=min(nodes,key=lambda q:abs(q['chi2']-target));physical=x0+np.array(endpoint['v'][:6])*SCALE
   # Convex-hull feasibility is a support diagnostic, not an uncertainty test.
   lp=linprog(np.zeros(len(U)),A_eq=np.vstack([U.T,np.ones(len(U))]),b_eq=np.r_[endpoint['v'][:6],1.],bounds=(0,None),method='highs')
   item=dict(nodes=nodes,endpoint=physical[j],config=dict(records[int(chi.argmin())]['config'],**dict(zip(K,physical.tolist()))),t0=ref_t0+20*endpoint['v'][6],chi2=endpoint['chi2'],inside_training_convex_hull=bool(lp.success),status='provisional_emulator_crossing' if endpoint['success'] else 'optimizer_limit')
  else:item=dict(nodes=nodes,status='no_crossing_within_sampled_coordinate_bounds')
  results['sides'][k+str(sign)]=item;results['model_evaluations']=evals;(P/'emulator_profiles.json').write_text(json.dumps(results,indent=2));print(k,sign,item['status'],item.get('endpoint'),item.get('chi2'),'hull',item.get('inside_training_convex_hull'),'calls',evals,flush=True)
