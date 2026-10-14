"""Audit archived Case B responses and compute bounded exploratory profiles.
No radiative-transfer solver is called. Results are NOT final confidence limits.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import io,json,zipfile,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares,linprog
from validation_utils import profile,K,SCALE
P=Path(__file__).resolve().parent

def save(name,obj): (P/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def main():
 with zipfile.ZipFile(sys.argv[1]) as z:
  manifest=json.loads(z.read('manifest.json'))
  obs=np.genfromtxt(io.BytesIO(z.read('project/data/UGC11487_WISE_TDE_model_input_22epochs.csv')),delimiter=',',names=True)
  ref=manifest['best_saved']['id'];rp=json.loads(z.read('models/'+ref+'/provenance.json'))
  rc=json.loads(z.read('models/'+ref+'/config.json')); records=[];F=[];errors=[];excluded=[];t=None
  for row in manifest['models']:
   prefix='models/'+row['id']+'/'
   c=json.loads(z.read(prefix+'config.json'));s=json.loads(z.read(prefix+'summary.json'));p=json.loads(z.read(prefix+'provenance.json'))
   fixed=lambda c:{k:v for k,v in c.items() if k not in K}
   if fixed(c)!=fixed(rc) or p['numerics']!=rp['numerics'] or p['hashes']!=rp['hashes'] or p.get('python')!=rp.get('python') or p.get('numpy')!=rp.get('numpy'):
    excluded.append(row['id']);continue
   raw=z.read(prefix+'response.csv');assert hashlib.sha256(raw).hexdigest()==row['response_sha256']
   a=np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True);tt=a['time_obs_days']
   if t is None:t=tt
   assert np.array_equal(t,tt)
   f=np.array([a['W1_Jy'],a['W2_Jy']]);assert np.isfinite(f).all()
   chi,t0=profile(t,f,obs);errors.append(chi-s['fit']['chi2_total']);records.append(dict(id=row['id'],config=c,chi2=chi,t0=t0));F.append(f)
  save('reference_provenance.json',rp)
 F=np.array(F);X=np.array([[r['config'][k] for k in K] for r in records]);chi=np.array([r['chi2'] for r in records]);ib=chi.argmin();x0=X[ib];U=(X-x0)/SCALE
 report={'n_responses':len(F),'excluded_incompatible':excluded,'max_chi2_reproduction_error':float(np.max(np.abs(errors))),'reference':records[ib], 'unique_per_parameter':dict(zip(K,[len(np.unique(x)) for x in X.T])),'parameter_bounds':dict(zip(K,zip(X.min(0).tolist(),X.max(0).tolist())))}
 save('audit.json',report);save('records.json',records);np.savez_compressed(P/'cache.npz',X=X,F=F,t=t,chi2=chi,obs=obs)
 print('AUDIT',report,flush=True)
 Y=F.reshape(len(F),-1)
 rng=np.random.default_rng(481);order=rng.permutation(len(F));test=order[:max(20,len(F)//5)];train=order[len(test):]
 # A fixed holdout split; no hyperparameter selection on this split.
 mean=Y[train].mean(0);_,_,vt=np.linalg.svd(Y[train]-mean,full_matrices=False);basis=vt[:60]
 rb=RBFInterpolator(U[train],(Y[train]-mean)@basis.T,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
 pred=(rb(U[test])@basis+mean).reshape(len(test),2,-1)
 err=np.array([profile(t,f,obs)[0] for f in pred])-chi[test]
 cv={'test_count':len(test),'rmse_chi2':float(np.sqrt(np.mean(err**2))),'max_abs_chi2':float(np.max(abs(err))),'q95_abs_chi2':float(np.quantile(abs(err),.95)),'note':'Random held-out configurations; clustered sampling can make this optimistic.'}
 save('holdout.json',cv);print('HOLDOUT',cv,flush=True)
 mean=Y.mean(0);_,_,vt=np.linalg.svd(Y-mean,full_matrices=False);basis=vt[:60]
 rb=RBFInterpolator(U,(Y-mean)@basis.T,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
 ot=obs['mjd'];y=np.array([obs['F_'+b+'_Jy'] for b in ('W1','W2')]);sig=np.array([obs['sigma_'+b+'_Jy'] for b in ('W1','W2')]);host=np.array([obs['baseline_'+b+'_mJy'][0]/1000 for b in ('W1','W2')])[:,None]
 rt=records[ib]['t0'];lo=np.r_[U.min(0),(56500-rt)/20];hi=np.r_[U.max(0),(57500-rt)/20]
 def residual(v):
  f=(rb(v[:6][None,:])[0]@basis+mean).reshape(2,-1)
  m=np.array([np.interp(ot-(rt+20*v[6]),t,w,left=0,right=0) for w in f]);return ((y-host-m)/sig).ravel()
 def solve(start,fixed=None):
  fixed=fixed or {};free=[j for j in range(7) if j not in fixed];v=np.clip(start,lo+1e-10,hi-1e-10).copy()
  for j,value in fixed.items():v[j]=value
  def fun(w):
   vv=v.copy();vv[free]=w;return residual(vv)
  r=least_squares(fun,v[free],bounds=(lo[free],hi[free]),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-7)
  v[free]=r.x
  return dict(v=v.tolist(),chi2=float(r.fun@r.fun),success=bool(r.success),optimality=float(r.optimality),nfev=int(r.nfev))
 starts=[np.r_[U[i],(records[i]['t0']-rt)/20] for i in np.argsort(chi)[:5]]
 fits=[solve(v) for v in starts];base=min(fits,key=lambda r:r['chi2']);target=base['chi2']+1
 print('BASE',base,flush=True)
 results={'scope':'Exploratory interpolated profiles bounded by archived coordinate coverage; not final errors','baseline':base,'multistart':fits,'sides':{},'reference':records[ib],'scale':SCALE.tolist(),'x0':x0.tolist(),'time_reference':rt,'lower':lo.tolist(),'upper':hi.tolist()}
 for j,k in enumerate(K+['t0_mjd']):
  for sign in (-1,1):
   center=base['v'][j];bound=lo[j] if sign<0 else hi[j];inner=base;iv=center;outer=None;nodes=[]
   for frac in [.2,.5,1.]:
    value=center+(bound-center)*frac
    fits=[solve(v,{j:value}) for v in [inner['v'],base['v']]];point=min(fits,key=lambda r:r['chi2']);point['fixed_value']=value;nodes.append(point)
    if point['chi2']>=target:outer=point;ov=value;break
    inner=point;iv=value
   if outer:
    for _ in range(9):
     value=(iv+ov)/2;fits=[solve(v,{j:value}) for v in [inner['v'],outer['v']]];point=min(fits,key=lambda r:r['chi2']);point['fixed_value']=value;nodes.append(point)
     if abs(point['chi2']-target)<.005:break
     if point['chi2']<target:inner=point;iv=value
     else:outer=point;ov=value
    end=min(nodes,key=lambda r:abs(r['chi2']-target));v=np.array(end['v']);hull=linprog(np.zeros(len(U)),A_eq=np.vstack([U.T,np.ones(len(U))]),b_eq=np.r_[v[:6],1],bounds=(0,None),method='highs').success
    result={'status':'unvalidated_crossing','value':float(x0[j]+v[j]*SCALE[j]) if j<6 else float(rt+20*v[6]),'inside_hull':bool(hull),'endpoint':end}
   else:result={'status':'no_crossing_within_archive_bounds','boundary_value':float(x0[j]+bound*SCALE[j]) if j<6 else float(rt+20*bound),'boundary_delta_chi2':nodes[-1]['chi2']-base['chi2']}
   result['nodes']=nodes;results['sides'][k+str(sign)]=result;save('profiles.json',results);print(k,sign,{k:v for k,v in result.items() if k not in ['nodes','endpoint']},flush=True)
 if min(q['chi2'] for s in results['sides'].values() for q in s['nodes'])<base['chi2']-.01:results['warning']='Lower minimum found while profiling; baseline must be updated.'
 save('profiles.json',results)

if __name__=='__main__':main()
