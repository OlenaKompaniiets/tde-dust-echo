"""Response-only profile audit. No RT launches; uncertainty limits are provisional."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,io,json,zipfile,time
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares,linprog,minimize
from scipy.interpolate import RBFInterpolator
from validate import K,SCALE,profile

P=Path(__file__).resolve().parent
import hashlib
archives=[P.parent/'CASE_B_412_REVIEW/input_412.zip',P.parent/'SEND_CASE_B_SUPPORT.zip',P.parent/'SEND_CASE_B_TWO_CHECKS.zip',P.parent/'SEND_CASE_B_PROFILE_CHECKS.zip']
R=[];Fs=[];newerr=[];known={};hashchecks=[]
with zipfile.ZipFile(archives[0]) as z:
 obs=np.genfromtxt(io.BytesIO(z.read('project/data/UGC11487_WISE_TDE_model_input_22epochs.csv')),delimiter=',',names=True)
for archive in archives:
 with zipfile.ZipFile(archive) as z:
  names=sorted(n for n in z.namelist() if n.startswith('models/') and n.endswith('/summary.json'))
  for n in names:
   prefix=n.rsplit('/',1)[0]+'/';id=prefix.split('/')[1];s=json.loads(z.read(n));c=json.loads(z.read(prefix+'config.json'));pr=json.loads(z.read(prefix+'provenance.json'));raw=z.read(prefix+'response.csv')
   if id in known:
    assert known[id]==hashlib.sha256(raw).hexdigest();continue
   assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:20]==id
   assert s['config']==c and s['static_opacity_gate_passed'] and s['provenance_sha256']==pr['sha256']
   if not R:reference_pr=pr;reference_c=c
   fixed=lambda cc:{k:v for k,v in cc.items() if k not in K}
   assert fixed(c)==fixed(reference_c)
   for k in ['hashes','python','numpy','numerics']:assert pr[k]==reference_pr[k]
   assert s['numerics']==pr['numerics']
   a=np.genfromtxt(io.BytesIO(raw),names=True,delimiter=',');tt=a['time_obs_days'];f=np.array([a['W1_Jy'],a['W2_Jy']])
   if not R:t=tt
   assert np.array_equal(t,tt) and np.isfinite(f).all()
   val,epoch=profile(t,f,obs);newerr.append(val-s['fit']['chi2_total']);R.append(dict(id=id,config=c,chi2=val,t0=epoch));Fs.append(f);known[id]=hashlib.sha256(raw).hexdigest()
F=np.array(Fs);X=np.array([[r['config'][k] for k in K] for r in R]);chi=np.array([r['chi2'] for r in R]);assert len(R)==438
np.savez_compressed(P/'response_cache.npz',X=X,F=F,t=t,chi2=chi,obs=obs)
(P/'records.json').write_text(json.dumps(R))
print('AUDIT',len(R),'max reproduction error',max(abs(np.array(newerr))),flush=True)
ref=R[int(chi.argmin())];x0=X[chi.argmin()];tref=ref['t0'];U=(X-x0)/SCALE;Y=F.reshape(len(F),-1);mean=Y.mean(0);_,sv,vt=np.linalg.svd(Y-mean,full_matrices=False);B=vt[:60];coeff=(Y-mean)@B.T
# Globally smooth TPS in normalized physical coordinates; exact analytic derivatives.
r=cdist(U,U);kernel=np.where(r>0,r*r*np.log(np.maximum(r,1e-300)),0);poly=np.c_[np.ones(len(U)),U];M=np.block([[kernel+1e-6*np.eye(len(U)),poly],[poly.T,np.zeros((7,7))]]);weights=lu_solve(lu_factor(M),np.vstack([coeff,np.zeros((7,60))]));w,aff=weights[:-7],weights[-7:]
nearest=RBFInterpolator(U,coeff,neighbors=600,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
y=np.array([obs['F_'+b+'_Jy'] for b in ['W1','W2']]);sigma=np.array([obs['sigma_'+b+'_Jy'] for b in ['W1','W2']]);host=np.array([obs['baseline_'+b+'_mJy'][0]/1000 for b in ['W1','W2']])[:,None]
lo=np.r_[U.min(0),(56500-tref)/20];hi=np.r_[U.max(0),(57500-tref)/20]

def curves(v,derivative=False):
 d=v[:6]-U;r=np.linalg.norm(d,axis=1);log=np.log(np.maximum(r,1e-300));k=r*r*log;k[r==0]=0
 c=k@w+np.r_[1.,v[:6]]@aff;f=(c@B+mean).reshape(2,-1)
 if not derivative:return f
 dk=d*(2*log+1)[:,None];dk[r==0]=0;df=((dk.T@w+aff[1:])@B).reshape(6,2,-1)
 return f,df

def residual_jac(v):
 f,df=curves(v,True);ts=obs['mjd']-(tref+20*v[6]);i=np.clip(np.searchsorted(t,ts,side='right')-1,0,len(t)-2);h=t[i+1]-t[i];a=(ts-t[i])/h;valid=(ts>=t[0])&(ts<=t[-1]);model=(1-a)*f[:,i]+a*f[:,i+1];model[:,~valid]=0
 dm=(1-a)*df[:,:,i]+a*df[:,:,i+1];dm[:,:,~valid]=0
 dt=-20*(f[:,i+1]-f[:,i])/h;dt[:,~valid]=0
 res=((y-host-model)/sigma).ravel();jac=-np.concatenate([dm,dt[None]],axis=0).reshape(7,-1).T/sigma.ravel()[:,None]
 return res,jac

# Verify derivatives independently at a non-grid point.
vtest=np.r_[np.full(6,.0123),.04321];r0,j=residual_jac(vtest);jnum=np.column_stack([(residual_jac(vtest+np.eye(7)[i]*1e-5)[0]-residual_jac(vtest-np.eye(7)[i]*1e-5)[0])/2e-5 for i in range(7)]);err=float(np.max(abs(j-jnum))/max(1,float(np.max(abs(jnum)))));assert err<1e-5,err
print('MERGED',len(R),'DERIVATIVE ERROR',err,flush=True)

def vector(config,t0):return np.r_[(np.array([config[k] for k in K])-x0)/SCALE,(t0-tref)/20]
# Profile t0 exactly over every piecewise-linear response segment.
# A weighted quadratic in t0 on each segment has an analytic constrained minimum.
breaks=np.unique(np.r_[56500.,57500.,(obs['mjd'][:,None]-t[None,:]).ravel()]);breaks=breaks[(breaks>=56500)&(breaks<=57500)];mid=(breaks[:-1]+breaks[1:])/2
query=obs['mjd'][None,:]-mid[:,None];ii=np.clip(np.searchsorted(t,query,side='right')-1,0,len(t)-2);hh=t[ii+1]-t[ii];aa=(query-t[ii])/hh;ok=(query>=t[0])&(query<=t[-1])
def exact_epoch(f):
 m=(1-aa)[None]*f[:,ii]+aa[None]*f[:,ii+1];slope=-(f[:,ii+1]-f[:,ii])/hh[None];m[:,~ok]=0;slope[:,~ok]=0
 rr=(y[:,None,:]-host[:,None,:]-m)/sigma[:,None,:];bb=slope/sigma[:,None,:];den=(bb*bb).sum(axis=(0,2));offset=np.divide((rr*bb).sum(axis=(0,2)),den,out=np.zeros_like(den),where=den>0);epoch=np.clip(mid+offset,breaks[:-1],breaks[1:]);loss=((rr-bb*(epoch-mid)[None,:,None])**2).sum(axis=(0,2));j=int(np.argmin(loss));return float(epoch[j])
def fit(start,fixed=None):
 fixed=fixed or {};free=[j for j in range(6) if j not in fixed];v=np.clip(start,lo+1e-9,hi-1e-9).copy()
 for j,q in fixed.items():v[j]=q
 def both(p):
  v[free]=p;f=curves(v);v[6]=(exact_epoch(f)-tref)/20;rr,jj=residual_jac(v)
  return float(rr@rr),2*jj[:,free].T@rr
 q=minimize(both,v[free],jac=True,method='L-BFGS-B',bounds=list(zip(lo[free]+1e-9,hi[free]-1e-9)),options={'ftol':1e-12,'gtol':1e-6,'maxiter':400,'maxls':40})
 v[free]=q.x;value,gradient=both(q.x)
 return dict(v=v.tolist(),chi2=value,success=bool(q.success),optimality=float(np.max(abs(gradient))),nfev=int(q.nfev),message=str(q.message))
def multi(starts,fixed=None):
 results=[fit(s,fixed) for s in starts];best=min(results,key=lambda q:q['chi2']);best=dict(best,multistart_chi2=[q['chi2'] for q in results],multistart_spread=max(q['chi2'] for q in results)-min(q['chi2'] for q in results));return best

# Same analytic TPS derivatives and exact t0 profiling as Case A.
starts=[vector(R[i]['config'],R[i]['t0']) for i in np.argsort(chi)[:5]]
if (P/'profiles.json').exists():
 previous=json.loads((P/'profiles.json').read_text());nodes=[previous['baseline']]+[q for side in previous['sides'].values() for q in side['nodes']]
 starts += [vector(q['config'],q['t0']) for q in sorted(nodes,key=lambda q:q['chi2']) if 'config' in q][:5]
starts += [vector(q['config'],q['t0']) for q in json.loads((P/'additional_starts.json').read_text())]
base=multi(starts);b=np.array(base['v']);target=base['chi2']+1
result=dict(n_models=len(R),RT_reference=ref,baseline=base,sides={},status='provisional_response_profiles_not_RT_validated',derivative_relative_error=err,max_chi2_reproduction_error=float(max(abs(np.array(newerr)))),parameter_bounds=dict(zip(K,zip(X.min(0).tolist(),X.max(0).tolist()))))
def audit(q):
 v=np.array(q['v']);phys=x0+v[:6]*SCALE
 lp=linprog(np.zeros(len(U)),A_eq=np.vstack([U.T,np.ones(len(U))]),b_eq=np.r_[v[:6],1],bounds=(0,None),method='highs')
 return dict(config=dict(ref['config'],**dict(zip(K,phys.tolist()))),t0=tref+20*v[6],inside_training_convex_hull=bool(lp.success),minimum_flux_Jy=float(curves(v).min()))
def save():
 (P/'profiles.json').write_text(json.dumps(result,indent=2))
base.update(audit(base));print('BASE',base,flush=True);save()
for j,k in enumerate(K):
 for sign in (-1,1):
  bound=lo[j]+1e-8 if sign<0 else hi[j]-1e-8
  inner=dict(base,value=b[j]);outer=None;nodes=[]
  for distance in [.25,.5,1.,1.7,2.7,4.5,8.,20.]:
   value=float(max(bound,b[j]-distance) if sign<0 else min(bound,b[j]+distance))
   pt=multi([inner['v'],b,starts[0]],{j:value});pt['value']=value;nodes.append(pt)
   if pt['chi2']>=target:outer=pt;break
   inner=pt
   if abs(value-bound)<1e-7:break
  if outer is not None:
   for _ in range(10):
    frac=float(np.clip((target-inner['chi2'])/(outer['chi2']-inner['chi2']),.2,.8));value=inner['value']+frac*(outer['value']-inner['value'])
    pt=multi([inner['v'],outer['v'],b],{j:value});pt['value']=value;nodes.append(pt)
    if abs(pt['chi2']-target)<.003:break
    if pt['chi2']<target:inner=pt
    else:outer=pt
   pt=min(nodes,key=lambda q:abs(q['chi2']-target));pt.update(audit(pt));side=dict(status='unvalidated_crossing',endpoint=pt['config'][k],endpoint_details=pt,nodes=nodes)
  else:side=dict(status='no_crossing_within_archive_bounds',boundary_value=float(x0[j]+bound*SCALE[j]),boundary_delta_chi2=nodes[-1]['chi2']-base['chi2'],nodes=nodes)
  result['sides'][k+str(sign)]=side;save();print('SIDE',k,sign,side.get('endpoint'),side['status'],flush=True)
# Verify the custom TPS against SciPy at all returned configurations.
rb=RBFInterpolator(U,coeff,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
qs=[base]+[v['endpoint_details'] for v in result['sides'].values() if 'endpoint_details' in v]
diffs=[float(np.max(abs(curves(np.array(q['v']))-(rb(np.array(q['v'][:6])[None])[0]@B+mean).reshape(2,-1)))) for q in qs]
result['scipy_implementation_max_flux_difference_Jy']=max(diffs);assert max(diffs)<1e-8
result['lowest_profile_chi2']=min(q['chi2'] for v in result['sides'].values() for q in v['nodes']);save();print('DONE',flush=True)
