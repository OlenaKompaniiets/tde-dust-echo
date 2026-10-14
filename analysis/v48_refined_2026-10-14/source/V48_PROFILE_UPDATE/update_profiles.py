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
ap=argparse.ArgumentParser();ap.add_argument('--cache',default=str(P.parent/'V48_RESPONSE_VALIDATION'));ap.add_argument('--validation',default=str(P/'input_validation.zip'));args=ap.parse_args()
old=Path(args.cache);a=np.load(old/'response_cache.npz');X,F,t,chi,obs=[a[k] for k in ['X','F','t','chi2','obs']];R=json.loads((old/'records.json').read_text());prior=json.loads((old/'emulator_profiles.json').read_text());z=zipfile.ZipFile(args.validation);plan=json.loads(z.read('validation_plan.json'));new_report=json.loads(z.read('report.json'));newerr=[]
known={r['id'] for r in R}
for pt in plan['points']:
 s=json.loads(z.read('models/'+pt['id']+'/summary.json'));c=np.genfromtxt(io.BytesIO(z.read('models/'+pt['id']+'/response.csv')),names=True,delimiter=',');assert np.array_equal(t,c['time_obs_days']);f=np.array([c['W1_Jy'],c['W2_Jy']]);value,t0=profile(t,f,obs);newerr.append(value-s['fit']['chi2_total'])
 if pt['id'] not in known:
  R.append(dict(id=pt['id'],config=s['config'],chi2=value,t0=t0));X=np.vstack([X,[s['config'][k] for k in K]]);F=np.concatenate([F,f[None]]);chi=np.r_[chi,value]
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
starts=[np.zeros(7)]+[vector(R[i]['config'],R[i]['t0']) for i in np.argsort(chi)[1:3]]
base=multi(starts);b=np.array(base['v']);print('BASE',base,flush=True)
result={'n_models':len(R),'RT_reference':ref,'baseline':base,'derivative_relative_error':err,'max_new_chi2_reproduction_error':float(max(abs(np.array(newerr)))),'sides':{},'status':'response_emulator_only_no_RT_confidence_certification'}

def audit(point):
 v=np.array(point['v']);physical=x0+v[:6]*SCALE;f=(nearest(v[:6][None])[0]@B+mean).reshape(2,-1);alternate,t0=profile(t,f,obs)
 lp=linprog(np.zeros(len(U)),A_eq=np.vstack([U.T,np.ones(len(U))]),b_eq=np.r_[v[:6],1],bounds=(0,None),method='highs')
 return dict(config=dict(ref['config'],**dict(zip(K,physical.tolist()))),t0=tref+20*v[6],alternate_600_chi2=alternate,inside_training_convex_hull=bool(lp.success),minimum_flux_Jy=float(curves(v).min()))
result['baseline'].update(audit(base));target=base['chi2']+1
for j,k in enumerate(K):
 for sign in [-1,1]:
  name=k+str(sign);priorpoint=prior['sides'][name];oldv=vector(priorpoint['config'],priorpoint['t0']);initial=oldv[j];inside=dict(base,value=b[j]);outside=None;nodes=[];warm=oldv
  for n in range(5):
   value=float(np.clip(b[j]+(initial-b[j])*1.4**n,lo[j]+1e-8,hi[j]-1e-8))
   pt=multi([warm,b,oldv],{j:value});pt['value']=value;nodes.append(pt);warm=np.array(pt['v'])
   if pt['chi2']>=target:outside=pt;break
   inside=pt
  if outside:
   for n in range(7):
    frac=float(np.clip((target-inside['chi2'])/(outside['chi2']-inside['chi2']),.2,.8));value=inside['value']+frac*(outside['value']-inside['value']);pt=multi([np.array(inside['v']),np.array(outside['v']),b],{j:value});pt['value']=value;nodes.append(pt)
    if abs(pt['chi2']-target)<.003:break
    if pt['chi2']<target:inside=pt
    else:outside=pt
   endpoint=min(nodes,key=lambda q:abs(q['chi2']-target));endpoint.update(audit(endpoint));item=dict(endpoint=endpoint['config'][k],endpoint_details=endpoint,nodes=nodes,status='provisional_crossing')
  else:item=dict(nodes=nodes,status='no_crossing')
  result['sides'][name]=item;(P/'updated_profiles.json').write_text(json.dumps(result,indent=2));print('SIDE',name,item.get('endpoint'),item.get('endpoint_details',{}).get('chi2'),'spread',item.get('endpoint_details',{}).get('multistart_spread'),'alternate',item.get('endpoint_details',{}).get('alternate_600_chi2'),flush=True)
np.savez_compressed(P/'merged_response_cache.npz',X=X,F=F,t=t,chi2=chi,obs=obs);(P/'merged_records.json').write_text(json.dumps(R))
print('DONE',flush=True)
# Independent implementation check and additional starts for the two multimodal sides.
scipy_global=RBFInterpolator(U,coeff,kernel='thin_plate_spline',degree=1,smoothing=1e-6)
checks=[]
for vv in [np.array(base['v'])]+[np.array(d['endpoint_details']['v']) for d in result['sides'].values()]:
 ours=curves(vv);other=(scipy_global(vv[:6][None])[0]@B+mean).reshape(2,-1)
 checks.append(float(np.max(abs(ours-other))))
assert max(checks)<1e-8,max(checks)
robust={};rng=np.random.default_rng(481)
for name in ['inc-1','tau_visc1']:
 endpoint=result['sides'][name]['endpoint_details'];start=np.array(endpoint['v']);j=K.index(name[:-2] if name.endswith('-1') else name[:-1]);runs=[fit(start+np.r_[rng.normal(0,.15,6),0],{j:start[j]}) for _ in range(8)]
 robust[name]={'additional_starts':runs,'previous_chi2':endpoint['chi2'],'best_chi2':min(r['chi2'] for r in runs)}
 print('EXTRA STARTS',name,endpoint['chi2'],robust[name]['best_chi2'],flush=True)
result['implementation_check_max_flux_difference_Jy']=max(checks);result['additional_multistart_checks']=robust
(P/'updated_profiles.json').write_text(json.dumps(result,indent=2))
