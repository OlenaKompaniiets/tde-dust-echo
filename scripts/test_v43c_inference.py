"""Verify the shared-band constrained amplitude fit against an independent QP."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
from scripts import run_v43c_geometry_audit as a
rng=np.random.default_rng(43)
worst=0.
for k in range(100):
    A=rng.uniform(0,2,(44,2));y=rng.normal(0,1,44)+A@rng.uniform(0,1,2)
    if k==0:A[:]=0
    if k==1:A[:,1]=A[:,0]
    a.v.FMAX=a.v.FTOT=.5
    f,*_=a.v.solve_amp(y[:22],y[22:],np.ones(22),np.ones(22),0,0,A[:22,0],A[22:,0],A[:22,1],A[22:,1])
    ref=minimize(lambda x:np.sum((A@x-y)**2),[.2,.2],jac=lambda x:2*A.T@(A@x-y),
                 bounds=[(0,.5)]*2,constraints={'type':'ineq','fun':lambda x:.5-x.sum(),'jac':lambda x:-np.ones(2)},
                 method='SLSQP',options={'ftol':1e-10,'maxiter':500})
    chi=np.sum((A@f-y)**2);delta=abs(chi-ref.fun);worst=max(worst,delta)
    assert f.min()>=-1e-10 and f.sum()<=.5+1e-10
    assert delta<1e-5,(k,chi,ref.fun)
print('PASS: 100 constrained shared-band fits; largest chi2 difference',worst)
assert len(a.tasks('geometry'))==720
assert all(t['cap']<=np.cos(np.deg2rad(t['theta']))+1e-12 for t in a.tasks('geometry'))
assert a.taskid(a.tasks('control')[0])!=a.taskid(a.tasks('control')[1])
print('PASS: 720 geometry blocks, geometric caps, distinct task identifiers')
