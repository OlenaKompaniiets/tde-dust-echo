"""Regression gates for V4.6. Run before any scientific grid."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts import run_v4_model_grid as g
from src.v4_wise_emission import grain_size_number_weights
from src.v45_direct_absorption import solve_direct
from src.v46_ir_reprocessing import solve_direct_plus_ir

op,th,a,lookup=g._cached_dust_static('graphite',.1,1.)
wa=grain_size_number_weights(a,q=3.5); qs=th.source_mean_qabs(a)
lam=th.wavelength_m; C=(np.pi*(a*1e-6)**2)[:,None]*op.q_abs(th.wavelength_micron[None,:],a[:,None])
r=np.array([.2,.35,1.,1.25]); rw=np.ones(4)/4
numbers=np.outer(np.array([2e35,2e35,4e35,4e35]),wa)
t=np.arange(0,401,20.); L=3e43*np.exp(-((t-120)/100)**2); sub=1800.; f=.75
old=solve_direct(t,L,lam,th.source_sed,C,qs,lookup,r,numbers,f,sub,attenuation=True)
z0=solve_direct_plus_ir(t,L,lam,th.source_sed,C,qs,lookup,r,numbers,f,sub,attenuation=True,ir_reprocess_fraction=0.)
# With diffuse propagation disabled, V4.6 must exactly recover V4.5 thermal/direct solution.
assert np.allclose(z0['T'],old['T'],rtol=2e-12,atol=1e-10)
assert np.array_equal(z0['alive'],old['alive'])
assert np.allclose(z0['absorbed_direct_erg_s'],old['absorbed_erg_s'],rtol=2e-12,atol=1e20)
assert z0['energy_budget_relative_max']<2e-9
z5=solve_direct_plus_ir(t,L,lam,th.source_sed,C,qs,lookup,r,numbers,f,sub,attenuation=True,ir_reprocess_fraction=.5)
assert z5['energy_budget_relative_max']<2e-9
assert np.all(z5['absorbed_ir_erg_s']>=0)
assert np.sum(z5['absorbed_ir_erg_s'])>0
print('PASS V4.6 -> V4.5 zero-diffuse regression')
print('PASS V4.6 energy closure',z5['energy_budget_relative_max'])
print('secondary/direct absorbed ratio',z5['absorbed_ir_erg_s'].sum()/z5['absorbed_direct_erg_s'].sum())
