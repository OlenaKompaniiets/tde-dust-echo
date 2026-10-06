"""Independent grey-slab absorption, equilibrium, destruction and thin-limit tests."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import PchipInterpolator
from src.v4_thermal_lookup import ThermalLookup
from src.v45_direct_absorption import solve_direct,quadrature_weights
sigma=5.670374419e-8;pc=3.085677581491367e16
Tgrid=np.geomspace(.01,1e5,1000);F=4*sigma*Tgrid**4
a=np.array([.1,.2]);lookup=ThermalLookup(a,Tgrid,np.stack([np.log10(F)]*2),[PchipInterpolator(np.log10(F),np.log10(Tgrid)) for _ in a])
lam=np.array([1e-7,1e-6,1e-5]);w=quadrature_weights(lam);sed=np.ones(3)/(lam[-1]-lam[0])
C=np.pi*(a*1e-6)**2;cross=np.repeat(C[:,None],3,axis=1);r=np.array([.2,1.]);f=.7
tau=np.array([.4,.7]);numbers=tau[:,None]*4*np.pi*f*(r[:,None]*pc)**2*.5/C[None,:]
t=np.arange(4.);L=np.array([0,1,2,1.])*4*np.pi*(r[0]*pc)**2*4*sigma*800**4*1e7
args=(t,L,lam,sed,cross,np.ones(2),lookup,r,numbers,f,2000.)
z=solve_direct(*args)
expected=f*L*(-np.expm1(-sum(tau)))
assert np.allclose(z['absorbed_erg_s'].sum(axis=1),expected,rtol=1e-12)
expectedT=(L[1]*1e-7/(4*np.pi*(r[0]*pc)**2)*(-np.expm1(-tau[0]))/tau[0]/(4*sigma))**.25
assert np.allclose(z['T'][1,0],expectedT,rtol=1e-12)
assert np.all(z['T'][0]==0) and np.all(z['alive'][0])
assert z['energy_budget_relative_max']<1e-12
thin=list(args);thin[8]=numbers*1e-8
zthin=solve_direct(*thin);zfree=solve_direct(*thin,attenuation=False)
err=np.max(np.abs(zthin['T']-zfree['T']))/zfree['T'].max();assert err<1e-8
hot=list(args);hot[10]=750.;zhot=solve_direct(*hot)
assert np.any(~zhot['alive'])
assert not np.any(np.diff(zhot['alive'].astype(int),axis=0)>0)
assert zhot['energy_budget_relative_max']<1e-12
assert np.all(zhot['T'][~zhot['alive']]==0)
print(json.dumps(dict(status='PASS',grey_energy_relative_error=z['energy_budget_relative_max'],
 grey_equilibrium_relative_error=float(z['T'][1,0,0]/expectedT-1),thin_temperature_relative_error=float(err),
 irreversible_sublimation=True,zero_source_no_heating=True),indent=2))
# Real graphite: no-attenuation temperatures must reproduce the existing thermal core.
from scripts import run_v4_model_grid as g
op,th,grains,look=g._cached_dust_static('graphite',.1,1.)
from src.v4_wise_emission import grain_size_number_weights
rr=np.array([.2,.5,1.,1.5]);tt=np.arange(5.)*10;lum=np.array([0,1e43,3e43,1e43,1e42])
cc=np.pi*(grains*1e-6)**2
cross=cc[:,None]*op.q_abs(th.wavelength_micron[None,:],grains[:,None])
num=np.ones((4,len(grains)))*1e35
new=solve_direct(tt,lum,th.wavelength_m,th.source_sed,cross,th.source_mean_qabs(grains),look,rr,num,.7,1800.,attenuation=False)
old=g._build_dust_response_cached(tt,lum,rr,grains,look,'graphite',1e9)
assert np.all(old.alive) and np.all(new['alive'])
err=np.max(np.abs(new['T']-old.temperature_K))/np.max(old.temperature_K)
assert err<1e-10
print('PASS real-graphite unattenuated thermal-core equivalence:',err)
