"""Regression checks independent of observed-fit success. Run from any cwd."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.signal import convolve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.v4_driver import causal_exponential_response
from src.v4_observer_echo import deposit_linear,PC_TO_LIGHT_DAYS
from src.v4_3d_echo import integrate_radial_responses_over_clumps as echo
from src.v4_opacity import load_dust_opacity
from src.v4_thermal import DustThermalEquilibrium
from src.v4_thermal_lookup import build_thermal_lookup,lookup_temperature_from_flux,direct_temperature_from_flux
import run_v4_model_grid as g

def main():
    report={}
    t=np.arange(-1200.,25000.,10.);x=((t>=0)&(t<100)).astype(float)
    for tau in [100.,500.]:
        y=causal_exponential_response(t,x,tau)
        assert np.all(y[t<0]==0) and np.all(y>=0)
        assert abs(y.sum()/x.sum()-1)<1e-12
        report[f'visc_mass_ratio_{tau}']=float(y.sum()/x.sum())
    assert np.isclose(deposit_linear([0,1,2],[2,2-1e-12],[3,4]).sum(),7)
    # Independent direct deposition, arbitrary clumps and redshift, two bands.
    t=np.arange(-20.,201.,2.);rg=np.array([.01,.03,.08])
    rc=np.array([.01,.02,.05,.08]);zc=np.array([.009,0.,-.02,.03]);z=.01895
    y=np.exp(-.5*((t-30)/7)**2);y[t<0]=0
    rr=np.stack([np.outer(y,[1.,2.,.4]),np.outer(y,[2.,.1,1.])])
    es=echo(t,rg,rr,rc,zc,redshift=z,source_time_input=True)
    for b,e in enumerate(es):
        direct=np.zeros_like(e.response)
        for i in range(rc.size):
            hi=np.clip(np.searchsorted(rg,rc[i],side='right'),1,len(rg)-1);lo=hi-1
            f=(rc[i]-rg[lo])/(rg[hi]-rg[lo])
            yy=(rr[b,:,lo]*(1-f)+rr[b,:,hi]*f)/rc.size
            ta=(1+z)*(t+(rc[i]-zc[i])*PC_TO_LIGHT_DAYS)
            direct+=deposit_linear(e.time_obs_days,ta,yy)
        err=np.max(np.abs(direct-e.response))
        assert err<1e-12,(b,err)
        assert np.isclose(e.total_input_weight,e.total_output_weight,rtol=1e-12)
        report[f'transfer_direct_error_band{b}']=float(err)
    # Wide radial brackets used to create a spurious pre-echo in old code.
    t=np.arange(0.,3601.);rg=np.array([1.,2.])
    rr=np.stack([np.exp(-.5*((t-r*PC_TO_LIGHT_DAYS)/2)**2) for r in rg],axis=1)
    e=echo(t,rg,rr[None,:,:],[1.5],[1.5])[0]
    frac=e.response[e.time_obs_days < -10].sum()/e.response.sum()
    assert frac<1e-6,frac
    report['wide_radial_pre_echo_fraction']=float(frac)
    try:
        echo(np.arange(20.),[1,2],np.ones((1,20,2)),[1],[0],output_time_obs_days=np.arange(0.,40.,2.))
        raise AssertionError('Cadence mismatch accepted')
    except ValueError:pass
    th=DustThermalEquilibrium(load_dust_opacity(g.DRAINE_DIR,'graphite'),source_temperature_K=g.SOURCE_T_K)
    grains=np.array([.01,.1,1.]);lut=build_thermal_lookup(th,grains)
    T,alive,_=g._temperature_cube_from_cached_lookup(lut,np.array([[0.],[1e10],[1e43]]),np.array([1.]),grains,'graphite',1e9)
    assert alive.all(),alive
    report['cold_grains_survive']=True
    for k,a in enumerate(grains):
        for F in [.01,.1,1.]:
            direct=direct_temperature_from_flux(th,F,a)
            got=lookup_temperature_from_flux(lut,F,k)
            assert abs(got/direct-1)<1e-4
    # Unambiguous hot pulse destroys irreversibly, including after fading.
    T,alive,_=g._temperature_cube_from_cached_lookup(lut,np.array([[0.],[1e50],[0.]]),np.array([1.]),grains,'graphite',1e9)
    assert alive[0].all() and not alive[1:].any()
    report['hot_destruction_irreversible']=True
    report['status']='PASS'
    out=ROOT/'results/validation';out.mkdir(parents=True,exist_ok=True)
    (out/'regression_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
