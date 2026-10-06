"""
UGC 11487 V4.4 — two-layer radiative-transfer CONTROL.

This is intentionally a small mechanism test around the V4.3b best solution.
It does NOT fit a new arbitrary source temperature, add a third dust zone, or
use empirical W2-lag penalties.

Fixed physical state from V4.3b best:
  STARS: input/m3.0_t0.0/0.850.dat
  tau_visc = 350 d
  compact graphite: 0.15--0.50 pc, p=0
  extended graphite: 0.75--3.00 pc, p=1
  grains: 0.1--1.0 micron, q=3.5
  f_abs compact = 0.2627, extended = 0.5125
  t0 = 57099.29 MJD

Control grid:
  direct transmission = 1.00, 0.75, 0.50, 0.25
  secondary coupling  = 0.00, 0.25, 0.50, 0.75, 1.00

The compact dust spectrum is calculated physically at every model time and
used to heat the extended graphite with its actual wavelength-dependent Qabs.
The compact population is treated as an unresolved isotropic reprocessor for
this control.  Therefore the experiment tests whether the mechanism has the
right sign/magnitude; it is not yet full 3-D radiative transfer.

Run:
  python scripts\\run_v44_rt_control.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import scripts.run_v4_model_grid as g
from src.v4_two_layer_rt import compact_population_spectrum, mixed_extended_temperature

OUT=ROOT/"results"/"v44_rt_control"
STARS="input/m3.0_t0.0/0.850.dat"; TAU=350.; ETA=.10
COMPACT=(.15,.50,0.); EXT=(.75,3.0,1.)
GRAIN=(.10,1.0,3.5); DENS=1e9; THETA=60.; INC=55.
FC=.2627; FE=.5125; T0=57099.29
TRANSMISSIONS=(1.00,.75,.50,.25)
COUPLINGS=(0.,.25,.50,.75,1.0)


def echo_from_temperature(td,zone,opacity,thermal,grains,T,alive,pop,q):
    rin,rout,p=zone; radii,_=g.radial_grid_and_weights(rin,rout,p,g.N_RADII)
    w1,w2,vw,vf=g._cached_wise_static(); dl=g.D_L_MPC*1e6*g.PC_M
    pg1=g.absolute_population_band_cube(opacity,w1,grains,T,alive,q,g.Z,dl,vw,vf,g.WISE_TGRID)
    pg2=g.absolute_population_band_cube(opacity,w2,grains,T,alive,q,g.Z,dl,vw,vf,g.WISE_TGRID)
    rc,zc=g._cached_geometry(rin,rout,p,THETA,INC)
    e1,e2=g.integrate_radial_responses_over_clumps(
        dust_time_rest_days=td,radius_grid_pc=radii,
        radial_responses=np.stack((pop.total_grain_number*pg1,pop.total_grain_number*pg2)),
        clump_radius_pc=rc,clump_z_obs_pc=zc,redshift=g.Z,chunk_size=256,source_time_input=True)
    return np.asarray(e1.time_obs_days,float),np.asarray(e1.response,float),np.asarray(e2.response,float)


def main():
    OUT.mkdir(parents=True,exist_ok=True); tstart=time.perf_counter()
    obs=g.load_observations(); mjd=obs.mjd.to_numpy(float)
    y1=obs.F_W1_Jy.to_numpy(float); y2=obs.F_W2_Jy.to_numpy(float)
    s1=obs.sigma_W1_Jy.to_numpy(float); s2=obs.sigma_W2_Jy.to_numpy(float)
    b1=float(obs.baseline_W1_mJy.iloc[0])/1000.; b2=float(obs.baseline_W2_mJy.iloc[0])/1000.

    driver,diag=g.build_physical_driver(stars_file=g.STARS_ROOT/STARS,M_BH_Msun=g.M_BH_MSUN,
        eta_rad=ETA,tau_visc_days=TAU,t_circ_days=g.T_CIRC_DAYS,dt_days=g.DRIVER_DT_DAYS,
        pre_peak_days=g.PRE_PEAK_DAYS,post_fallback_days=g.POST_FALLBACK_DAYS)
    td=driver.t_days.to_numpy(float); L=driver.L_bol_erg_s.to_numpy(float)
    amin,amax,q=GRAIN
    opacity,thermal,grains,lookup=g._cached_dust_static("graphite",amin,amax)

    # Compact physical state for f_abs=1 temperature; temperature does not depend on number normalization.
    rc_grid,rwc=g.radial_grid_and_weights(*COMPACT,g.N_RADII)
    dc=g._build_dust_response_cached(time_days=td,luminosity_erg_s=L,radius_pc=rc_grid,
        grains=grains,lookup=lookup,composition="graphite",density_cm3=DENS)
    pc=g.build_population_normalization(thermal_solver=thermal,grain_radius_micron=grains,
        clump_radius_pc=rc_grid,f_abs_initial=1.,q=q,clump_weight=rwc)
    # Actual compact spectrum uses the fixed V4.3b compact absorbed fraction.
    lam,Llam,Lcbol=compact_population_spectrum(thermal,opacity,dc.temperature_K,dc.alive,
        rc_grid,rwc,grains,q,pc.total_grain_number,FC)

    # Compact observer echo with its actual FC normalization.
    tc,c1ref,c2ref=echo_from_temperature(td,COMPACT,opacity,thermal,grains,dc.temperature_K,dc.alive,pc,q)
    c1=FC*c1ref; c2=FC*c2ref

    re_grid,rwe=g.radial_grid_and_weights(*EXT,g.N_RADII)
    pe=g.build_population_normalization(thermal_solver=thermal,grain_radius_micron=grains,
        clump_radius_pc=re_grid,f_abs_initial=1.,q=q,clump_weight=rwe)

    rows=[]; best_payload=None
    trel=mjd-T0
    ci1=np.interp(trel,tc,c1,left=0,right=0); ci2=np.interp(trel,tc,c2,left=0,right=0)

    for tr in TRANSMISSIONS:
      for eps in COUPLINGS:
        T,alive,Fdir,Fsec=mixed_extended_temperature(td,L,re_grid,grains,thermal,lookup,opacity,
            "graphite",DENS,Llam,lam,direct_transmission=tr,secondary_coupling=eps)
        te,e1ref,e2ref=echo_from_temperature(td,EXT,opacity,thermal,grains,T,alive,pe,q)
        ei1=FE*np.interp(trel,te,e1ref,left=0,right=0); ei2=FE*np.interp(trel,te,e2ref,left=0,right=0)
        m1=b1+ci1+ei1; m2=b2+ci2+ei2
        r1=(y1-m1)/s1; r2=(y2-m2)/s2
        ch1=float(r1@r1); ch2=float(r2@r2); chi=ch1+ch2
        secfrac=float(np.nanmax(Fsec/(Fdir[:,:,None]+Fsec+1e-300)))
        row=dict(direct_transmission=tr,secondary_coupling=eps,chi2_w1=ch1,chi2_w2=ch2,
            chi2_total=chi,extended_Tpeak_K=float(np.nanmax(T)),extended_alive_final=float(np.mean(alive[-1])),
            max_secondary_fraction_of_equivalent_heating=secfrac,
            compact_Lbol_peak_erg_s=float(np.max(Lcbol)),mass_ratio=float(diag["accretion_mass_ratio"]))
        rows.append(row); print(f"Tdir={tr:.2f} eps={eps:.2f} chi2={chi:.3f} W1={ch1:.2f} W2={ch2:.2f} Text={row['extended_Tpeak_K']:.0f}K")
        if best_payload is None or chi<best_payload[0]: best_payload=(chi,row,m1,m2,r1,r2)

    df=pd.DataFrame(rows).sort_values("chi2_total").reset_index(drop=True)
    df.to_csv(OUT/"v44_rt_control.csv",index=False)
    chi,b,m1,m2,r1,r2=best_payload
    tab=pd.DataFrame(dict(mjd=mjd,W1_data_mJy=1000*y1,W1_model_mJy=1000*m1,W1_resid_sigma=r1,
                          W2_data_mJy=1000*y2,W2_model_mJy=1000*m2,W2_resid_sigma=r2))
    tab.to_csv(OUT/"v44_best_epoch_table.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,7));
    ax.errorbar(mjd,1000*y1,yerr=1000*s1,fmt="o",label="WISE W1"); ax.errorbar(mjd,1000*y2,yerr=1000*s2,fmt="s",label="WISE W2")
    ax.plot(mjd,1000*m1,"-o",ms=3,label="V4.4 W1"); ax.plot(mjd,1000*m2,"-s",ms=3,label="V4.4 W2")
    ax.set(xlabel="MJD",ylabel="Flux density (mJy)",title=f"V4.4 RT control best | chi2={chi:.2f} (Tdir={b['direct_transmission']:.2f}, eps={b['secondary_coupling']:.2f})")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/"v44_best_lightcurve.png",dpi=180); plt.close(fig)
    manifest=dict(experiment="V4.4 two-layer RT control",stars=STARS,tau_visc_days=TAU,t0_mjd=T0,
        compact_zone=COMPACT,extended_zone=EXT,grain_config=GRAIN,f_abs_compact=FC,f_abs_extended=FE,
        direct_transmissions=TRANSMISSIONS,secondary_couplings=COUPLINGS,
        approximation="compact population collapsed to unresolved isotropic reprocessor; actual compact dust spectrum and Qabs-weighted secondary heating",
        best=b,wall_time_s=time.perf_counter()-tstart)
    (OUT/"v44_manifest.json").write_text(json.dumps(manifest,indent=2,default=float),encoding="utf8")
    print("="*90); print("V4.4 CONTROL COMPLETE"); print(df.head(10).to_string(index=False)); print("results:",OUT); print("="*90)

if __name__=="__main__": main()
