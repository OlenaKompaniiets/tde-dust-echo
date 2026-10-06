"""
UGC 11487 V4.3b — local extended-dust geometry test.

Purpose
-------
Keep the V4.3 temporal solution family and compact graphite component fixed,
then test whether a cooler/more extended graphite response can reproduce the
remaining W2 excess without adding a third dust zone or phenomenological
band-specific terms.

Temporal drivers tested:
  1) V4.3 global best: m3.0_t0.0/0.850.dat
  2) robust near-best:  m0.1_t1.0/0.850.dat
  3) robust near-best:  m0.1_t0.0/0.850.dat

tau_visc = 200, 250, 300, 350, 400 d

Compact population fixed:
  graphite, Rin=0.15 pc, Rout=0.50 pc, p=0,
  a=0.1--1.0 micron, q=3.5.

Extended population local grid:
  Rin = 0.40, 0.50, 0.60, 0.75, 1.00 pc
  Rout = 1.5, 2.0, 3.0, 4.0, 5.0 pc, requiring Rout > Rin
  p = 0, 1, 2
  graphite, a=0.1--1.0 micron, q=3.5.

For every physical block, t0 is scanned coarsely then continuously refined.
The two non-negative f_abs coefficients are solved directly from the 44 WISE
measurements with f_comp<=0.8, f_ext<=0.8, total<=0.8.

No peak lag, centroid lag, fluence ratio, width, colour, or flux-flux slope is
used as a penalty.

Run:
    python scripts\\run_v43b_extended_geometry_test.py --workers 16

Resume:
    same command; completed blocks are skipped.
"""

from __future__ import annotations
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.run_v4_model_grid as g

OUT = ROOT/"results"/"v43b_extended_geometry_test"
CSV = OUT/"v43b_blocks.csv"
ERR = OUT/"v43b_errors.jsonl"

DRIVERS = (
    "input/m3.0_t0.0/0.850.dat",
    "input/m0.1_t1.0/0.850.dat",
    "input/m0.1_t0.0/0.850.dat",
)
TAUS = (200.,250.,300.,350.,400.)
ETA=0.10
DENS=1e9
THETA=60.
INC=55.
GRAIN=(0.10,1.00,3.5)
COMPACT=(0.15,0.50,0.0)
EXT_RIN=(0.40,0.50,0.60,0.75,1.00)
EXT_ROUT=(1.50,2.00,3.00,4.00,5.00)
EXT_P=(0.0,1.0,2.0)
T0GRID=np.arange(56800.,57400.1,20.)
FMAX=.8
FTOT=.8

def solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2):
    y=np.r_[ (y1-b1)/s1, (y2-b2)/s2 ]
    A=np.column_stack((np.r_[c1/s1,c2/s2],np.r_[e1/s1,e2/s2]))
    sol=lsq_linear(A,y,bounds=(np.zeros(2),np.full(2,FMAX)),
                   method="trf",lsmr_tol="auto")
    f=np.asarray(sol.x,float)
    if f.sum()>FTOT:
        d=A[:,0]-A[:,1]; y0=y-A[:,1]*FTOT
        den=float(d@d)
        fc=.5*FTOT if den<=0 else float((d@y0)/den)
        fc=float(np.clip(fc,0,FTOT)); f=np.array([fc,FTOT-fc])
    m1=b1+f[0]*c1+f[1]*e1
    m2=b2+f[0]*c2+f[1]*e2
    r1=(y1-m1)/s1; r2=(y2-m2)/s2
    return f,m1,m2,float(r1@r1),float(r2@r2)

def build_echo(stars_rel,tau,zone):
    driver,diag=g.build_physical_driver(
        stars_file=g.STARS_ROOT/stars_rel,M_BH_Msun=g.M_BH_MSUN,
        eta_rad=ETA,tau_visc_days=float(tau),t_circ_days=g.T_CIRC_DAYS,
        dt_days=g.DRIVER_DT_DAYS,pre_peak_days=g.PRE_PEAK_DAYS,
        post_fallback_days=g.POST_FALLBACK_DAYS)
    td=driver["t_days"].to_numpy(float)
    L=driver["L_bol_erg_s"].to_numpy(float)
    amin,amax,q=GRAIN
    opacity,thermal,grains,lookup=g._cached_dust_static("graphite",amin,amax)
    rin,rout,p=zone
    radii,rw=g.radial_grid_and_weights(rin,rout,p,g.N_RADII)
    dust=g._build_dust_response_cached(
        time_days=td,luminosity_erg_s=L,radius_pc=radii,grains=grains,
        lookup=lookup,composition="graphite",density_cm3=DENS)
    pop=g.build_population_normalization(
        thermal_solver=thermal,grain_radius_micron=grains,
        clump_radius_pc=radii,f_abs_initial=1.,q=q,clump_weight=rw)
    w1,w2,vw,vf=g._cached_wise_static()
    dl=g.D_L_MPC*1e6*g.PC_M
    pg1=g.absolute_population_band_cube(
        opacity,w1,grains,dust.temperature_K,dust.alive,q,g.Z,dl,vw,vf,g.WISE_TGRID)
    pg2=g.absolute_population_band_cube(
        opacity,w2,grains,dust.temperature_K,dust.alive,q,g.Z,dl,vw,vf,g.WISE_TGRID)
    rc,zc=g._cached_geometry(rin,rout,p,THETA,INC)
    e1,e2=g.integrate_radial_responses_over_clumps(
        dust_time_rest_days=td,radius_grid_pc=radii,
        radial_responses=np.stack((pop.total_grain_number*pg1,
                                   pop.total_grain_number*pg2)),
        clump_radius_pc=rc,clump_z_obs_pc=zc,redshift=g.Z,
        chunk_size=256,source_time_input=True)
    return dict(time=np.asarray(e1.time_obs_days,float),
                w1=np.asarray(e1.response,float),
                w2=np.asarray(e2.response,float),
                alive=float(np.mean(dust.alive[-1])),
                Tpeak=float(np.nanmax(dust.temperature_K)),
                mass=float(diag["accretion_mass_ratio"]))

def ip(pop,t):
    return (np.interp(t,pop["time"],pop["w1"],left=0,right=0),
            np.interp(t,pop["time"],pop["w2"],left=0,right=0))

def block_id(s,tau,rin,rout,p):
    return hashlib.sha1(f"{s}|{tau}|{rin}|{rout}|{p}".encode()).hexdigest()[:16]

def worker(task):
    s,tau,rin,rout,p,rec=task
    obs=pd.DataFrame(rec)
    mjd=obs.mjd.to_numpy(float); y1=obs.F_W1_Jy.to_numpy(float)
    y2=obs.F_W2_Jy.to_numpy(float); s1=obs.sigma_W1_Jy.to_numpy(float)
    s2=obs.sigma_W2_Jy.to_numpy(float)
    b1=float(obs.baseline_W1_mJy.iloc[0])/1000
    b2=float(obs.baseline_W2_mJy.iloc[0])/1000
    cp=build_echo(s,tau,COMPACT)
    ep=build_echo(s,tau,(rin,rout,p))
    def obj(t0,payload=False):
        tr=mjd-float(t0); c1,c2=ip(cp,tr); e1,e2=ip(ep,tr)
        z=solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2)
        return ((z[3]+z[4],z) if payload else z[3]+z[4])
    vals=np.array([obj(x) for x in T0GRID])
    tc=float(T0GRID[int(np.argmin(vals))])
    lo=max(T0GRID[0],tc-25); hi=min(T0GRID[-1],tc+25)
    op=minimize_scalar(obj,bounds=(lo,hi),method="bounded",
                       options={"xatol":.05,"maxiter":80})
    chi,z=obj(float(op.x),True); f,m1,m2,ch1,ch2=z
    return dict(block_id=block_id(s,tau,rin,rout,p),stars_rel=s,
        tau_visc_days=tau,extended_Rin_pc=rin,extended_Rout_pc=rout,
        extended_p=p,best_t0_mjd=float(op.x),f_abs_compact=float(f[0]),
        f_abs_extended=float(f[1]),f_abs_total=float(f.sum()),
        chi2_w1=ch1,chi2_w2=ch2,chi2_total=chi,
        compact_alive_final=cp["alive"],extended_alive_final=ep["alive"],
        compact_Tpeak_K=cp["Tpeak"],extended_Tpeak_K=ep["Tpeak"],
        mass_ratio=cp["mass"])

def done_ids():
    if not CSV.exists(): return set()
    try:return set(pd.read_csv(CSV).block_id.astype(str))
    except:return set()

def append(x):
    pd.DataFrame([x]).to_csv(CSV,mode="a",header=not CSV.exists(),index=False)

def best_products(obs,b):
    mjd=obs.mjd.to_numpy(float); y1=obs.F_W1_Jy.to_numpy(float)
    y2=obs.F_W2_Jy.to_numpy(float); s1=obs.sigma_W1_Jy.to_numpy(float)
    s2=obs.sigma_W2_Jy.to_numpy(float)
    b1=float(obs.baseline_W1_mJy.iloc[0])/1000
    b2=float(obs.baseline_W2_mJy.iloc[0])/1000
    cp=build_echo(b["stars_rel"],b["tau_visc_days"],COMPACT)
    ep=build_echo(b["stars_rel"],b["tau_visc_days"],
                  (b["extended_Rin_pc"],b["extended_Rout_pc"],b["extended_p"]))
    tr=mjd-b["best_t0_mjd"]; c1,c2=ip(cp,tr); e1,e2=ip(ep,tr)
    f,m1,m2,ch1,ch2=solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2)
    tab=pd.DataFrame(dict(mjd=mjd,W1_data_mJy=1000*y1,W1_model_mJy=1000*m1,
        W1_sigma_mJy=1000*s1,W1_resid_sigma=(y1-m1)/s1,
        W2_data_mJy=1000*y2,W2_model_mJy=1000*m2,
        W2_sigma_mJy=1000*s2,W2_resid_sigma=(y2-m2)/s2))
    tab.to_csv(OUT/"v43b_best_epoch_table.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,7))
    ax.errorbar(mjd,1000*y1,yerr=1000*s1,fmt="o",label="WISE W1")
    ax.errorbar(mjd,1000*y2,yerr=1000*s2,fmt="s",label="WISE W2")
    ax.plot(mjd,1000*m1,"-o",ms=3,label="V4.3b W1")
    ax.plot(mjd,1000*m2,"-s",ms=3,label="V4.3b W2")
    ax.set(xlabel="MJD",ylabel="Flux density (mJy)",
           title=f"V4.3b best | chi2={ch1+ch2:.2f} (W1={ch1:.2f}, W2={ch2:.2f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT/"v43b_best_lightcurve.png",dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=16)
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    obs=g.load_observations(); rec=obs.to_dict("records"); done=done_ids()
    tasks=[]
    for s in DRIVERS:
      for tau in TAUS:
       for ri in EXT_RIN:
        for ro in EXT_ROUT:
         if ro<=ri: continue
         for p in EXT_P:
          bid=block_id(s,tau,ri,ro,p)
          if bid not in done: tasks.append((s,tau,ri,ro,p,rec))
    total=sum(1 for s in DRIVERS for tau in TAUS for ri in EXT_RIN
              for ro in EXT_ROUT if ro>ri for p in EXT_P)
    print("="*90)
    print("UGC 11487 V4.3b — LOCAL EXTENDED-DUST GEOMETRY TEST")
    print("total blocks:",total," completed:",len(done)," remaining:",len(tasks))
    print("workers:",a.workers)
    print("="*90)
    t=time.perf_counter(); errors=0
    if tasks:
      with ProcessPoolExecutor(max_workers=a.workers) as ex:
       fs={ex.submit(worker,x):x for x in tasks}
       for i,f in enumerate(as_completed(fs),1):
        x=fs[f]
        try:
         r=f.result(); append(r)
         print(f"[{i:04d}/{len(tasks):04d}] chi2={r['chi2_total']:.3f} "
               f"W1={r['chi2_w1']:.1f} W2={r['chi2_w2']:.1f} "
               f"{r['stars_rel']} tau={r['tau_visc_days']:.0f} "
               f"Rext={r['extended_Rin_pc']:.2f}-{r['extended_Rout_pc']:.2f} "
               f"p={r['extended_p']:.0f} f={r['f_abs_total']:.3f}")
        except Exception as e:
         errors+=1
         with ERR.open("a",encoding="utf8") as h:
          h.write(json.dumps({"task":x[:5],"error":repr(e)})+"\n")
         print("ERROR",x[:5],repr(e))
    d=pd.read_csv(CSV).sort_values("chi2_total").reset_index(drop=True)
    b=d.iloc[0].to_dict(); best_products(obs,b)
    man=dict(experiment="V4.3b local extended-dust geometry test",
        drivers=DRIVERS,tau_visc_days=TAUS,eta_rad=ETA,
        compact_zone=COMPACT,extended_Rin_pc=EXT_RIN,
        extended_Rout_pc=EXT_ROUT,extended_p=EXT_P,grain_config=GRAIN,
        composition="graphite_graphite",f_abs_total_max=FTOT,
        n_blocks_total=total,best=b,errors_this_run=errors,
        wall_time_s=time.perf_counter()-t)
    (OUT/"v43b_manifest.json").write_text(json.dumps(man,indent=2,default=float))
    print("="*90)
    print("V4.3b COMPLETE")
    print("successful:",len(d),"/",total," errors:",errors)
    print("BEST chi2:",f"{b['chi2_total']:.6f}",
          " W1:",f"{b['chi2_w1']:.6f}"," W2:",f"{b['chi2_w2']:.6f}")
    print("BEST STARS:",b["stars_rel"]," tau:",b["tau_visc_days"])
    print("BEST Rext:",b["extended_Rin_pc"],"-",b["extended_Rout_pc"],
          " p:",b["extended_p"])
    print("BEST t0:",b["best_t0_mjd"]," f_abs:",b["f_abs_total"])
    print("results:",OUT)
    print("="*90)

if __name__=="__main__":
    main()
