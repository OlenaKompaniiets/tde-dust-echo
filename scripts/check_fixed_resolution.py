"""Numerical sensitivity for one specified control, not a parameter search."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_v4_model_grid as g
import pandas as pd
rel='input/m0.3_t0.0/0.900.dat'
b=dict(eta_rad=.1,tau_visc_days=500.,composition='graphite',density_cm3=1e9,
       grain_config=(.01,1.,3.5),r_in_pc=.15,r_out_pc=1.,p_radial=2.,
       theta_open_deg=60.,inclination_deg=55.,stars_file=str(g.STARS_ROOT/rel),stars_rel=rel)
obs=g.load_observations().to_dict(orient='list');rows=[]
for label,dt,nr,na,nc in [('default',10.,48,41,12000),('dt5',5.,48,41,12000),('radial96',10.,96,41,12000),('grain81_clump24000',10.,48,81,24000)]:
    g.DRIVER_DT_DAYS=dt;g.N_RADII=nr;g.N_GRAINS=na;g.N_CLUMPS=nc
    g.experiment_manifest.cache_clear();g.experiment_digest.cache_clear()
    g._cached_dust_static.cache_clear();g._cached_geometry.cache_clear()
    row=g.evaluate_block(b,obs);row.update(check=label,dt=dt,nr=nr,na=na,nc=nc)
    rows.append(row);print(label,row.get('chi2_total'),row.get('runtime_s'),flush=True)
    if row['status']!='ok':raise RuntimeError(row)
out=ROOT/'results/validation';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'resolution_check.csv',index=False)
