"""Nine representative blocks, full production resolution; separate checkpoint."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_v4_model_grid as g

def main():
    out=ROOT/'results/validation';out.mkdir(parents=True,exist_ok=True)
    path=out/'control_blocks.csv'
    if path.exists():raise RuntimeError('Control output exists; archive it before rerunning')
    (out/'control_manifest.json').write_text(json.dumps(g.experiment_manifest(),indent=2,default=str))
    obs=g.load_observations().to_dict(orient='list')
    for rel in ['input/m0.3_t0.0/0.900.dat','input/m1.0_t0.57/2.000.dat','input/m1.0_t1.0/2.000.dat']:
        for tau in [0.,100.,500.]:
            b=dict(eta_rad=.1,tau_visc_days=tau,composition='graphite',density_cm3=1e9,
                   grain_config=(.01,1.,3.5),r_in_pc=.15,r_out_pc=1.,p_radial=2.,
                   theta_open_deg=60.,inclination_deg=55.,stars_file=str(g.STARS_ROOT/rel),stars_rel=rel)
            row=g.evaluate_block(b,obs)
            g.append_row(path,row)
            print(rel,tau,row.get('chi2_total'),row.get('alive_grid_fraction_final'),row.get('runtime_s'),flush=True)
            if row['status']!='ok':raise RuntimeError(row)
    print('CONTROL RUN COMPLETE',path)
if __name__=='__main__':main()
