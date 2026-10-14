"""Rebuild sampled emulator profiles using the already supplied response-analysis package; no RT."""
import sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;source=P.parent/'V48_PROFILE_UPDATE';old=P.parent/'V48_RESPONSE_VALIDATION'
sys.path.insert(0,str(source));sys.argv=[str(source/'update_profiles.py'),'--cache',str(old),'--validation',str(P/'previous_validation.zip')]
# Reuse the exact fitted response functions, stopping before the original profile loop.
ns={'__file__':str(source/'update_profiles.py'),'__name__':'profile_sampling'}
exec(compile((source/'update_profiles.py').read_text().split('starts=[np.zeros(7)]')[0],str(source/'update_profiles.py'),'exec'),ns)
r=json.loads((P/'updated_profiles.json').read_text());base=np.array(r['baseline']['v']);out={'reference_chi2':r['baseline']['chi2'],'profiles':{},'scope':'sampled emulator profiles; RT points are separate fixed configurations'}
for j,k in enumerate(ns['K']):
 low=r['sides'][k+'-1']['endpoint'];high=r['sides'][k+'1']['endpoint'];center=r['baseline']['config'][k];grid=np.unique(np.r_[np.linspace(low,center,10),np.linspace(center,high,10)]);warm=base.copy();rows=[]
 for value in grid:
  q=ns['multi']([warm,base],{j:(value-ns['x0'][j])/ns['SCALE'][j]});warm=np.array(q['v']);rows.append(dict(value=float(value),chi2=q['chi2'],optimality=q['optimality'],multistart_spread=q['multistart_spread']))
 out['profiles'][k]=rows;(P/'profile_samples.json').write_text(json.dumps(out,indent=2));print('SAMPLED',k,flush=True)
