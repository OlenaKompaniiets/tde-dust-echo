"""Independently reconstruct archived chi-square; no RT run."""
from pathlib import Path
import argparse,json,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=Path(__file__).resolve().parents[1]/'examples/case_a');a=ap.parse_args()
s=json.loads((a.folder/'summary.json').read_text());e=np.genfromtxt(a.folder/'epochs.csv',delimiter=',',names=True)
chis=[]
for b in ('W1','W2'):
 r=(e[b+'_data_mJy']-e[b+'_model_mJy'])/e[b+'_sigma_mJy'];v=float(r@r);chis.append(v)
 np.testing.assert_allclose(v,s['fit']['chi2_'+b.lower()],rtol=0,atol=1e-7)
 print(b,v)
np.testing.assert_allclose(sum(chis),s['fit']['chi2_total'],rtol=0,atol=1e-7)
print('PASS chi2 =',sum(chis),'nominal conditional dof =',2*len(e)-7)
