"""Rebuild comparison figure and parameter table from archived results only."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'paper';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':18,'axes.labelsize':18,'legend.fontsize':18,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7})
fig,axs=plt.subplots(2,2,figsize=(14.2,9),sharex=True,gridspec_kw={'height_ratios':[3,1]})
styles={'local':('#0072B2','-'),'grid':('#D55E00','--')};summaries={}
for name,(color,ls) in styles.items():
 folder=P/'archived'/name;summaries[name]=json.loads((folder/'summary.json').read_text())
 e=np.genfromtxt(folder/'epochs.csv',delimiter=',',names=True)
 d=np.genfromtxt(folder/'dense.csv',delimiter=',',names=True)
 for j,b in enumerate(('W1','W2')):
  # Use the actual archived dense model curves.
  axs[0,j].plot(d['mjd'],d[b+'_model_mJy'],ls,color=color,lw=1.2,label=name)
  axs[1,j].plot(e['mjd'],e[b+'_residual_sigma'],'o',ms=3,color=color,label=name)
  if name=='local':axs[0,j].errorbar(e['mjd'],e[b+'_data_mJy'],yerr=e[b+'_sigma_mJy'],fmt='o',ms=3,color='black',elinewidth=.7,capsize=2,label='WISE')
  axs[0,j].set_title(b);axs[1,j].axhline(0,color='.5',lw=.6);axs[1,j].set_xlabel('MJD')
axs[0,0].set_ylabel('Flux density [mJy]');axs[1,0].set_ylabel(r'$(F_{obs}-F_{mod})/\sigma$');axs[0,0].legend(frameon=False)
fig.tight_layout();fig.savefig(O/'fit_comparison.pdf');fig.savefig(O/'fit_comparison.png',dpi=300);plt.close(fig)
keys=['eta','tau_visc','tau_C','tau_E','theta','inc']
with (O/'parameters.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['quantity','local','grid','uncertainty_status'])
 for k in keys:w.writerow([k,*[summaries[n]['config'][k] for n in styles],'not yet estimated'])
 for k in ['chi2_total','chi2_w1','chi2_w2','t0_mjd']:w.writerow([k,*[summaries[n]['fit'][k] for n in styles],'not yet estimated' if k=='t0_mjd' else 'objective value'])
 for k in ['E_source_window_erg','E_direct_absorbed_all_arrivals_erg','E_IR_eventual_escape_from_computed_emission_erg']:
  w.writerow([k,*[summaries[n]['energy'][k] for n in styles],'conditional derived quantity; uncertainty pending'])
print(O)
