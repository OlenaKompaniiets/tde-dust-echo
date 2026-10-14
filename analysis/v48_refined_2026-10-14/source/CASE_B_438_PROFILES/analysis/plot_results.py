import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linprog
p=Path(__file__).parent;r=json.loads((p/'profiles.json').read_text());a=np.load(p/'response_cache.npz');X=a['X'];K=['eta','tau_visc','tau_C','tau_E','theta','inc'];scale=np.array([.01,25,.025,.15,2.5,2.5]);x0=np.array([r['RT_reference']['config'][k] for k in K]);U=(X-x0)/scale;base=r['baseline'];rows=[]
plt.rcParams.update({'font.size':15,'axes.labelsize':16,'xtick.labelsize':13,'ytick.labelsize':13,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(15,9));labels=[r'$\eta$',r'$\tau_{\rm visc}$ (d)',r'$\tau_{\rm C}$',r'$\tau_{\rm E}$',r'$\theta_{\rm open}$ (deg)',r'$i$ (deg)']
for j,(k,ax) in enumerate(zip(K,axs.flat)):
 qs=[base]+[q for sign in [-1,1] for q in r['sides'][k+str(sign)]['nodes']];d={}
 for q in qs:
  v=np.array(q['v']);value=float(x0[j]+v[j]*scale[j]);old=d.get(value)
  if old is None or q['chi2']<old['chi2']:d[value]=q
 vals=sorted(d);chi=np.array([d[v]['chi2']-base['chi2'] for v in vals]);inside=[]
 for value in vals:
  q=d[value];u=np.array(q['v'][:6]);ok=bool(linprog(np.zeros(len(U)),A_eq=np.vstack([U.T,np.ones(len(U))]),b_eq=np.r_[u,1],bounds=(0,None),method='highs').success);inside.append(ok)
  rows.append(dict(parameter=k,value=value,chi2_emulator=q['chi2'],delta_chi2= q['chi2']-base['chi2'],inside_training_convex_hull=ok))
 vals=np.array(vals);inside=np.array(inside);ax.plot(vals,chi,'--',color='#375b80',lw=1.3)
 ax.scatter(vals[inside],chi[inside],s=28,color='#256a92',label='Inside training hull')
 ax.scatter(vals[~inside],chi[~inside],s=32,facecolors='none',edgecolors='#bd5a32',label='Extrapolated')
 ax.axhline(1,color='.35',ls=':',lw=1.2);ax.axvline(r['RT_reference']['config'][k],color='.65',ls=':',lw=1)
 ax.set(xlabel=labels[j],ylabel=rf'$\chi^2_{{\rm em}}-{base["chi2"]:.5f}$',ylim=(-.12,3.5));ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(direction='in',top=True,right=True)
 for sign in [-1,1]:
  s=r['sides'][k+str(sign)]
  if s['status'].startswith('no_crossing'):
   ax.axvline(s['boundary_value'],color='#bd5a32',lw=1,ls='--')
handles,labs=axs[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.96));fig.suptitle('Case B: provisional response-emulator profiles',y=.995,fontsize=18)
fig.text(.5,.015,'438 RT responses. Emulator minimum; confidence limits are not RT validated.\nVertical grey: best saved RT parameter. Orange dashed: unclosed side at archive coverage boundary.',ha='center',fontsize=12)
fig.tight_layout(rect=(0,.065,1,.91));fig.savefig(p/'case_b_profiles_438.pdf');fig.savefig(p/'case_b_profiles_438.png',dpi=180);plt.close(fig)
with (p/'profile_points.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('figure and',len(rows),'profile points saved')
