"""Plots from the exploratory profiles and, when present, direct RT checks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':17,'axes.labelsize':18,'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42,'ps.fonttype':42})

def main():
 r=json.loads((P/'profiles.json').read_text());keys=['eta','tau_visc','tau_C','tau_E','theta','inc','t0_mjd']
 labels=[r'$\eta$',r'$\tau_{\rm visc}$ (d)',r'$\tau_{\rm C}$',r'$\tau_{\rm E}$',r'$\theta_{\rm open}$ (deg)',r'$i$ (deg)',r'$t_0$ (MJD)']
 out=P.parent/'results/v48_case_b_validation';out.mkdir(parents=True,exist_ok=True)
 fig,axes=plt.subplots(3,3,figsize=(18,13))
 for j,(k,label,ax) in enumerate(zip(keys,labels,axes.flat)):
  nodes=[r['baseline']]+[q for sign in [-1,1] for q in r['sides'][k+str(sign)]['nodes']]
  xx=[r['x0'][j]+q['v'][j]*r['scale'][j] if j<6 else r['time_reference']+20*q['v'][6] for q in nodes];yy=[q['chi2']-r['baseline']['chi2'] for q in nodes];order=np.argsort(xx)
  ax.plot(np.array(xx)[order],np.array(yy)[order],'o--',color='#3977a5',ms=4)
  ax.xaxis.set_major_locator(MaxNLocator(4));ax.axhline(1,color='0.4',ls=':');ax.set(xlabel=label,ylabel=r'$\chi^2_{\rm em}-\chi^2_{\rm em,ref}$',ylim=(-.1,4))
  for sign in [-1,1]:
   side=r['sides'][k+str(sign)]
   if 'boundary_value' in side:ax.axvline(side['boundary_value'],color='#c85040',ls=':')
 for ax in axes.flat[7:]:ax.axis('off')
 axes.flat[7].text(0,.95,'Exploratory interpolation\nNot final confidence intervals\n\nDotted red: archive bound\nCrossings require direct RT checks',va='top',fontsize=17,transform=axes.flat[7].transAxes)
 fig.tight_layout();fig.savefig(out/'case_b_exploratory_profiles.pdf');fig.savefig(out/'case_b_exploratory_profiles.png',dpi=180);plt.close(fig)
 report_path=out/'report.json'
 if report_path.exists():
  report=json.loads(report_path.read_text());rr=report['results']
  if rr:
   fig,ax=plt.subplots(figsize=(11,6));names=[v['name'] for v in rr];ax.scatter(range(len(rr)),[v['chi2']-r['reference']['chi2'] for v in rr]);ax.axhline(0,color='0.5',ls=':');ax.set_xticks(range(len(rr)),names,rotation=60,ha='right');ax.set(ylabel=r'$\chi^2_{\rm RT}-59.75434$',title='Direct RT checks: Case B');fig.tight_layout();fig.savefig(out/'case_b_rt_checks.pdf');fig.savefig(out/'case_b_rt_checks.png',dpi=200);plt.close(fig)
 print('FIGURES',out,flush=True)

if __name__=='__main__':main()
