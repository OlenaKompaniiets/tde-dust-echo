"""Scientific figures from saved responses and fits. No RT or optimizer calls."""
from pathlib import Path
import json,zipfile,io,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
KEYS=['eta','tau_visc','tau_C','tau_E','theta','inc']
LABELS=[r'$\eta$',r'$\tau_{\rm visc}$ [d]',r'$\tau_{\rm C}$',r'$\tau_{\rm E}$',r'$\theta$ [deg]',r'$i$ [deg]']
SHORT={'center':'centre','eta-1':r'$\eta-$','eta1':r'$\eta+$','tau_visc-1':r'$\tau_v-$','tau_visc1':r'$\tau_v+$','tau_C-1':r'$\tau_C-$','tau_C1':r'$\tau_C+$','tau_E-1':r'$\tau_E-$','tau_E1':r'$\tau_E+$','theta-1':r'$\theta-$','theta1':r'$\theta+$','inc-1':r'$i-$','inc1':r'$i+$'}
def make_figures(out):
 out=Path(out);dest=out/'figures';dest.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':18,'axes.labelsize':20,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':15,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'savefig.dpi':300})
 profile=json.loads((HERE/'updated_profiles.json').read_text());samples=json.loads((HERE/'profile_samples.json').read_text());oldzip=zipfile.ZipFile(HERE/'previous_validation.zip');oldreport=json.loads(oldzip.read('report.json'));oldplan=json.loads(oldzip.read('validation_plan.json'));oldpoints={p['name']:p for p in oldplan['points']};newplan=json.loads((HERE/'validation_plan.json').read_text());newpoints={p['name']:p for p in newplan['points']}
 newreport=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'results':[]};oldresults={r['name']:r for r in oldreport['results']};newresults={r['name']:r for r in newreport['results']};ref=profile['baseline']['chi2'];captions=[]
 def save(fig,name):
  for ext in ['pdf','png']:fig.savefig(dest/(name+'.'+ext),bbox_inches='tight')
  plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(16,10),layout='constrained')
 for k,label,ax in zip(KEYS,LABELS,axs.ravel()):
  rows=samples['profiles'][k];ax.plot([r['value'] for r in rows],[r['chi2']-ref for r in rows],color='#1864ab',lw=2,label='Emulator profile');ax.axhline(1,color='.5',ls='--',lw=1);ax.axvline(profile['baseline']['config'][k],color='.7',ls=':',lw=1)
  for sign in [-1,1]:
   name=k+str(sign);a=oldresults[name];ax.scatter(oldpoints[name]['config'][k],a['chi2']-ref,marker='s',facecolors='none',edgecolors='.35',s=70,label='Previous RT configuration' if sign==-1 else None,zorder=4)
   if name in newresults:ax.scatter(newpoints[name]['config'][k],newresults[name]['chi2']-ref,marker='D',color='#c43d20',s=65,label='Updated RT configuration' if sign==-1 else None,zorder=5)
   else:ax.scatter(newpoints[name]['config'][k],newpoints[name]['predicted_chi2']-ref,marker='o',facecolors='none',edgecolors='#1864ab',s=80,label='Updated candidate (emulator)' if sign==-1 else None,zorder=5)
  ax.set(xlabel=label,ylabel=r'$\chi^2-\chi^2_{\rm em,ref}$',ylim=(-.06,1.65));ax.grid(alpha=.12)
 handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2);save(fig,'fig01_chi2_profiles')
 captions.append('Fig. 1. Sampled conditional emulator profiles, with three-point-start endpoint searches and two starts per displayed profile sample. Straight segments join computed samples. The common reference is the reference emulator fit chi2=%.8f. Squares and diamonds are direct RT evaluations of full fixed configurations, with t0 fitted; they are not independently minimized RT profiles. Hollow blue circles are candidates awaiting RT. The dashed line marks delta chi2=1. Fixed STARS/dust/noise assumptions apply.'%ref)
 # RT-versus-emulator comparison: keep previous out-of-sample errors even after retraining.
 names=[p['name'] for p in newplan['points']];x=np.arange(len(names));fig,axs=plt.subplots(2,1,figsize=(15,9),sharex=True,layout='constrained')
 previous=np.array([oldresults[n]['prediction_error'] for n in names]);axs[0].plot(x,previous,'s',mfc='none',color='.4',label='Previous emulator: direct RT test')
 for j,name in enumerate(names):
  if name in newresults:axs[0].plot(j,newresults[name]['chi2']-newpoints[name]['predicted_chi2'],'D',color='#c43d20',label='Updated emulator: direct RT test' if not any(n in newresults for n in names[:j]) else None)
 axs[0].axhline(0,color='.5',lw=1);axs[0].set_ylabel(r'$\chi^2_{\rm RT}-\chi^2_{\rm em}$');axs[0].legend()
 oldbase=oldresults['center']['chi2'];newbase=newresults.get('center',{}).get('chi2')
 axs[1].plot(x,[oldresults[n]['chi2']-oldbase for n in names],'s',mfc='none',color='.4')
 if newbase is not None:
  for j,n in enumerate(names):
   if n in newresults:axs[1].plot(j,newresults[n]['chi2']-newbase,'D',color='#c43d20')
 axs[1].axhline(1,color='.5',ls='--');axs[1].set_ylabel(r'$\chi^2_{\rm RT}-\chi^2_{\rm RT,centre}$');axs[1].set_xticks(x,[SHORT[n] for n in names]);save(fig,'fig02_rt_validation')
 captions.append('Fig. 2. Direct RT validation errors (top) and change from each batch\'s RT centre (bottom). Previous errors use predictions made before the 13 models were added to training. Updated markers appear only when new RT results exist. Each batch has its own centre; model configurations differ between batches.')
 # Best actually calculated response only; observational uncertainties always shown.
 candidates=[('old',r) for r in oldresults.values()]+[('new',r) for r in newresults.values()];which,best=min(candidates,key=lambda q:q[1]['chi2']);key=best['id']
 if which=='old':c=np.genfromtxt(io.BytesIO(oldzip.read('models/'+key+'/response.csv')),delimiter=',',names=True)
 else:c=np.genfromtxt(HERE.parent/'results/v48_paper_final/models/fine'/key/'response.csv',delimiter=',',names=True)
 obs=np.load(HERE/'observations.npy');fig,axs=plt.subplots(2,2,figsize=(15,9),sharex='col',gridspec_kw={'height_ratios':[3,1]},layout='constrained');t0=best['fit']['t0_mjd']
 for j,(band,color) in enumerate([('W1','#1864ab'),('W2','#c43d20')]):
  host=obs['baseline_'+band+'_mJy'][0];xobs=obs['mjd']-57000;yobs=obs['F_'+band+'_Jy']*1000;err=obs['sigma_'+band+'_Jy']*1000;model=np.interp(obs['mjd']-t0,c['time_obs_days'],c[band+'_Jy'],left=0,right=0)*1000+host
  tx=c['time_obs_days']+t0-57000;use=(tx>=min(xobs)-100)&(tx<=max(xobs)+100)
  axs[0,j].plot(tx[use],c[band+'_Jy'][use]*1000+host,color=color,lw=2,label='Direct RT model');axs[0,j].errorbar(xobs,yobs,yerr=err,fmt='o',ms=5,color='black',capsize=3,lw=1,label='WISE observations');axs[0,j].set(title=band,ylabel='Flux density [mJy]');axs[0,j].legend()
  axs[1,j].errorbar(xobs,(yobs-model)/err,yerr=np.ones(len(err)),fmt='o',ms=4,color=color,capsize=2);axs[1,j].axhline(0,color='.4',lw=1);axs[1,j].set(xlabel='MJD − 57000',ylabel=r'$(F_{\rm obs}-F_{\rm mod})/\sigma$')
 save(fig,'fig03_lightcurves_residuals');captions.append('Fig. 3. Best available direct RT response (%s; chi2=%.8f), including the fixed host fluxes. Error bars use exactly the input likelihood uncertainties; standardized residuals have unit error bars. The figure does not show an uncertainty envelope or emulator predictions.'%(key,best['chi2']))
 # Optimizer checks: actual completed starts, not fabricated iteration histories.
 entries=[('centre',profile['baseline'])]+[(SHORT[n],profile['sides'][n]['endpoint_details']) for n in names[1:]];fig,ax=plt.subplots(figsize=(15,5),layout='constrained')
 for j,(label,d) in enumerate(entries):
  for s,value in enumerate(d['multistart_chi2']):ax.scatter(j+(s-1)*.13,max(value-min(d['multistart_chi2']),1e-10),s=40,color=['#1864ab','#c43d20','#268b72'][s])
 ax.set_yscale('log');ax.set_xticks(range(len(entries)),[a[0] for a in entries]);ax.set_ylabel(r'$\chi^2_{\rm start}-\min(\chi^2_{\rm starts})$');save(fig,'fig04_multistart_consistency');captions.append('Fig. 4. Objective differences among three completed emulator minimizations at the centre and each endpoint. Values below 1e-10 are displayed at 1e-10. This is a multistart check, not a record of optimizer iterations or RT-grid evolution. Additional eight-start tests for i-minus and tau_visc-plus are stored in updated_profiles.json.')
 # Export the exact plotted RT comparisons.
 with (dest/'rt_validation_points.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['batch','name','model_id','chi2_RT','chi2_prediction','prediction_error'])
  for batch,rr,pp in [('previous',oldresults,oldpoints),('updated',newresults,newpoints)]:
   for n,d in rr.items():w.writerow([batch,n,d['id'],d['chi2'],pp[n]['predicted_chi2'],d['chi2']-pp[n]['predicted_chi2']])
 (dest/'CAPTIONS.txt').write_text('\n\n'.join(captions),encoding='utf-8');print('FIGURES',dest,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(HERE.parent/'results/v48_profile_final_check'));a=ap.parse_args();make_figures(a.output)
