"""All large-run figure types, from saved physical results; no RT computation.
18 pt original text -> 9 pt when the figure is placed at 50% of its PDF width.
"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.colors import Normalize
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.size':18,'axes.labelsize':18,'axes.titlesize':18,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1})
def table(p):return np.genfromtxt(p,delimiter=',',names=True)
def save(fig,out,name):
 for t in fig.findobj(match=Text):t.set_fontsize(18)
 for ax in fig.axes:
  ax.tick_params(labelsize=18)
 fig.savefig(out/(name+'.pdf'))
 fig.savefig(out/(name+'.png'),dpi=300)
 plt.close(fig)
def all_plots(folder):
 folder=Path(folder);out=folder/'publication_50pct';out.mkdir(exist_ok=True)
 s=json.loads((folder/'summary.json').read_text());cfg=s['config'];e=table(folder/'epochs.csv');d=table(folder/'dense.csv')
 fig,axes=plt.subplots(2,2,figsize=(14,9),sharex='col',layout='constrained',gridspec_kw={'height_ratios':[3,1]})
 for j,b in enumerate(('W1','W2')):
  axes[0,j].errorbar(e['mjd'],e[b+'_data_mJy'],yerr=e[b+'_sigma_mJy'],fmt='o',ms=5,c='k',capsize=3,label='WISE')
  axes[0,j].plot(d['mjd'],d[b+'_model_mJy'],c='#0072B2',label='Model');axes[0,j].set(title=b,ylabel='Flux density (mJy)');axes[0,j].legend(frameon=False)
  axes[1,j].plot(e['mjd'],e[b+'_residual_sigma'],'o',c='#0072B2');axes[1,j].axhline(0,c='.5',lw=1);axes[1,j].set(xlabel='MJD',ylabel=r'Residual / $\sigma$')
 save(fig,out,'angular_rt_lightcurves')
 # Host is constant and taken from the pre-event archived dense model plateau.
 # Verify against the first two baseline model epochs before using this inference.
 host=np.array([d[b+'_model_mJy'][0] for b in ('W1','W2')])
 if not all(np.allclose(e[b+'_model_mJy'][:2],host[i],rtol=0,atol=1e-6) for i,b in enumerate(('W1','W2'))):
  raise ValueError('Cannot recover fixed host unambiguously from baseline; provide explicit host metadata.')
 fig,ax=plt.subplots(figsize=(7,7),layout='constrained')
 x=e['W1_data_mJy']-host[0];y=e['W2_data_mJy']-host[1]
 ax.errorbar(x,y,xerr=e['W1_sigma_mJy'],yerr=e['W2_sigma_mJy'],fmt='none',ecolor='.55',capsize=2)
 sc=ax.scatter(x,y,c=e['mjd'],s=45,cmap='viridis',zorder=3)
 ax.plot(d['W1_model_mJy']-host[0],d['W2_model_mJy']-host[1],c='#D55E00',label='Model');ax.set(xlabel='W1 excess (mJy)',ylabel='W2 excess (mJy)');ax.legend(frameon=False);fig.colorbar(sc,ax=ax,label='MJD',shrink=.8)
 save(fig,out,'paper_flux_flux_errors')
 fig,ax=plt.subplots(figsize=(7,5.7),layout='constrained')
 color=lambda x,y:-2.5*np.log10((x/309.540)/(y/171.787))
 sig=2.5/np.log(10)*np.sqrt((e['W1_sigma_mJy']/e['W1_data_mJy'])**2+(e['W2_sigma_mJy']/e['W2_data_mJy'])**2)
 ax.errorbar(e['mjd'],color(e['W1_data_mJy'],e['W2_data_mJy']),yerr=sig,fmt='o',ms=5,c='k',capsize=3,label='WISE')
 ax.plot(d['mjd'],color(d['W1_model_mJy'],d['W2_model_mJy']),c='#0072B2',label='Model');ax.set(xlabel='MJD',ylabel='W1 − W2 (Vega mag)');ax.legend(frameon=False);save(fig,out,'paper_colour_evolution')
 missing=[]
 path=folder/'temperature_radial_evolution.csv'
 if path.exists():
  a=table(path);sizes=np.unique(a['a_micron']);times=np.unique(a['t_local_rest_days']);fig,axes=plt.subplots(1,len(sizes),figsize=(14,5.8),sharey=True,layout='constrained',squeeze=False)
  for ax,size in zip(axes[0],sizes):
   for color,t in zip(['#0072B2','#D55E00','#009E73'],times):
    rows=a[(a['a_micron']==size)&(a['t_local_rest_days']==t)]
    for pop,mask in enumerate([rows['r_pc']<cfg['compact'][1],rows['r_pc']>cfg['extended'][0]]):
     q=rows[mask];ax.plot(q['r_pc'],q['T_K'],'o-',ms=4,c=color,label=f'{t:g} d' if pop==0 else None)
   ax.axhline(s['sublimation_threshold_K'],ls=':',c='.5');ax.set(xlabel='Radius (pc)',title=f'a = {size:.3g} μm');ax.legend(frameon=False)
  axes[0,0].set_ylabel('Mean temperature (K)');save(fig,out,'paper_temperature_radial_evolution')
 else:missing.append(path.name)
 path=folder/'temperature_time.csv'
 if path.exists():
  a=table(path);fig,ax=plt.subplots(figsize=(7,5.7),layout='constrained')
  for r in np.unique(a['r_pc']):
   q=a[a['r_pc']==r];ax.plot(q['t_local_rest_days'],q['T_number_mean_K'],label=f'{r:.3f} pc')
  ax.set(xlabel='Local rest-frame time (d)',ylabel='Mean temperature (K)');ax.legend(frameon=False);save(fig,out,'paper_temperature_time')
 else:missing.append(path.name)
 path=folder/'thermal_snapshots_3d.csv'
 if path.exists():
  a=table(path);cols=[n for n in a.dtype.names if n.startswith('T_number_mean_')];vmax=max(1100.,max(float(a[k].max()) for k in cols));norm=Normalize(0,vmax);R=max(cfg['compact'][1],cfg['extended'][1])
  fig=plt.figure(figsize=(14,6.7));axes=[]
  for i,key in enumerate(cols):
   ax=fig.add_subplot(1,len(cols),i+1,projection='3d');axes.append(ax)
   sc=ax.scatter(a['x_pc'],a['y_pc'],a['z_pc'],c=a[key],cmap='inferno',norm=norm,s=4,alpha=.65,rasterized=True)
   ax.scatter([0],[0],[0],marker='*',s=65,c='cyan');ax.set(xlim=(-R,R),ylim=(-R,R),zlim=(-R,R),xlabel='x (pc)',ylabel='y (pc)',zlabel='z (pc)',title=key.split('_')[-2].replace('d',' d'))
   ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55);ax.set_proj_type("ortho")
   for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_ticks([-R,0,R]);axis.labelpad=12
   ax.yaxis.set_ticks([0,R])
  fig.subplots_adjust(left=.02,right=.91,top=.87,bottom=.26,wspace=.13)
  cax=fig.add_axes([.28,.10,.44,.035]);fig.colorbar(sc,cax=cax,orientation='horizontal',label='Number-weighted temperature (K)');save(fig,out,'thermal_evolution_3d')
 else:missing.append(path.name)
 path=folder/'thermal_solution.npz'
 if path.exists():
  with np.load(path,allow_pickle=False) as z:
   T=z['temperature'];N=z['numbers'];cr=z['cell_radial'];cm=z['cell_mu'];edges=z['radial_edges'];mu=z['mu_edges'];power=z['emitted_bin_power_W']
   phase=int(np.argmax(power.sum(axis=(1,2))));mapping=np.full((len(edges)-1,len(mu)-1),np.nan)
   for ic,(ir,im) in enumerate(zip(cr,cm)):mapping[ir,im]=np.average(T[phase,ic],weights=N[ic])
  fig,ax=plt.subplots(figsize=(7,6),layout='constrained');sc=ax.pcolormesh(edges,mu,mapping.T,shading='flat',cmap='inferno');ax.set(xlabel='Radius (pc)',ylabel=r'$\cos\theta$');fig.colorbar(sc,ax=ax,label='Temperature (K)');save(fig,out,'temperature_map')
 else:missing.append('thermal_solution.npz (required for temperature_map)')
 (out/'figure_manifest.json').write_text(json.dumps({'font_original_pt':18,'intended_scale':.5,'font_at_scale_pt':9,'source':str(folder),'figures':[q.name for q in sorted(out.glob('*.pdf'))],'missing_inputs':missing},indent=2))
 print('FIGURES',out,'MISSING',missing,flush=True)
 return missing
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folder',type=Path);a=ap.parse_args()
 for folder in ([a.folder] if a.folder else [P/'archived/local',P/'archived/grid']):all_plots(folder)
