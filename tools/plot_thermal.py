"""Post-process archived V4.8 thermal solution. No fitting or RT calculation.
Requires numpy and matplotlib. See README_UA.md for physical conventions.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
PC_DAYS = 3.085677581491367e16 / 299792458. / 86400.

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--times',type=float,nargs='+',default=[250,750,2000])
    args=ap.parse_args(); root=args.folder; out=args.out or root/'thermal_publication_v2'
    out.mkdir(parents=True,exist_ok=True)
    s=json.loads((root/'summary.json').read_text()); cfg=s['config']
    with np.load(root/'thermal_solution.npz',allow_pickle=False) as z:
        u=z['time']; T=z['temperature']; N=z['numbers']; edges=z['radial_edges']; cr=z['cell_radial']
    pts=np.genfromtxt(root/'thermal_snapshots_3d.csv',delimiter=',',names=True)
    xyz=np.array([pts[k] for k in ['x_pc','y_pc','z_pc']]).T
    cells=pts['cell'].astype(int); radii=np.linalg.norm(xyz,axis=1)
    mean=np.sum(T*N[None,:,:],axis=2)/N.sum(axis=1)[None,:]
    assert np.all(np.diff(u)>0) and np.all(N.sum(axis=1)>0)
    assert cells.min()>=0 and cells.max()<T.shape[1]
    def sample(t):
        # Interpolate the stored retarded-time solution at each spatial point.
        return np.array([np.interp(t-r*PC_DAYS,u,mean[:,c],left=np.nan,right=np.nan)
                         for r,c in zip(radii,cells)])
    temps=[sample(t) for t in args.times]
    # An opaque reference colour bar; scatter alpha is a separate display choice.
    cmap=LinearSegmentedColormap.from_list('thermal',['#c9e1ef','#74add1','#6855a2','#be4776','#ee853c','#f5c24b','#fff1a8'])
    vmax=float(np.ceil(mean.max()/100)*100); norm=Normalize(0,vmax)
    plt.rcParams.update({'font.family':'serif','font.size':18,'axes.labelsize':18,
        'axes.titlesize':18,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18,
        'pdf.fonttype':42,'ps.fonttype':42})
    n=len(args.times); fig=plt.figure(figsize=(5.2*n,11.7))
    R=float(cfg['extended'][1]); Rc=float(cfg['compact'][1])
    audit=[]
    for row,limit in enumerate([R,Rc]):
        for j,(t,values) in enumerate(zip(args.times,temps)):
            ax=fig.add_subplot(2,n,row*n+j+1,projection='3d')
            selected=(radii<=limit+1e-9)
            valid=selected&np.isfinite(values); unknown=selected&~np.isfinite(values)
            v=values[valid]; rgba=cmap(norm(v))
            rgba[:,3]=.015+.885*np.clip(v/800.,0,1)**1.8
            ax.scatter(*xyz[valid].T,c=rgba,s=9 if row else 5,edgecolors='none',depthshade=False,rasterized=True)
            # Missing times have NO assigned temperature, only faint neutral context.
            if unknown.any(): ax.scatter(*xyz[unknown].T,color=(.6,.65,.7,.018),s=5,edgecolors='none',depthshade=False,rasterized=True)
            theta=np.deg2rad(cfg['theta'])
            for radius in (cfg['compact'][:2] if row else cfg['extended'][:2]):
                phi=np.linspace(0,2*np.pi,181)
                for sign in (-1,1):
                    ax.plot(radius*np.sin(theta)*np.cos(phi),radius*np.sin(theta)*np.sin(phi),
                        np.full(phi.shape,sign*radius*np.cos(theta)),color='.65',lw=.6,alpha=.6)
            ax.scatter([0],[0],[0],marker='+',c='.3',s=35,linewidths=.8)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),
                xlabel='x (pc)',ylabel='y (pc)',zlabel='z (pc)')
            ax.set_title(f'({chr(97+row*n+j)})  '+rf'$t_{{\rm local}}={t:g}$ d',pad=10)
            ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho');ax.view_init(elev=25,azim=-55)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.set_ticks([-limit,0,limit]);axis.labelpad=13
                axis.pane.fill=False;axis.pane.set_edgecolor('white')
            ax.grid(False);ax.yaxis.set_ticks([0,limit])
            if row==0:
                audit.append({'time_local_days':t,'unsupported_points':int(unknown.sum()),
                              'supported_points':int(valid.sum()),'T_mean_min_K':float(v.min()),'T_mean_max_K':float(v.max())})
    fig.subplots_adjust(left=.02,right=.95,top=.96,bottom=.16,hspace=.15,wspace=.03)
    cax=fig.add_axes([.25,.065,.5,.022])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',
        label=r'Grain-number-weighted temperature $\langle T\rangle_N$ (K)')
    fig.savefig(out/'thermal_evolution_3d_transparent.pdf',dpi=450)
    fig.savefig(out/'thermal_evolution_3d_transparent.png',dpi=170);plt.close(fig)
    # Temporal profiles: preserve a gap before the stored time support, never add 0 K.
    rmean=.75*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
    used=np.unique(cr); chosen=[used[0],used[len(used)//2],used[-1]]
    times=np.linspace(0,5000,501);fig,ax=plt.subplots(figsize=(10,6.2),layout='constrained'); rows=[]
    for ir,col in zip(chosen,['#c44e52','#2878b5','#30976b']):
        cc=np.flatnonzero(cr==ir)
        series=np.sum(T[:,cc,:]*N[None,cc,:],axis=(1,2))/N[cc,:].sum()
        vals=np.interp(times-rmean[ir]*PC_DAYS,u,series,left=np.nan,right=np.nan)
        ax.plot(times,vals,color=col,lw=2,label=f'r = {rmean[ir]:.3f} pc')
        rows.extend(zip(times,np.full(times.shape,rmean[ir]),vals))
    ax.set(xlabel='Local rest-frame time from fallback peak (d)',ylabel=r'$\langle T\rangle_N$ (K)',xlim=(0,5000),ylim=(0,vmax))
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'temperature_time_support_corrected.pdf')
    fig.savefig(out/'temperature_time_support_corrected.png',dpi=170);plt.close(fig)
    np.savetxt(out/'temperature_time_support_corrected.csv',rows,delimiter=',',
        header='t_local_rest_days,r_pc,T_number_mean_K',comments='')
    payload=np.column_stack([xyz,cells]+temps)
    np.savetxt(out/'thermal_snapshots_support_corrected.csv',payload,delimiter=',',
        header=','.join(['x_pc','y_pc','z_pc','cell']+[f'T_number_mean_{t:g}d_K' for t in args.times]),comments='')
    manifest={'source':str(root.resolve()),'sha256_thermal_solution':hashlib.sha256((root/'thermal_solution.npz').read_bytes()).hexdigest(),
        'temperature':'grain-number-weighted equilibrium temperature, not colour temperature',
        'time':'local rest-frame time relative to unsmoothed fallback peak; u=t_local-r/c',
        'outside_time_support':'NaN, neutral geometry only; never interpreted as 0 K',
        'alpha':'0.015 + 0.885 * clip(T/800,0,1)**1.8; visual only, not physical transmission',
        'panels':'upper: full system; lower: compact population zoom; all share colour scale',
        'background_heating':'absent in saved solver; near-zero values are not a prediction of actual ambient dust temperature',
        'snapshots':audit,'font_pt':18,'physical_fit_changed':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
