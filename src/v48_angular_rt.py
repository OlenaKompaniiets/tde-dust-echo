"""Time-dependent absorption/re-emission RT in an axisymmetric effective torus.

3-D straight ray paths, finite-volume (r,mu) thermal cells, stationary opacity.
Absorption, secondary heating and observer attenuation share the same density
and cross sections. This is NOT a resolved-clump model or a scattering solver.
Static opacity is admissible ONLY while every grain remains below Tsub.
Quasi-Monte-Carlo spatial/angular quadrature and cell/time discretization need
resolution tests. No empirical lag, band amplitude or fitted temperature.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import exprel
from scipy.stats import qmc
from scipy.fft import rfft, irfft, next_fast_len
from src.v45_direct_absorption import quadrature_weights
from src.v4_thermal import H, C, K_B

PC=3.085677581491367e16
DAY=86400.
PC_DAYS=PC/C/DAY


@dataclass
class Mesh:
    radial_edges: np.ndarray
    mu_edges: np.ndarray
    occupied: np.ndarray
    radial_density: np.ndarray  # grain number / m^3, summed over sizes
    cell_radial: np.ndarray
    cell_mu: np.ndarray
    cell_volume: np.ndarray
    numbers: np.ndarray       # cell, grain size
    lookup: np.ndarray       # radial bin, mu bin -> thermal cell (-1 vacuum)
    @property
    def nc(self): return len(self.cell_radial)
    @property
    def radius(self): return self.radial_edges[-1]


def build_mesh(compact,extended,tau_c,tau_e,theta,nr,nmu,wa,sigma_source):
    if nr<1 or nmu<1 or min(tau_c,tau_e)<0 or sigma_source<=0:
        raise ValueError('Invalid mesh or opacity normalization')
    if not 0<compact[0]<compact[1]<extended[0]<extended[1]:
        raise ValueError('Disjoint positive compact and extended radial domains required')
    f=np.cos(np.deg2rad(theta))
    if not 0<f<=1:raise ValueError('Invalid opening angle')
    cedges=np.geomspace(compact[0],compact[1],nr+1)
    eedges=np.geomspace(extended[0],extended[1],nr+1)
    edges=np.unique(np.r_[0.,cedges,eedges])
    mu=np.linspace(-f,f,nmu+1); density=np.zeros(len(edges)-1)
    for params,depth in ((compact,tau_c),(extended,tau_e)):
        rin,rout,p=params
        select=(edges[:-1]>=rin*(1-1e-12)) & (edges[1:]<=rout*(1+1e-12))
        lo,hi=edges[:-1][select],edges[1:][select]
        integral=np.log(hi/lo) if p==3 else (hi**(3-p)-lo**(3-p))/(3-p)
        shape=3*integral/(hi**3-lo**3)
        norm=depth/(np.sum(shape*(hi-lo)*PC)*sigma_source)
        density[select]=norm*shape
    occupied=density>0; cell_r=[];cell_m=[];vol=[];numbers=[]
    lookup=np.full((len(density),nmu),-1,int)
    for ir in np.flatnonzero(occupied):
        for im in range(nmu):
            lookup[ir,im]=len(cell_r);cell_r.append(ir);cell_m.append(im)
            v=2*np.pi*(mu[im+1]-mu[im])*(edges[ir+1]**3-edges[ir]**3)/3*PC**3
            vol.append(v);numbers.append(v*density[ir]*wa)
    return Mesh(edges,mu,occupied,density,np.array(cell_r),np.array(cell_m),
                np.array(vol),np.array(numbers),lookup)


def sample_positions(mesh,ic,u,with_phi=True):
    ir,im=mesh.cell_radial[ic],mesh.cell_mu[ic]
    lo,hi=mesh.radial_edges[ir:ir+2]
    r=(lo**3+u[:,0]*(hi**3-lo**3))**(1/3)
    mu=mesh.mu_edges[im]+u[:,1]*np.diff(mesh.mu_edges)[im]
    phi=2*np.pi*u[:,2] if with_phi else np.zeros(len(u))
    transverse=r*np.sqrt(np.maximum(0,1-mu**2))
    return np.column_stack((transverse*np.cos(phi),transverse*np.sin(phi),r*mu))


def path_segments(mesh,x,n):
    """Exact spherical/conical boundary intersections, distances in pc."""
    x=np.asarray(x,float);n=np.asarray(n,float)
    if not np.isclose(n@n,1.,atol=1e-10):raise ValueError('Unit ray required')
    b=float(x@n);rr=float(x@x)
    end=-b+np.sqrt(max(0.,b*b+mesh.radius**2-rr))
    cuts=[0.,end]
    for rad in mesh.radial_edges[1:]:
        discr=b*b+rad*rad-rr
        if discr>=0:
            root=np.sqrt(discr)
            for s in (-b-root,-b+root):
                if 1e-11<s<end-1e-11:cuts.append(s)
    # Squared conical equation includes both branches; midpoint lookup chooses
    # the actual +/- mu cell. Extra valid cuts do not change optical depth.
    for mu in np.unique(np.abs(mesh.mu_edges)):
        a=n[2]**2-mu**2;bb=2*(x[2]*n[2]-mu**2*b);cc=x[2]**2-mu**2*rr
        if abs(a)<1e-14:
            roots=[] if abs(bb)<1e-14 else [-cc/bb]
        else:
            disc=bb*bb-4*a*cc
            roots=[] if disc<0 else [(-bb-np.sqrt(disc))/(2*a),(-bb+np.sqrt(disc))/(2*a)]
        for s in roots:
            if 1e-11<s<end-1e-11:cuts.append(s)
    cuts=np.array(sorted(cuts));cuts=cuts[np.r_[True,np.diff(cuts)>1e-10]]
    segments=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        p=x+(left+right)/2*n;r=np.linalg.norm(p)
        if r<=0:continue
        mu=p[2]/r
        ir=np.searchsorted(mesh.radial_edges,r,side='right')-1
        im=np.searchsorted(mesh.mu_edges,mu,side='right')-1
        if 0<=ir<len(mesh.radial_density) and 0<=im<len(mesh.mu_edges)-1:
            cell=mesh.lookup[ir,im]
            if cell>=0:segments.append((int(cell),float(left),float(right)))
    return segments,float(end)


def mean_absorption_fraction(tau):
    """Mean distance into uniform segment / length, conditional on absorption."""
    tau=np.asarray(tau,float)
    result=np.empty_like(tau);small=tau<1e-3;large=tau>50;mid=~(small|large)
    result[small]=.5-tau[small]/12+tau[small]**3/720
    result[large]=1/tau[large]
    result[mid]=1/tau[mid]-1/np.expm1(tau[mid])
    return result


def deposit_last_axis(target,delay,weight,dt):
    """target wavelength x lag; nonnegative two-bin deposition."""
    x=np.broadcast_to(np.asarray(delay,float)/dt,np.shape(weight))
    lo=np.floor(x).astype(int);frac=x-lo
    if lo.min()<0 or (lo+1).max()>=target.shape[-1]:raise ValueError('Delay outside allocated kernel')
    wave=np.arange(len(weight))
    np.add.at(target,(wave,lo),weight*(1-frac))
    np.add.at(target,(wave,lo+1),weight*frac)


def build_transport(mesh,sigma,dt,nrays,seed=481,progress=print):
    """K[lambda,destination,source,lag], probability of absorption per packet.

    All cells use u=t_local-|x|/c. For a path x->y of length s,
    delta_u=(|x|+s-|y|)/c >=0. Thus finite radial cells never create an
    acausal direct-light front by instantaneously mixing different radii.
    Emission locations uniform in each constant-density cell; directions
    isotropic. Escape probabilities are obtained from the SAME rays.
    """
    if nrays<8 or nrays&(nrays-1):raise ValueError('Ray count must be a power of two >=8')
    nw=len(sigma);nc=mesh.nc;nk=int(np.ceil(2*mesh.radius*PC_DAYS/dt))+3
    K=np.zeros((nw,nc,nc,nk),dtype=np.float64)
    E=np.zeros((nw,nc,nk));prob_error=0.
    for src in range(nc):
        u=qmc.Sobol(4,scramble=True,seed=seed+src).random_base2(int(np.log2(nrays)))
        positions=sample_positions(mesh,src,u,with_phi=False)
        mu=2*u[:,2]-1;phi=2*np.pi*u[:,3]
        directions=np.column_stack((np.sqrt(1-mu*mu)*np.cos(phi),np.sqrt(1-mu*mu)*np.sin(phi),mu))
        for x,n in zip(positions,directions):
            segments,end=path_segments(mesh,x,n);survive=np.ones(nw)
            for dest,left,right in segments:
                ir=mesh.cell_radial[dest]
                tau=mesh.radial_density[ir]*(right-left)*PC*sigma
                absorbed=survive*(-np.expm1(-tau))
                travel=left+(right-left)*mean_absorption_fraction(tau)
                destination_radius=np.linalg.norm(x[None,:]+travel[:,None]*n[None,:],axis=1)
                delay=np.maximum(0.,np.linalg.norm(x)+travel-destination_radius)*PC_DAYS
                deposit_last_axis(K[:,dest,src,:],delay,absorbed/nrays,dt)
                survive*=np.exp(-tau)
            delay=max(0.,np.linalg.norm(x)+end-mesh.radius)*PC_DAYS
            deposit_last_axis(E[:,src,:],delay,survive/nrays,dt)
        total=K[:,:,src,:].sum(axis=(1,2))+E[:,src,:].sum(axis=1)
        prob_error=max(prob_error,float(np.max(np.abs(total-1))))
        if progress and (src%8==0 or src==nc-1):progress(f'ray transport {src+1}/{nc} cells',flush=True)
    if prob_error>1e-11:raise RuntimeError(f'Photon probability closure failed: {prob_error}')
    return K,E,prob_error


def direct_kernels(mesh,sigma,source_fraction,dt):
    nw=len(sigma);nk=int(np.ceil(mesh.radius*PC_DAYS/dt))+3
    D=np.zeros((nw,mesh.nc,nk));survive=np.ones(nw)
    for ir in np.flatnonzero(mesh.occupied):
        lo,hi=mesh.radial_edges[ir:ir+2]
        tau=mesh.radial_density[ir]*(hi-lo)*PC*sigma
        absorbed=survive*(-np.expm1(-tau))*source_fraction
        delay=np.zeros(nw)  # radial direct propagation leaves source phase u unchanged
        for im in range(len(mesh.mu_edges)-1):
            cell=mesh.lookup[ir,im]
            deposit_last_axis(D[:,cell,:],delay,absorbed*np.diff(mesh.mu_edges)[im]/2,dt)
        survive*=np.exp(-tau)
    f=(mesh.mu_edges[-1]-mesh.mu_edges[0])/2
    direct_escape_fraction=float(np.sum(source_fraction*((1-f)+f*survive)))
    if abs(D.sum()+direct_escape_fraction-1)>1e-11:raise RuntimeError('Direct photon conservation')
    return D,direct_escape_fraction


def observer_columns(mesh,inclination,npoints,seed=9481):
    """Peel-off paths; same opacity geometry as the heating rays.

    Return number columns and delays (|x|-n_obs dot x) / c for every source cell.
    These paths continue through inward-facing/opposite-side dust as needed.
    """
    if npoints<8 or npoints&(npoints-1):raise ValueError('Observer points must be power of two')
    inc=np.deg2rad(inclination);n=np.array([np.sin(inc),0.,np.cos(inc)])
    columns=np.zeros((mesh.nc,npoints));delays=np.zeros_like(columns)
    for src in range(mesh.nc):
        u=qmc.Sobol(3,scramble=True,seed=seed+src).random_base2(int(np.log2(npoints)))
        for j,x in enumerate(sample_positions(mesh,src,u)):
            segments,_=path_segments(mesh,x,n)
            columns[src,j]=sum(mesh.radial_density[mesh.cell_radial[d]]*(b-a)*PC for d,a,b in segments)
            delays[src,j]=(np.linalg.norm(x)-float(n@x))*PC_DAYS
    return columns,delays


def make_observer_kernel(columns,delays,sigma,dt,radius):
    offset=0.
    nk=int(np.ceil(2*radius*PC_DAYS/dt))+3
    O=np.zeros((len(sigma),columns.shape[0],nk))
    for src in range(columns.shape[0]):
        for col,delay in zip(columns[src],delays[src]):
            deposit_last_axis(O[:,src,:],delay+offset,np.exp(-col*sigma)/columns.shape[1],dt)
    return O,offset


class Thermal:
    """Individual grain radiative equilibrium on the RT spectral quadrature.

    Absorbed spectral power is partitioned by grain absorption cross-section.
    Newton correction enforces Planck equilibrium before emission. No fitted
    or post-hoc luminosity rescaling. Cross sections SI; all bin powers W.
    """
    def __init__(self,wavelength,cross,wa,numbers,Tsub):
        self.lam=np.asarray(wavelength);self.cross=np.asarray(cross)
        self.w=quadrature_weights(self.lam);self.wa=wa;self.numbers=numbers
        self.sigma=wa@cross;self.Tsub=float(Tsub)
        self.share=(wa[:,None]*cross/np.maximum(self.sigma,1e-300)[None,:]).T
        self.grid=np.geomspace(.5,max(5000.,Tsub*2),2400)
        self.tables=[]
        for a in range(len(wa)):
            spec=self._grain(self.grid,a)
            self.tables.append(spec.sum(axis=1))
        self.tables=np.array(self.tables)

    def _grain(self,T,a,derivative=False):
        T=np.asarray(T,float)
        x=H*C/(self.lam[None,:]*K_B*np.maximum(T[:,None],1e-30))
        B=(2*H*C*C/self.lam**5)[None,:]/np.expm1(np.minimum(x,700))
        B[x>700]=0.
        spec=4*np.pi*self.cross[a][None,:]*B*self.w[None,:]
        if derivative:
            factor=np.divide(x,-np.expm1(-np.minimum(x,700)))/np.maximum(T[:,None],1e-30)
            return spec,spec*factor
        return spec

    def emit(self,absorbed):
        nt,nc,nw=absorbed.shape
        absorbed=np.maximum(absorbed,0.)
        heating=absorbed@self.share
        emission=np.zeros_like(absorbed);temps=np.zeros((nt,nc,len(self.wa)))
        max_error=0.
        for a in range(len(self.wa)):
            power=(heating[:,:,a]/self.numbers[None,:,a]).reshape(-1)
            good=power>self.tables[a,0]
            temp=np.zeros(len(power));positive=self.tables[a]>0
            temp[good]=np.exp(np.interp(np.log(power[good]),np.log(self.tables[a,positive]),np.log(self.grid[positive])))
            for _ in range(2):
                sp,der=self._grain(temp,a,True)
                step=np.divide(sp.sum(axis=1)-power,der.sum(axis=1),out=np.zeros_like(power),where=good)
                temp[good]-=step[good]
            if temp.max()>=self.Tsub:
                raise RuntimeError(f'SUBLIMATION: grain {a}, T={temp.max():.3f} >= {self.Tsub:.3f} K. Static-opacity RT invalid; do not interpret this run.')
            sp=self._grain(temp,a);sp[~good]=0.
            reconstructed=sp.sum(axis=1)
            scale=max(float(power.max()),1e-300)
            max_error=max(max_error,float(np.max(np.abs(reconstructed-power))/scale))
            temps[:,:,a]=temp.reshape(nt,nc)
            emission+=sp.reshape(nt,nc,nw)*self.numbers[None,:,a,None]
        if max_error>2e-7:raise RuntimeError(f'Thermal Planck closure failed: {max_error}')
        return emission,temps,max_error


def solve_time_rt(L,D,K,E,thermal,dt,max_iterations=60,tolerance=1e-6,progress=print):
    """Nonlinear causal Lambda iteration over all times, including zero-lag self absorption.

    Linear deposition approximates within-cell flight-time distributions;
    positivity and finite-window radiation energy are checked explicitly.
    """
    nt=len(L);nf=next_fast_len(nt+max(D.shape[-1],K.shape[-1])-1)
    df=rfft(D,n=nf,axis=-1)
    direct=irfft(df*rfft(L,n=nf)[None,None,:],n=nf,axis=-1)[...,:nt].transpose(2,1,0)
    direct=np.maximum(direct,0.)
    del df
    kf=rfft(K.astype(np.float32),n=nf,axis=-1).transpose(0,3,1,2).copy()
    history=[];absorbed=direct.copy();previous=None
    nonzero=np.flatnonzero(L>0)
    first_source=int(nonzero[0]) if len(nonzero) else nt
    for iteration in range(max_iterations):
        emitted,T,thermal_error=thermal.emit(absorbed)
        power_scale=max(float(emitted.max()),1e-300)
        ef=rfft((emitted/power_scale).astype(np.float32).transpose(2,1,0),n=nf,axis=-1).transpose(0,2,1)
        absorbed_fft=np.einsum('wfij,wfj->wfi',kf,ef,optimize=True)
        secondary=irfft(absorbed_fft.transpose(0,2,1),n=nf,axis=-1)[...,:nt].transpose(2,1,0).astype(float)*power_scale
        secondary=np.maximum(secondary,0.)
        secondary[:first_source]=0.  # eliminate FFT roundoff before causal support
        new_absorbed=direct+secondary
        error=float(np.max(np.abs(new_absorbed-absorbed))/max(float(new_absorbed.max()),1e-300))
        energy_error=float(np.sum(np.abs(new_absorbed-absorbed))/max(float(np.sum(new_absorbed)),1e-300))
        history.append(dict(iteration=iteration+1,max_spectral_relative=error,L1_relative=energy_error,
                            max_temperature_K=float(T.max()),thermal_relative=thermal_error))
        if progress:progress(f'RT iteration {iteration+1}: residual={error:.3g}; energy L1={energy_error:.3g}; Tmax={T.max():.1f} K',flush=True)
        if error<tolerance and energy_error<tolerance:
            break
        absorbed=new_absorbed
    else:raise RuntimeError('Nonlinear RT failed to converge; no valid fit produced')
    # Finite-window energy: all eventual escapes + future absorption beyond the
    # modeled time domain. The latter is radiation still in flight, not lost.
    escape_fraction=E.sum(axis=-1).T
    direct_total=float(L.sum()*D.sum())
    direct_in=float(direct.sum())
    ir_total=float((emitted*K.sum(axis=(1,3)).T[None,:,:]).sum())
    ir_in=float(secondary.sum())
    escaped_ir=float((emitted*escape_fraction[None,:,:]).sum())
    pending_direct=direct_total-direct_in;pending_ir=ir_total-ir_in
    pending_roundoff=min(0.,pending_direct)+min(0.,pending_ir)
    if pending_roundoff < -2e-6*max(direct_total,1e-300):
        raise RuntimeError('Convolution created more absorbed energy than available')
    closure=(escaped_ir+pending_direct+pending_ir-direct_total)
    # At convergence emitted equals direct+secondary within the recorded residual.
    closure_relative=closure/max(direct_total,1e-300)
    if abs(closure_relative)>max(5*tolerance,2e-6):raise RuntimeError(f'RT global budget mismatch {closure_relative}')
    tail_fraction=float(emitted[-min(10,nt):].sum()/max(emitted.sum(),1e-300))
    energy=dict(E_direct_absorbed_all_arrivals_erg=direct_total*dt*DAY*1e7,
        E_IR_eventual_escape_from_computed_emission_erg=escaped_ir*dt*DAY*1e7,
        E_direct_pending_absorption_erg=max(0.,pending_direct)*dt*DAY*1e7,
        E_IR_pending_absorption_erg=max(0.,pending_ir)*dt*DAY*1e7,
        convolution_pending_roundoff_erg=pending_roundoff*dt*DAY*1e7,
        E_intrinsic_emission_erg=float(emitted.sum()*dt*DAY*1e7),
        E_secondary_absorbed_in_window_erg=ir_in*dt*DAY*1e7,
        global_budget_relative=closure_relative,last_10_steps_emission_fraction=tail_fraction)
    if tail_fraction>.001:raise RuntimeError(f'Insufficient time window: tail fraction {tail_fraction}')
    return dict(emitted=emitted,T=T,direct_absorbed=direct,secondary_absorbed=secondary,
                energy=energy,iterations=history)


def observer_spectrum(T,thermal,opacity,grains,lambda_observer_m,redshift,
                      columns,delays,dt,radius):
    """Directional isotropic-equivalent L_lambda on native observer RSR nodes.

    The spectral luminosity is evaluated at rest wavelengths; cosmological
    flux factors and WISE calibration are applied by the caller.
    """
    lam=np.asarray(lambda_observer_m)/(1+redshift)
    cross=np.pi*(grains[:,None]*1e-6)**2*opacity.q_abs(lam[None,:]*1e6,grains[:,None])
    sigma=thermal.wa@cross
    O,offset=make_observer_kernel(columns,delays,sigma,dt,radius)
    nt,nc,na=T.shape;spec=np.zeros((nt,nc,len(lam)))
    for a in range(na):
        temp=T[:,:,a]
        x=H*C/(lam[None,None,:]*K_B*np.maximum(temp[:,:,None],1e-30))
        B=(2*H*C*C/lam**5)[None,None,:]/np.expm1(np.minimum(x,700));B[x>700]=0
        spec+=4*np.pi*thermal.numbers[None,:,a,None]*cross[a][None,None,:]*B
    nf=next_fast_len(nt+O.shape[-1]-1)
    sf=rfft(spec.transpose(2,1,0),n=nf,axis=-1)
    of=rfft(O,n=nf,axis=-1)
    result=irfft(np.sum(sf*of,axis=1),n=nf,axis=-1)[:,:nt+O.shape[-1]-1].T
    result=np.maximum(result,0.)
    return result,offset
