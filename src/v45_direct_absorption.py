"""Radial finite-volume DIRECT absorption control in a uniform angular wedge.

Clock u=t_dust-r/c; no additional r/c in heating. Each cell attenuates the
remaining source spectrum. Grain temperatures use the cell-averaged absorbed
spectrum, with phi(tau)=(1-exp(-tau))/tau. No scattering, diffuse IR heating,
or IR escape attenuation; emergent MIR is therefore an optically-thin-IR
approximation. This is an effective medium, not resolved optically thick clouds.
"""
from __future__ import annotations
import numpy as np
from src.v4_thermal_lookup import lookup_temperature_from_flux


def quadrature_weights(x):
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('Increasing wavelength grid required')
    w=np.empty_like(x);w[0]=(x[1]-x[0])/2;w[-1]=(x[-1]-x[-2])/2
    w[1:-1]=(x[2:]-x[:-2])/2
    return w


def mean_transmission(tau):
    tau=np.asarray(tau,float)
    if np.any(~np.isfinite(tau)) or np.any(tau<0):raise ValueError('Invalid absorption depth')
    return np.divide(-np.expm1(-tau),tau,out=np.ones_like(tau),where=tau>0)


def solve_direct(time_days, luminosity_erg_s, wavelength_m, source_sed,
                 cross_sections_m2, qsrc, lookup, radius_pc, cell_grain_numbers,
                 solid_angle_fraction, sublimation_K, *, attenuation=True):
    """numbers shape Nr,Na; output temperatures/alive Nt,Nr,Na.

    tau inputs elsewhere refer to source-SED-mean INITIAL radial absorption
    optical depths, not covering factors or fitted MIR amplitudes.
    Instantaneous irreversible destruction, zero latent heat, no reformation.
    Sublimation is iterated within each cell before propagation to the next.
    """
    t=np.asarray(time_days,float);L=np.asarray(luminosity_erg_s,float)
    lam=np.asarray(wavelength_m,float);sed=np.asarray(source_sed,float)
    C=np.asarray(cross_sections_m2,float);qsrc=np.asarray(qsrc,float)
    r=np.asarray(radius_pc,float);numbers=np.asarray(cell_grain_numbers,float)
    f=float(solid_angle_fraction);w=quadrature_weights(lam)
    if not 0<f<=1:raise ValueError('Invalid solid angle fraction')
    if L.shape!=t.shape or np.any(~np.isfinite(L)) or np.any(L<0) or np.any(np.diff(t)<=0):raise ValueError('Invalid driver')
    if np.any(r<=0) or np.any(np.diff(r)<=0):raise ValueError('Increasing radial cells required')
    if numbers.shape!=(len(r),len(qsrc)) or C.shape!=(len(qsrc),len(lam)):raise ValueError('Shape mismatch')
    if np.any(numbers<0) or np.any(~np.isfinite(numbers)) or np.any(C<0) or np.any(qsrc<=0):raise ValueError('Invalid dust')
    if sed.shape!=lam.shape or np.any(sed<0) or not np.isclose(sed@w,1,rtol=1e-8):raise ValueError('Source SED must integrate to unity')
    # Luminosity PER UNIT WAVELENGTH within wedge, W/m.
    incoming=f*L[:,None]*1e-7*sed[None,:]
    T=np.zeros((len(t),len(r),len(qsrc)));alive=np.ones(T.shape,bool)
    absorbed=np.zeros((len(t),len(r)));grain_power=np.zeros_like(absorbed)
    initial_tau=np.zeros(len(lam)); max_cell_tau=0.;iterations=[]
    # Q = C/(pi a^2) recovered from lookup grain radii.
    area=np.pi*(np.asarray(lookup.grain_radius_micron)*1e-6)**2
    Q=C/area[:,None]
    for ir,rr in enumerate(r):
        shell_area=4*np.pi*f*(rr*3.085677581491367e16)**2
        initial_tau+=numbers[ir]@C/shell_area
        live=np.ones((len(t),len(qsrc)),bool)
        # Monotone removal; at most Nt*Na removals. Practical cases converge in 1–3 iterations.
        for it in range(100):
            tau=(live*numbers[ir][None,:])@C/shell_area
            field=incoming/shell_area
            if attenuation:field=field*mean_transmission(tau)
            absorbed_per_area=(field*w[None,:])@Q.T
            equivalent=absorbed_per_area/qsrc[None,:]
            temp=np.column_stack([lookup_temperature_from_flux(lookup,equivalent[:,ia],ia) for ia in range(len(qsrc))])
            hot=(~np.isfinite(temp))|(temp>=sublimation_K)
            destroyed=np.maximum.accumulate(hot&live,axis=0)
            new_live=live & ~destroyed
            if np.array_equal(new_live,live):break
            live=new_live
        else:raise RuntimeError('Sublimation did not converge; refine cells/time step')
        T[:,ir,:]=np.where(live,temp,0);alive[:,ir,:]=live
        if np.any(~np.isfinite(T[:,ir,:])):raise RuntimeError('Invalid surviving temperature')
        grain_power[:,ir]=np.sum(absorbed_per_area*area[None,:]*numbers[ir][None,:]*live,axis=1)
        if attenuation:
            removed=incoming*(-np.expm1(-tau))
            absorbed[:,ir]=removed@w
            incoming=incoming*np.exp(-tau)
        else:
            absorbed[:,ir]=grain_power[:,ir]
        max_cell_tau=max(max_cell_tau,float(tau.max()));iterations.append(it+1)
    escaped_direct=(1-f)*L+(incoming@w)*1e7
    budget=escaped_direct+absorbed.sum(axis=1)*1e7
    scale=max(float(L.max()),1.)
    err=float(np.max(np.abs(budget-L))/scale) if attenuation else None
    heating_error=float(np.max(np.abs(grain_power-absorbed))/max(float(absorbed.max()),1.))
    if attenuation and (err>1e-10 or heating_error>1e-9):raise RuntimeError('Direct energy bookkeeping failed')
    return dict(T=T,alive=alive,absorbed_erg_s=absorbed*1e7,
                escaped_direct_erg_s=escaped_direct,initial_tau_lambda=initial_tau,
                energy_budget_relative_max=err,heating_partition_relative_max=heating_error,
                max_cell_tau_lambda=max_cell_tau,max_sublimation_iterations=max(iterations))
