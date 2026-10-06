"""
v4_two_layer_rt.py
===================
Controlled two-layer radiative-transfer extension for UGC 11487 V4.

This is deliberately a 1-D population-averaged control, not a full 3-D RT
solver.  It adds one piece of physics absent from V4.3b:

  central UV/optical source -> compact dust -> reprocessed IR -> extended dust

The compact population spectrum is computed from the same graphite opacity,
grain-size distribution, radiative-equilibrium temperatures, and physical
population normalization used by V4.  The extended grains are heated by the
sum of

  (1) an attenuated direct central-source field, and
  (2) a delayed compact-dust spectrum.

For each extended grain size the secondary heating uses the actual
Q_abs-weighted compact spectrum, rather than assigning an arbitrary IR
blackbody temperature.

Control approximation
---------------------
The compact population is collapsed to an unresolved isotropic reprocessor
when illuminating the extended zone.  Therefore this module is suitable for
testing the SIGN and approximate magnitude of secondary heating.  It is not a
replacement for full angle-dependent 3-D radiative transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.v4_thermal import planck_lambda_si, MICRON_M, PC_M
from src.v4_thermal_lookup import lookup_temperature_from_flux
from src.v4_sublimation import sublimation_temperature_K
from src.v4_wise_emission import grain_size_number_weights

ERG_S_TO_W = 1e-7
DAY_S = 86400.0
C_M_S = 299792458.0
PC_TO_LIGHT_DAYS = PC_M / C_M_S / DAY_S


@dataclass
class TwoLayerHeatingResult:
    temperature_K: np.ndarray
    alive: np.ndarray
    direct_equivalent_flux_W_m2: np.ndarray
    secondary_equivalent_flux_W_m2: np.ndarray
    compact_Lbol_erg_s: np.ndarray
    compact_spectrum_W_m: np.ndarray
    wavelength_m: np.ndarray


def compact_population_spectrum(
    thermal_solver, opacity, temperature_K, alive,
    radii_pc, radial_weights, grains_micron, q,
    total_grain_number_fabs1, f_abs_compact,
):
    """Intrinsic compact-dust L_lambda(t) [W m^-1] and Lbol [erg/s]."""
    T=np.asarray(temperature_K,float); alive=np.asarray(alive,bool)
    r=np.asarray(radii_pc,float); wr=np.asarray(radial_weights,float); wr=wr/wr.sum()
    a=np.asarray(grains_micron,float); wa=grain_size_number_weights(a,q=q)
    if T.shape != alive.shape or T.shape[1:] != (r.size,a.size):
        raise ValueError("compact temperature/alive shape mismatch")
    lam_um=np.asarray(thermal_solver.wavelength_micron,float)
    lam_m=np.asarray(thermal_solver.wavelength_m,float)
    qla=np.asarray(opacity.q_abs(lam_um[:,None],a[None,:]),float)  # Nlam,Na
    a_m=a*MICRON_M
    geom=4*np.pi**2*a_m**2  # one-grain Llambda factor multiplying Q B
    out=np.zeros((T.shape[0],lam_m.size),float)
    N=float(total_grain_number_fabs1)*float(f_abs_compact)
    for ia in range(a.size):
        valid=alive[:,:,ia] & np.isfinite(T[:,:,ia]) & (T[:,:,ia]>0)
        if not np.any(valid): continue
        # Loop over radius; only a small validated radial grid is used.
        for ir in range(r.size):
            m=valid[:,ir]
            if not np.any(m): continue
            B=np.zeros((T.shape[0],lam_m.size),float)
            # Vector Planck expression, equivalent to planck_lambda_si.
            TT=T[m,ir,ia][:,None]
            ll=lam_m[None,:]
            h=6.62607015e-34; c=2.99792458e8; kb=1.380649e-23
            x=h*c/(ll*kb*TT); xc=np.minimum(x,700.)
            bb=2*h*c**2/ll**5/np.expm1(xc); bb=np.where(x>700.,0.,bb)
            B[m]=bb
            out += N*wr[ir]*wa[ia]*geom[ia]*B*qla[:,ia][None,:]
    Lbol_W=np.trapezoid(out,lam_m,axis=1)
    return lam_m,out,Lbol_W/ERG_S_TO_W


def _retard_matrix(time_days, values, radius_pc):
    time=np.asarray(time_days,float); values=np.asarray(values,float); r=np.asarray(radius_pc,float)
    if values.shape[0] != time.size: raise ValueError("time/value mismatch")
    delay=r*PC_TO_LIGHT_DAYS
    if values.ndim==1:
        z=np.empty((time.size,r.size),float)
        for j,d in enumerate(delay):
            z[:,j]=np.interp(time-d,time,values,left=0,right=0)
        return z
    # time x wavelength -> time x radius x wavelength
    z=np.empty((time.size,r.size,values.shape[1]),float)
    for j,d in enumerate(delay):
        tq=time-d
        for k in range(values.shape[1]):
            z[:,j,k]=np.interp(tq,time,values[:,k],left=0,right=0)
    return z


def mixed_extended_temperature(
    time_days, source_luminosity_erg_s, extended_radii_pc,
    grains_micron, thermal_solver, thermal_lookup, opacity,
    composition, density_cm3,
    compact_Llambda_W_m, compact_wavelength_m,
    direct_transmission=1.0, secondary_coupling=0.0,
):
    """Extended T(t,r,a) for attenuated direct + compact-dust secondary field.

    `secondary_coupling` is the fraction (0..1) of the isotropic compact-dust
    luminosity made available to illuminate the extended population in this
    population-averaged control.  It cannot create energy: the secondary
    spectrum itself is the physically emitted compact luminosity.
    """
    time=np.asarray(time_days,float); L=np.asarray(source_luminosity_erg_s,float)
    r=np.asarray(extended_radii_pc,float); a=np.asarray(grains_micron,float)
    tr=float(direct_transmission); eps=float(secondary_coupling)
    if not (0<=tr<=1 and 0<=eps<=1): raise ValueError("transmission/coupling must be in [0,1]")
    # Direct source-equivalent bolometric flux (the lookup is defined for the source SED).
    Lret=_retard_matrix(time,L,r)
    r_m=r*PC_M
    Fdir=tr*(Lret*ERG_S_TO_W)/(4*np.pi*r_m[None,:]**2)  # Nt,Nr

    # Compact spectrum propagated to each extended radius under unresolved-central approximation.
    spec=np.asarray(compact_Llambda_W_m,float); lam=np.asarray(compact_wavelength_m,float)
    sret=_retard_matrix(time,spec,r)  # Nt,Nr,Nlam
    # Spectral flux density at each extended radius.
    Fsec_lam=eps*sret/(4*np.pi*r_m[None,:,None]**2)

    qsrc=np.asarray(thermal_solver.source_mean_qabs(a),float)
    lam_um = np.asarray(thermal_solver.wavelength_micron, dtype=float)
    qsec = np.asarray(opacity.q_abs(lam_um[:, None], a[None, :]), float)  # Nlam, Na
    # Absorbed secondary flux divided by <Q>src gives the source-equivalent
    # incident flux accepted by the already validated thermal lookup.
    Fsec_eq=np.empty((time.size,r.size,a.size),float)
    for ia in range(a.size):
        absorbed=np.trapezoid(Fsec_lam*qsec[:,ia][None,None,:],lam,axis=2)
        Fsec_eq[:,:,ia]=absorbed/qsrc[ia]

    T=np.zeros((time.size,r.size,a.size),float)
    alive=np.ones_like(T,dtype=bool)
    Tsub=float(sublimation_temperature_K(density_cm3=density_cm3,composition=composition))
    for ia in range(a.size):
        Feq=Fdir+Fsec_eq[:,:,ia]
        Teq=lookup_temperature_from_flux(thermal_lookup,Feq,ia)
        hot=(Feq>0)&(~np.isfinite(Teq))
        destroy=hot|(Teq>=Tsub)
        ever=np.maximum.accumulate(destroy,axis=0)
        alive[:,:,ia]=~ever
        store=np.asarray(Teq,float).copy()
        first=destroy & ~np.vstack([np.zeros((1,r.size),bool),ever[:-1]])
        store[first]=Tsub
        store[ever & ~first]=np.nan
        T[:,:,ia]=store
    return T,alive,Fdir,Fsec_eq
