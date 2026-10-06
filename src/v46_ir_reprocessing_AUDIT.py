"""V4.6 radial outward diffuse-IR reprocessing control.

Extension of V4.5 finite-volume direct absorption. Central-source photons are
attenuated exactly as in V4.5. Dust emission from an inner radial cell may also
illuminate outer cells. The reprocessed field is spectral and is computed from
the same Q_abs(a,lambda) and Planck emission used by the thermal core.

This is deliberately NOT full 3-D radiative transfer: diffuse emission is split
into an outward fraction and an inner-boundary escape fraction. Only the outward
part is propagated radially through later cells; scattering and non-radial paths
are omitted. In source-retarded time u=t_dust-r/c, instantaneous radial outward
reprocessing introduces no additional u-delay.
"""
from __future__ import annotations
import numpy as np
from src.v45_direct_absorption import quadrature_weights, mean_transmission
from src.v4_thermal_lookup import lookup_temperature_from_flux
from src.v4_thermal import planck_lambda_si

PC_M=3.085677581491367e16


def solve_direct_plus_ir(time_days, luminosity_erg_s, wavelength_m, source_sed,
                          cross_sections_m2, qsrc, lookup, radius_pc,
                          cell_grain_numbers, solid_angle_fraction,
                          sublimation_K, *, attenuation=True,
                          ir_reprocess_fraction=0.5):
    """Radial finite-volume direct + outward diffuse-IR transfer.

    ir_reprocess_fraction is the fraction of each cell's isotropic dust emission
    launched into the outward radial stream. 0 reproduces V4.5 exactly in dust
    temperatures and direct transfer; 0.5 is the fiducial isotropic hemisphere
    split. The complementary fraction escapes through the inner boundary.
    """
    t=np.asarray(time_days,float); L=np.asarray(luminosity_erg_s,float)
    lam=np.asarray(wavelength_m,float); sed=np.asarray(source_sed,float)
    C=np.asarray(cross_sections_m2,float); qsrc=np.asarray(qsrc,float)
    r=np.asarray(radius_pc,float); numbers=np.asarray(cell_grain_numbers,float)
    f=float(solid_angle_fraction); eps=float(ir_reprocess_fraction)
    w=quadrature_weights(lam)
    if not 0<=eps<=1: raise ValueError('ir_reprocess_fraction must be in [0,1]')
    if not 0<f<=1: raise ValueError('Invalid solid angle fraction')
    if L.shape!=t.shape or np.any(~np.isfinite(L)) or np.any(L<0) or np.any(np.diff(t)<=0): raise ValueError('Invalid driver')
    if np.any(r<=0) or np.any(np.diff(r)<=0): raise ValueError('Increasing radial cells required')
    if numbers.shape!=(len(r),len(qsrc)) or C.shape!=(len(qsrc),len(lam)): raise ValueError('Shape mismatch')
    if sed.shape!=lam.shape or np.any(sed<0) or not np.isclose(sed@w,1,rtol=1e-8): raise ValueError('Source SED must integrate to unity')

    # W/m spectral luminosity streams inside the dusty wedge.
    direct=f*L[:,None]*1e-7*sed[None,:]
    diffuse=np.zeros_like(direct)
    T=np.zeros((len(t),len(r),len(qsrc))); alive=np.ones(T.shape,bool)
    abs_direct=np.zeros((len(t),len(r))); abs_ir=np.zeros_like(abs_direct)
    emit_bol=np.zeros_like(abs_direct)
    raw_emit_bol=np.zeros_like(abs_direct)
    direct_after_cell=np.zeros_like(abs_direct)
    diffuse_after_cell=np.zeros_like(abs_direct)
    inner_escape=np.zeros(len(t)); initial_tau=np.zeros(len(lam))
    max_cell_tau=0.; iterations=[]
    area=np.pi*(np.asarray(lookup.grain_radius_micron)*1e-6)**2
    Q=C/area[:,None]

    for ir,rr in enumerate(r):
        shell_area=4*np.pi*f*(rr*PC_M)**2
        initial_tau += numbers[ir]@C/shell_area
        live=np.ones((len(t),len(qsrc)),bool)
        incoming=direct+diffuse
        for it in range(100):
            tau=(live*numbers[ir][None,:])@C/shell_area
            field=incoming/shell_area
            if attenuation: field=field*mean_transmission(tau)
            absorbed_per_area=(field*w[None,:])@Q.T
            equivalent=absorbed_per_area/qsrc[None,:]
            temp=np.column_stack([lookup_temperature_from_flux(lookup,equivalent[:,ia],ia) for ia in range(len(qsrc))])
            hot=(~np.isfinite(temp))|(temp>=sublimation_K)
            destroyed=np.maximum.accumulate(hot&live,axis=0)
            new_live=live & ~destroyed
            if np.array_equal(new_live,live): break
            live=new_live
        else: raise RuntimeError('Sublimation did not converge')
        T[:,ir,:]=np.where(live,temp,0); alive[:,ir,:]=live

        # Spectral removal from each incoming stream, independently but through same tau.
        if attenuation:
            frac=-np.expm1(-tau)
            rem_d=direct*frac; rem_i=diffuse*frac
            abs_direct[:,ir]=rem_d@w*1e7; abs_ir[:,ir]=rem_i@w*1e7
            direct=direct*np.exp(-tau); diffuse=diffuse*np.exp(-tau)
        else:
            # Required only as a regression/reference mode; no diffuse absorption.
            grain=np.sum(absorbed_per_area*area[None,:]*numbers[ir][None,:]*live,axis=1)
            abs_direct[:,ir]=grain*1e7; abs_ir[:,ir]=0.

        absorbed=abs_direct[:,ir]+abs_ir[:,ir]

        # Physical dust-emission spectral shape. Normalize at each time to absorbed
        # bolometric power, enforcing exact radiative-equilibrium energy conservation.
        spec=np.zeros_like(direct)
        for ia in range(len(qsrc)):
            ok=live[:,ia] & (T[:,ir,ia]>0)
            if not np.any(ok): continue
            B=np.array([planck_lambda_si(lam,x) for x in T[ok,ir,ia]])
            spec[ok] += numbers[ir,ia]*4*np.pi*C[ia][None,:]*B  # W/m
        raw=spec@w*1e7
        raw_emit_bol[:,ir]=raw
        scale=np.divide(absorbed,raw,out=np.zeros_like(absorbed),where=raw>0)
        spec*=scale[:,None]
        emit_bol[:,ir]=spec@w*1e7
        if np.max(np.abs(emit_bol[:,ir]-absorbed))/max(float(absorbed.max()),1.)>2e-9:
            raise RuntimeError('IR emission normalization failed')

        # Isotropic split: inward part leaves this 1-D control through inner boundary;
        # outward part joins the diffuse stream and may be reabsorbed by later cells.
        inner_escape += (1-eps)*emit_bol[:,ir]
        diffuse += eps*spec
        direct_after_cell[:,ir]=(direct@w)*1e7
        diffuse_after_cell[:,ir]=(diffuse@w)*1e7
        max_cell_tau=max(max_cell_tau,float(tau.max())); iterations.append(it+1)

    escaped_direct=(1-f)*L+(direct@w)*1e7
    escaped_diffuse=(diffuse@w)*1e7
    escaped_total=escaped_direct+inner_escape+escaped_diffuse
    # Total source energy is conserved instantaneously in this radial/instantaneous control.
    err=float(np.max(np.abs(escaped_total-L))/max(float(L.max()),1.))
    if attenuation and err>2e-9: raise RuntimeError(f'V4.6 energy bookkeeping failed: {err}')
    return dict(T=T,alive=alive,absorbed_direct_erg_s=abs_direct,
                absorbed_ir_erg_s=abs_ir,emitted_dust_erg_s=emit_bol,
                raw_emitted_dust_erg_s=raw_emit_bol,
                direct_after_cell_erg_s=direct_after_cell,
                diffuse_after_cell_erg_s=diffuse_after_cell,
                escaped_direct_erg_s=escaped_direct,
                escaped_ir_inner_erg_s=inner_escape,
                escaped_ir_outer_erg_s=escaped_diffuse,
                escaped_total_erg_s=escaped_total,
                initial_tau_lambda=initial_tau,
                energy_budget_relative_max=err,
                max_cell_tau_lambda=max_cell_tau,
                max_sublimation_iterations=max(iterations) if iterations else 0,
                ir_reprocess_fraction=eps)
