"""Causal radial interpolation and shared 3D transfer (V4_FIXED).

Production responses use u=t_dust-r/c (source_time_input=True).
Default legacy input remains dust-local t; each radial node is shifted by
-r_node/c BEFORE interpolation to the clump radius. Thus all terms refer
to the same source phase, with arrival u+(r_clump-z_clump)/c.

Uniform input/output cadence must agree after cosmological dilation.
Transfer kernels aggregate the identical two-bin linear deposition for all
clumps, then convolve radial responses. No temporal/flux fitting here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve
from src.v4_observer_echo import PC_TO_LIGHT_DAYS

@dataclass
class Echo3DResult:
    time_obs_days: np.ndarray
    response: np.ndarray
    n_clumps: int
    total_input_weight: float
    total_output_weight: float


def _radial_brackets(rg, rc):
    rg, rc = np.asarray(rg,float), np.asarray(rc,float)
    if rg.ndim != 1 or rg.size < 2 or np.any(~np.isfinite(rg)) or np.any(rg<=0) or np.any(np.diff(rg)<=0):
        raise ValueError("Positive increasing radial grid required")
    if rc.ndim != 1 or rc.size == 0 or np.any(~np.isfinite(rc)):
        raise ValueError("Finite nonempty clump radii required")
    tol=1e-12*max(1.,rg[-1])
    if np.any(rc<rg[0]-tol) or np.any(rc>rg[-1]+tol):
        raise ValueError("Clump radius outside radial response grid")
    rc=np.clip(rc,rg[0],rg[-1])
    hi=np.clip(np.searchsorted(rg,rc,side='right'),1,rg.size-1)
    lo=hi-1
    return lo,hi,(rc-rg[lo])/(rg[hi]-rg[lo])


def integrate_radial_responses_over_clumps(
    dust_time_rest_days, radius_grid_pc, radial_responses,
    clump_radius_pc, clump_z_obs_pc, clump_weight=None, redshift=0.,
    output_time_obs_days=None, chunk_size=256, source_time_input=False,
):
    t=np.asarray(dust_time_rest_days,float);rg=np.asarray(radius_grid_pc,float)
    rr=np.asarray(radial_responses,float);rc=np.asarray(clump_radius_pc,float)
    zc=np.asarray(clump_z_obs_pc,float)
    if t.ndim!=1 or t.size<2 or np.any(~np.isfinite(t)):
        raise ValueError("Finite time grid required")
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],rtol=1e-9,atol=1e-10):
        raise ValueError("Uniform increasing time grid required")
    if rr.ndim!=3 or rr.shape[1:]!=(t.size,rg.size) or np.any(~np.isfinite(rr)) or np.any(rr<0):
        raise ValueError("Finite nonnegative responses of shape (Nb,Nt,Nr) required")
    if zc.shape!=rc.shape or np.any(~np.isfinite(zc)) or np.any(np.abs(zc)>rc+1e-12):
        raise ValueError("Clump coordinates must satisfy |z|<=r")
    if not np.isfinite(redshift) or redshift<0:
        raise ValueError("Invalid redshift")
    lo,hi,f=_radial_brackets(rg,rc)
    w=np.full(rc.size,1./rc.size) if clump_weight is None else np.asarray(clump_weight,float)
    if w.shape!=rc.shape or np.any(~np.isfinite(w)) or np.any(w<0):
        raise ValueError("Invalid clump weights")
    dilation=1.+redshift;dto=dt[0]*dilation
    # Each radial interpolation leg has its own correction for native clock.
    nodes=np.concatenate((lo,hi)); weights=np.concatenate((w*(1-f),w*f))
    delay=np.concatenate((rc-zc,rc-zc))*PC_TO_LIGHT_DAYS
    if not source_time_input:
        delay-=rg[nodes]*PC_TO_LIGHT_DAYS
    shift=delay*dilation
    active=weights>0
    nodes,weights,shift=nodes[active],weights[active],shift[active]
    if shift.size==0:
        raise ValueError("Positive total clump weight required")
    if output_time_obs_days is None:
        start=np.floor((t[0]*dilation+shift.min())/dto)*dto
        end=np.ceil((t[-1]*dilation+shift.max())/dto)*dto
        tout=start+np.arange(int(round((end-start)/dto))+1)*dto
    else:
        tout=np.asarray(output_time_obs_days,float)
        if tout.ndim!=1 or tout.size<2 or np.any(~np.isfinite(tout)) or not np.allclose(np.diff(tout),dto,rtol=1e-9,atol=1e-10):
            raise ValueError("Output cadence must match input cadence times (1+z)")
    out=np.zeros((rr.shape[0],tout.size)); totals=np.zeros(rr.shape[0])
    for j in range(rg.size):
        select=nodes==j
        if not np.any(select):continue
        x=(t[0]*dilation+shift[select]-tout[0])/dto
        # Round only machine-close integer indices; no tolerance double counting.
        nearest=np.rint(x);x=np.where(np.abs(x-nearest)<1e-10,nearest,x)
        left=np.floor(x).astype(int); frac=x-left;wj=weights[select]
        kmin=int(left.min());kmax=int(left.max())+1
        kernel=np.bincount(left-kmin,weights=wj*(1-frac),minlength=kmax-kmin+1)
        kernel+=np.bincount(left-kmin+1,weights=wj*frac,minlength=kernel.size)
        support_k=np.flatnonzero(kernel>0)
        for b in range(rr.shape[0]):
            y=rr[b,:,j];totals[b]+=y.sum()*wj.sum()
            support_y=np.flatnonzero(y>0)
            if support_y.size==0:continue
            conv=fftconvolve(y,kernel)
            # Outside exact discrete support the physical answer is exactly zero.
            conv[:support_y[0]+support_k[0]]=0.
            conv[support_y[-1]+support_k[-1]+1:]=0.
            conv=np.maximum(conv,0.)
            src0=max(0,-kmin);dst0=max(0,kmin)
            n=min(conv.size-src0,tout.size-dst0)
            if n>0:out[b,dst0:dst0+n]+=conv[src0:src0+n]
    return [Echo3DResult(tout.copy(),out[b],rc.size,float(totals[b]),float(out[b].sum())) for b in range(rr.shape[0])]


def integrate_radial_response_over_clumps(
    dust_time_rest_days,radius_grid_pc,radial_response,clump_radius_pc,
    clump_z_obs_pc,clump_weight=None,redshift=0.,output_time_obs_days=None,
    chunk_size=256,source_time_input=False,
):
    return integrate_radial_responses_over_clumps(
        dust_time_rest_days,radius_grid_pc,np.asarray(radial_response)[None,:,:],
        clump_radius_pc,clump_z_obs_pc,clump_weight,redshift,
        output_time_obs_days,chunk_size,source_time_input)[0]


def response_centroid_days(time_days,response):
    t,y=np.asarray(time_days,float),np.asarray(response,float)
    mask=np.isfinite(t)&np.isfinite(y)&(y>0)
    return float(np.sum(t[mask]*y[mask])/np.sum(y[mask])) if mask.any() else np.nan


def response_peak_time_days(time_days,response):
    t,y=np.asarray(time_days,float),np.asarray(response,float)
    return float(t[np.argmax(y)]) if np.any(np.isfinite(y)&(y>0)) else np.nan
