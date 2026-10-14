"""Evaluate one immutable list of 15 RT configurations; no parameter search."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'V48_CASE_B_CHECK'
if not (OLD/'run_validation.py').is_file():
    raise RuntimeError('Missing V48_CASE_B_CHECK/run_validation.py')
sys.path.insert(0, str(OLD))
from cache_utils import read, write
from budget import Budget
from rt_queue import acquire, schedule

HERE = Path(__file__).resolve().parent

def ident(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]

def checked_result(modelroot, point, provenance):
    d = modelroot / point['id']
    s, p = read(d / 'summary.json'), read(d / 'provenance.json')
    if not (d / 'response.csv').is_file():
        raise ValueError('Missing saved response: ' + str(d))
    if (s['config'] != point['config'] or s['numerics'] != provenance['numerics']
            or p['hashes'] != provenance['hashes'] or s['provenance_sha256'] != p['sha256']
            or p.get('python') != provenance['python'] or p.get('numpy') != provenance['numpy']):
        raise ValueError('Incompatible cached result: ' + str(d))
    chi = float(s['fit']['chi2_total'])
    if not s['static_opacity_gate_passed'] or not np.isfinite(chi):
        raise ValueError('Invalid or unsupported static-opacity result: ' + str(d))
    response = np.genfromtxt(d/'response.csv', delimiter=',', names=True)
    required = {'time_obs_days', 'W1_Jy', 'W2_Jy'}
    if not required.issubset(response.dtype.names or ()):
        raise ValueError('Response columns missing: '+str(d))
    t = np.atleast_1d(response['time_obs_days'])
    f = np.array([np.atleast_1d(response[b+'_Jy']) for b in ('W1','W2')])
    if len(t)<2 or not np.isfinite(t).all() or not np.isfinite(f).all() or not (np.diff(t)>0).all():
        raise ValueError('Invalid response sampling: '+str(d))
    obs = np.genfromtxt(ROOT/'data/UGC11487_WISE_TDE_model_input_22epochs.csv', delimiter=',', names=True)
    reproduced = response_chi2(t, f, obs, s['fit']['t0_mjd'])
    if abs(reproduced-chi)>1e-6:
        raise ValueError('Response does not reproduce saved chi2: '+str(d))
    fixed_t0 = point.get('fixed_t0')
    evaluated_chi = chi if fixed_t0 is None else response_chi2(t, f, obs, fixed_t0)
    return {'id': point['id'], 'name': point['name'], 'chi2': evaluated_chi,
            'chi2_profiled_t0': chi, 'fit': s['fit'], 'fixed_t0': fixed_t0,
            'parameter': point.get('parameter'), 'side': point.get('side'),
            'prediction_error': None if point['predicted_chi2'] is None else evaluated_chi-point['predicted_chi2'],
            'inside_training_convex_hull': point['inside_training_convex_hull'],
            'purpose': point['purpose']}

def response_chi2(t, f, obs, t0):
    total = 0.
    for j, band in enumerate(('W1','W2')):
        sigma = np.atleast_1d(obs['sigma_'+band+'_Jy'])
        host = np.atleast_1d(obs['baseline_'+band+'_mJy'])[0]/1000.
        values = np.atleast_1d(obs['F_'+band+'_Jy'])
        mjd = np.atleast_1d(obs['mjd'])
        if not np.isfinite(sigma).all() or not (sigma>0).all() or not np.isfinite(values).all() or not np.isfinite(mjd).all() or not np.isfinite(host):
            raise ValueError('Invalid observations or uncertainties')
        model = np.interp(mjd-t0,t,f[j],left=0,right=0)
        residual = (values-host-model)/sigma
        total += float(residual@residual)
    return total

def make_diagnostic(out, plan, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    center = next((r['chi2'] for r in results if r['name']=='center'), None)
    keys = ['eta','tau_visc','tau_C','tau_E','theta','inc','t0_mjd']
    fig, axes = plt.subplots(3,3,figsize=(13,10))
    lookup = {p['id']:p for p in plan['points']}
    for ax,key in zip(axes.flat,keys):
        curve = sorted(plan['profile_curves'][key],key=lambda x:x['value'])
        ax.plot([v['value'] for v in curve],[v['delta'] for v in curve],color='0.55',label='Emulator profile')
        for r in results:
            if r.get('parameter')!=key or center is None: continue
            point = lookup[r['id']]
            value = point['fixed_t0'] if key=='t0_mjd' else point['config'][key]
            ax.scatter([value],[r['chi2']-center],color='tab:blue' if point['inside_training_convex_hull'] else 'tab:orange',marker='o' if point['inside_training_convex_hull'] else 'x',zorder=3)
        ax.axhline(1,color='k',ls='--',lw=.8)
        ax.set(xlabel=key,ylabel=r'$\Delta\chi^2$')
    for ax in list(axes.flat)[7:]: ax.set_visible(False)
    fig.suptitle('Provisional profiles; RT endpoints relative to checked center\nBlue: inside training hull; orange: outside. Intervals not certified.')
    fig.tight_layout(rect=[0,0,1,.93])
    for ext in ('png','pdf'):fig.savefig(out/('case_b_refine453_diagnostic.'+ext),dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, choices=range(1, 16), default=4)
    ap.add_argument('--minutes', type=float, default=45)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--stop', action='store_true')
    a = ap.parse_args()
    if not 0 < a.minutes < float('inf'): ap.error('--minutes must be positive')
    root = ROOT
    os.chdir(root)
    out = root / 'results/v48_case_b_refine453'
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        print(json.dumps(read(out/'report.json'), indent=2) if (out/'report.json').exists() else 'Not started')
        return
    if a.stop:
        (out/'STOP').touch(); print('STOP requested for this validation only'); return
    plan = json.loads(PLAN_JSON)
    points = plan['points']
    if len(points) != 15 or len({p['id'] for p in points}) != 15:
        raise RuntimeError('Expected exactly 15 distinct fixed configurations')
    if any(ident(p['config']) != p['id'] for p in points):
        raise RuntimeError('Configuration hash mismatch')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    locks, finished = [], {}
    modelroot = root/'results/v48_paper_final/models/fine'
    reason = 'preflight'
    error_details = None
    started = time.monotonic()
    try:
        for directory in ('v48_uncertainties', 'v48_parallel_final', 'v48_targeted_profiles',
                          'v48_profile_continue', 'v48_uncertainty_repair', 'v48_response_validation',
                          'v48_profile_final_check', 'v48_case_b_validation', 'v48_case_b_support',
                          'v48_case_b_two_checks', 'v48_case_b_profile_checks', 'v48_case_b_refine438', 'v48_case_b_refine453'):
            locks.append(acquire(root/'results'/directory/'coordinator.lock'))
        binding = out/'plan_identity.json'
        if binding.exists() and read(binding)['sha256'] != signature:
            raise RuntimeError('Saved plan differs; preserve results for review')
        write(binding, {'sha256': signature, 'fixed_models': 15})
        sys.path.insert(0, str(root/'V48_PAPER_FINAL'))
        import run_final as rf
        if rf.ROOT.resolve() != root.resolve(): raise RuntimeError('Wrong project root')
        print('PREFLIGHT unchanged solver/data/runtime', flush=True)
        rf.prepare()
        provenance = read(OLD/'seed.json')['provenance']
        for name, expected in provenance['hashes'].items():
            pieces = name.replace('\\', '/').split('/UGC11487_V4_FIXED/', 1)
            if len(pieces) != 2: raise RuntimeError('Unknown provenance path: '+name)
            path = root/pieces[1]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Source/data mismatch: '+str(path))
        if platform.python_version() != provenance['python'] or np.__version__ != provenance['numpy']:
            raise RuntimeError('Python/NumPy differ from the archived calculation; no RT launched')
        reference_point = {'id': plan['reference']['id'], 'name': 'reference',
                           'config': read(modelroot/plan['reference']['id']/'config.json'),
                           'predicted_chi2': plan['reference']['chi2'],
                           'inside_training_convex_hull': True, 'purpose': 'reference'}
        reference_result = checked_result(modelroot, reference_point, provenance)
        if ident(reference_point['config']) != reference_point['id'] or abs(reference_result['chi2']-plan['reference']['chi2']) > 1e-6:
            raise RuntimeError('Reference configuration or chi-square differs')
        # Scan ONLY the 15 selected directories, not the thousands of archived models.
        for point in points:
            d = modelroot/point['id']
            if (d/'summary.json').exists():
                finished[point['id']] = checked_result(modelroot, point, provenance)
        print('FIXED MODELS 15; CACHED', len(finished), '; TO RUN', 15-len(finished), flush=True)
        if a.check_only:
            reason = 'preflight_passed'; return
        budget = Budget(out/'total_budget.json', 19)
        print('TOTAL ATTEMPTS REMAINING', budget.remaining, '(15 targets + at most 4 retry attempts)', flush=True)
        (out/'STOP').unlink(missing_ok=True)
        # Rebuild the checkpoint from validated files; never trust stale completion flags.
        write(out/'checkpoint.json', {'finished': finished})
        def command(point):
            d = modelroot/point['id']; d.mkdir(parents=True, exist_ok=True)
            write(d/'config.json', point['config'])
            return [sys.executable, '-u', str(root/'V48_PAPER_FINAL/_solver_fine.py'),
                    '--resolution', 'fine', '--config', str(d/'config.json'), '--output', str(d)]
        def collect(point): return checked_result(modelroot, point, provenance)
        remaining = a.minutes*60-(time.monotonic()-started)
        if remaining <= 0:
            reason = 'time_limit_before_RT'
        else:
            finished, reason = schedule(points, out, command, collect, a.workers,
                                        remaining, 15, budget=budget)
    except Exception:
        error_details = traceback.format_exc()
        reason = 'error_requires_review'
        raise
    finally:
        # A failed lock acquisition must not overwrite another running coordinator's report.
        if len(locks) == 13:
            if 'provenance' in locals():
                for point in points:
                    try: finished[point['id']] = checked_result(modelroot, point, provenance)
                    except (OSError, ValueError, KeyError): pass
            center = next((r['chi2'] for r in finished.values() if r['name']=='center'), None)
            for r in finished.values():
                r['delta_from_archived_reference'] = r['chi2']-plan['reference']['chi2']
                r['delta_from_validated_center'] = None if center is None else r['chi2']-center
                r['delta_from_best_checked_reference'] = r['chi2']-min(plan['reference']['chi2'], center if center is not None else plan['reference']['chi2'])
            report = {'status': reason, 'completed': len(finished), 'planned': 15,
                      'validation_complete': len(finished)==15, 'confidence_intervals_certified': False,
                      'error_details': error_details, 'elapsed_minutes': (time.monotonic()-started)/60, 'results': list(finished.values()),
                      'note': 'Fixed-point RT checks only. Nuisance-minimum stationarity, profile crossings and numerical-resolution effects require assessment; no automatic confidence certification.'}
            try:
                make_diagnostic(out, plan, list(finished.values()))
            except Exception:
                report['figure_error'] = traceback.format_exc()
            write(out/'report.json', report)
            dest = out/'SEND_CASE_B_REFINE453.zip'
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('validation_plan.json', json.dumps(plan, indent=2))
                for filename in ('report.json', 'checkpoint.json', 'total_budget.json', 'plan_identity.json'):
                    p = out/filename
                    if p.exists(): z.write(p, filename)
                for p in out.glob('*.log'): z.write(p, 'logs/'+p.name)
                z.write(Path(__file__), 'run_case_b_refine453.py')
                for ext in ('png','pdf'):
                    p=out/('case_b_refine453_diagnostic.'+ext)
                    if p.exists(): z.write(p,p.name)
                for point in points:
                    for filename in ('summary.json', 'provenance.json', 'config.json', 'response.csv'):
                        p = modelroot/point['id']/filename
                        if p.exists(): z.write(p, 'models/'+point['id']+'/'+filename)
            print('FINISHED', reason, 'SAVED', len(finished), '/15', flush=True)
            print('SEND', dest, flush=True)
        for lock in reversed(locks): lock.close()

PLAN_JSON = '{"reference": {"id": "6dc9df8686595bfdf766", "chi2": 58.92754691047183}, "points": [{"name": "center", "id": "3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 58.84829903606471, "predicted_t0": 57065.182610885524, "fixed_t0": null, "parameter": null, "side": null, "inside_training_convex_hull": true, "purpose": "new emulator minimum; direct RT check"}, {"name": "eta_lower", "id": "46b3ba3f0795103aed54", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 224.98002255631425, "eta": 0.17283273239282054, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3677730171399324, "tau_E": 1.5193335257776386, "theta": 44.000426114017195, "inc": 41.05097754933525, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.85272428575601, "predicted_t0": 57081.4359112121, "fixed_t0": null, "parameter": "eta", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "eta_upper", "id": "c3a02c18d29bd054047c", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 261.56835346304865, "eta": 0.19138098436683557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.4007509362871516, "tau_E": 2.4685601133527073, "theta": 48.22445789872126, "inc": 43.72079041857325, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.850591762132815, "predicted_t0": 57052.38403081301, "fixed_t0": null, "parameter": "eta", "side": "upper", "inside_training_convex_hull": false, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_visc_lower", "id": "16daaf63e7fb805fe59c", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 208.41492094031426, "eta": 0.17746130164825188, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.37108768335667974, "tau_E": 1.7583744134322628, "theta": 45.25029913296837, "inc": 42.70317024715159, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.84431667933933, "predicted_t0": 57085.86542859424, "fixed_t0": null, "parameter": "tau_visc", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_visc_upper", "id": "8271b60e51aa38626f9d", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 274.62217584328323, "eta": 0.18660483214211038, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3992058760136433, "tau_E": 2.187187785348352, "theta": 48.1577354768756, "inc": 42.91743151380926, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.85171140852526, "predicted_t0": 57051.2260355962, "fixed_t0": null, "parameter": "tau_visc", "side": "upper", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_C_lower", "id": "5e1655888efb403647e7", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 225.46617780001873, "eta": 0.17679031806125647, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.35488516274863957, "tau_E": 1.5501694585778518, "theta": 45.362589680331006, "inc": 41.23687100457277, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.85218723416302, "predicted_t0": 57078.97307318001, "fixed_t0": null, "parameter": "tau_C", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_C_upper", "id": "78151a952207f3b83efb", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 257.53986087676253, "eta": 0.18706983899800314, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.4129661412525297, "tau_E": 2.2777899464507834, "theta": 47.29765064409383, "inc": 44.070863923419054, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.84913771586778, "predicted_t0": 57054.9598030825, "fixed_t0": null, "parameter": "tau_C", "side": "upper", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_E_lower", "id": "ac801b94cab912fc1db6", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 230.75515988736223, "eta": 0.1754319642332427, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3633710801010179, "tau_E": 1.3777120745161704, "theta": 44.44592682395724, "inc": 42.04639220742427, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.847916693549735, "predicted_t0": 57078.85755049017, "fixed_t0": null, "parameter": "tau_E", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "tau_E_upper", "id": "08806980a50de6f453d2", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 260.16350827415573, "eta": 0.18987314069858593, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.39887240776108995, "tau_E": 2.585386221893282, "theta": 47.72852238662786, "inc": 42.6703032989416, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.847370666791946, "predicted_t0": 57055.03213052905, "fixed_t0": null, "parameter": "tau_E", "side": "upper", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "theta_lower", "id": "409dd0b6119694c7dc7f", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 225.26805781568925, "eta": 0.17418729312635609, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3677546854418138, "tau_E": 1.5012048005296779, "theta": 43.36355107818547, "inc": 41.82029100254767, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.84727406795575, "predicted_t0": 57080.05131172257, "fixed_t0": null, "parameter": "theta", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "theta_upper", "id": "c748772d198bd01b6d49", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 257.30941992558314, "eta": 0.18558823465655572, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.386730693589292, "tau_E": 2.01589474969189, "theta": 49.24596203813148, "inc": 43.02512779703041, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.850032470397544, "predicted_t0": 57062.384587348606, "fixed_t0": null, "parameter": "theta", "side": "upper", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "inc_lower", "id": "a5849a0286009a366954", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 246.72525548860796, "eta": 0.1799161165737324, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3744332062772809, "tau_E": 2.0993030560713746, "theta": 46.29549070951649, "inc": 40.248774924880905, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.85294321101632, "predicted_t0": 57067.08465188149, "fixed_t0": null, "parameter": "inc", "side": "lower", "inside_training_convex_hull": false, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "inc_upper", "id": "89c2a25dd72973cbcba6", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 248.6227069281535, "eta": 0.1866209370288228, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.39997497243028834, "tau_E": 2.003004525005154, "theta": 47.3720348554682, "inc": 44.98587918552267, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.84835433570978, "predicted_t0": 57059.28824350175, "fixed_t0": null, "parameter": "inc", "side": "upper", "inside_training_convex_hull": false, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "t0_mjd_lower", "id": "b3cb5fbcd6fd42a625aa", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 268.0543147977808, "eta": 0.188325561038971, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.4023936653996417, "tau_E": 2.271715433375069, "theta": 47.603102236363526, "inc": 43.65553080589821, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.857965119167396, "predicted_t0": 57047.079105380595, "fixed_t0": 57047.079105380595, "parameter": "t0_mjd", "side": "lower", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}, {"name": "t0_mjd_upper", "id": "02a57f368f0158038111", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 213.2519071577061, "eta": 0.17579649719319604, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3685270777866629, "tau_E": 1.587500467116873, "theta": 45.05300111640038, "inc": 42.436380103846645, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "predicted_chi2": 59.8409144574032, "predicted_t0": 57088.7918363101, "fixed_t0": 57088.7918363101, "parameter": "t0_mjd", "side": "upper", "inside_training_convex_hull": true, "purpose": "provisional profile endpoint; direct RT check"}], "n_models_training": 453, "emulator_minimum_chi2": 58.84829903606471, "profile_curves": {"eta": [{"value": 0.1823643760379557, "delta": 0.0}, {"value": 0.17505868510036707, "delta": 0.5254002814163385}, {"value": 0.16410014869398415, "delta": 5.805832196034437}, {"value": 0.1695794168971756, "delta": 2.290240267891022}, {"value": 0.17231905099877134, "delta": 1.1578784884547204}, {"value": 0.1736888680495692, "delta": 0.7934739816679013}, {"value": 0.17300395952417028, "delta": 0.9604314343268285}, {"value": 0.1726615052614708, "delta": 1.0512065195062519}, {"value": 0.17283273239282054, "delta": 1.0044252496913018}, {"value": 0.18422586937036958, "delta": 0.07051740437194809}, {"value": 0.18701810936899047, "delta": 0.32324171714179784}, {"value": 0.19167184270002524, "delta": 1.047799680109243}, {"value": 0.18934497603450784, "delta": 0.653132319765156}, {"value": 0.19050840936726654, "delta": 0.863176884073404}, {"value": 0.1910901260336459, "delta": 0.9546244682735434}, {"value": 0.19138098436683557, "delta": 1.0022927260681058}], "tau_visc": [{"value": 245.24509766270944, "delta": 0.0}, {"value": 228.81909550080496, "delta": 0.1331176595269099}, {"value": 204.18009225794827, "delta": 1.3372407041678827}, {"value": 216.49959387937662, "delta": 0.5211875655293028}, {"value": 210.33984306866245, "delta": 0.8634535279116236}, {"value": 207.25996766330536, "delta": 1.0836907158207225}, {"value": 208.7999053659839, "delta": 0.9677355995914567}, {"value": 208.02993651464462, "delta": 1.0247661463497266}, {"value": 208.41492094031426, "delta": 0.9960176432746195}, {"value": 262.03199948018016, "delta": 0.3627406822431354}, {"value": 287.21235220638624, "delta": 1.4890818955007745}, {"value": 274.62217584328323, "delta": 1.0034123724605521}], "tau_C": [{"value": 0.38510950854710485, "delta": 0.0}, {"value": 0.37641576413765865, "delta": 0.10911292620456692}, {"value": 0.3633751475234894, "delta": 0.4623065480209476}, {"value": 0.34164078649987384, "delta": 2.3431434714236374}, {"value": 0.3525079670116816, "delta": 1.2042638272005135}, {"value": 0.3579415572675855, "delta": 0.7724694707932898}, {"value": 0.35522476213963355, "delta": 0.9763338086177811}, {"value": 0.35386636457565757, "delta": 1.0888587596798232}, {"value": 0.35454556335764553, "delta": 1.0318489783161482}, {"value": 0.35488516274863957, "delta": 1.0038881980983092}, {"value": 0.4222516854876713, "delta": 1.6387913846469857}, {"value": 0.40368059701738807, "delta": 0.47934918190200193}, {"value": 0.4129661412525297, "delta": 1.0008386798030742}], "tau_E": [{"value": 1.981951407809012, "delta": 0.0}, {"value": 1.8519637865292073, "delta": 0.04814056102495101}, {"value": 1.6569823546095004, "delta": 0.21567382295173587}, {"value": 1.332013301409989, "delta": 1.2412765390588945}, {"value": 1.4944978280097447, "delta": 0.5480374636243539}, {"value": 1.413255564709867, "delta": 0.8462058848811793}, {"value": 1.372634433059928, "delta": 1.0236646440805899}, {"value": 1.3929449988848974, "delta": 0.9312868329452186}, {"value": 1.3827897159724127, "delta": 0.976289327398284}, {"value": 1.3777120745161704, "delta": 0.9996176574850253}, {"value": 2.1574960809971633, "delta": 0.1329673772384865}, {"value": 2.42081309077939, "delta": 0.5976645999171097}, {"value": 2.8596747737497683, "delta": 1.6898168524562465}, {"value": 2.640243932264579, "delta": 1.1267054024237524}, {"value": 2.5305285115219847, "delta": 0.8737927968227339}, {"value": 2.585386221893282, "delta": 0.9990716307272365}], "theta": [{"value": 46.90535162653177, "delta": 0.0}, {"value": 44.70462118872436, "delta": 0.4929957954863866}, {"value": 41.403525532013255, "delta": 1.8474037058996728}, {"value": 43.054073360368804, "delta": 1.1368720561278778}, {"value": 43.87934727454658, "delta": 0.7735772282710229}, {"value": 43.4667103174577, "delta": 0.9540380854786363}, {"value": 43.26039183891325, "delta": 1.044596552487441}, {"value": 43.36355107818547, "delta": 0.9989750318910424}, {"value": 47.80302393420147, "delta": 0.12365595744291369}, {"value": 49.149532395706025, "delta": 0.9105896852425133}, {"value": 51.39371316488028, "delta": 2.4926938547703728}, {"value": 50.27162278029315, "delta": 1.681548476706844}, {"value": 49.71057758799959, "delta": 1.353547481784986}, {"value": 49.430054991852806, "delta": 1.1489922175181206}, {"value": 49.289793693779416, "delta": 1.044949450591119}, {"value": 49.21966304474272, "delta": 0.9763484457048506}, {"value": 49.254728369261066, "delta": 1.0102644800571738}, {"value": 49.237195707001895, "delta": 0.9932185372496534}, {"value": 49.24596203813148, "delta": 1.0017334343328343}], "inc": [{"value": 42.662643087378456, "delta": 0.0}, {"value": 41.31045435740171, "delta": 0.3479993214971131}, {"value": 39.2821712624366, "delta": 1.7853501716223619}, {"value": 40.29631280991915, "delta": 0.9704542233732312}, {"value": 39.789242036177875, "delta": 1.3564910890645976}, {"value": 40.04277742304851, "delta": 1.1575260010099626}, {"value": 40.16954511648383, "delta": 1.062685797465683}, {"value": 40.232928963201495, "delta": 1.0159834254207567}, {"value": 40.26462088656032, "delta": 0.9935683083696389}, {"value": 40.248774924880905, "delta": 1.0046441749516077}, {"value": 43.42028441365224, "delta": 0.10946147581963572}, {"value": 44.55674640306291, "delta": 0.6922441942854576}, {"value": 46.45084971874737, "delta": 2.2208887918878517}, {"value": 45.50379806090514, "delta": 1.4067723230083118}, {"value": 45.03027223198403, "delta": 1.0339059780989643}, {"value": 44.79350931752347, "delta": 0.8579872332441383}, {"value": 44.91189077475375, "delta": 0.9446130846425973}, {"value": 44.971081503368886, "delta": 0.9888482207993405}, {"value": 45.000676867676454, "delta": 1.011166977653076}, {"value": 44.98587918552267, "delta": 1.0000552996450693}], "t0_mjd": [{"value": 57065.182610885524, "delta": 0.0}, {"value": 56952.14608870842, "delta": 45.85346823869171}, {"value": 57008.66434979697, "delta": 4.20279587157647}, {"value": 57036.92348034125, "delta": 1.5924510245382777}, {"value": 57051.053045613386, "delta": 0.6478036368911901}, {"value": 57043.98826297732, "delta": 1.2711293740635767}, {"value": 57047.52065429535, "delta": 0.9668961956595012}, {"value": 57045.75445863634, "delta": 1.142334302119032}, {"value": 57046.637556465845, "delta": 1.0535296592482126}, {"value": 57047.079105380595, "delta": 1.0096660831026867}, {"value": 57047.29987983797, "delta": 0.9880092167735981}, {"value": 57152.14608870842, "delta": 24.369142482736585}, {"value": 57108.66434979697, "delta": 3.9503369184659363}, {"value": 57086.92348034125, "delta": 0.805797608316098}, {"value": 57097.79391506911, "delta": 2.120764025630372}, {"value": 57092.35869770518, "delta": 1.3971904252562481}, {"value": 57089.641089023215, "delta": 1.0833205490957454}, {"value": 57088.28228468223, "delta": 0.9400052748080441}, {"value": 57088.96168685272, "delta": 1.0104448889978528}, {"value": 57088.62198576747, "delta": 0.9749317682303271}, {"value": 57088.7918363101, "delta": 0.9926154213384919}]}, "scope": "Approximate conditional Delta chi2=1 profiles; endpoint RT validation only. No automatic interval certification."}'

if __name__ == "__main__": main()
