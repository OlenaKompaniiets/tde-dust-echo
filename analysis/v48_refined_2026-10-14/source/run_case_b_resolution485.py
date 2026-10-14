"""Check four numerical refinements at the fixed Case B reference; 1 cached baseline + 4 refinements."""
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

def expected_provenance(base, point):
    import copy
    expected = copy.deepcopy(base)
    level = point.get('level', 'fine')
    expected['numerics'].update(point.get('override', {}))
    if level != 'fine':
        matches = [k for k in expected['hashes'] if k.replace('\\','/').endswith('/V48_PAPER_FINAL/_solver_fine.py')]
        if len(matches) != 1: raise ValueError('Missing unique baseline solver provenance')
        old = matches[0]
        new = old.replace('_solver_fine.py', '_solver_'+level+'.py')
        del expected['hashes'][old]
        expected['hashes'][new] = hashlib.sha256((ROOT/'V48_PAPER_FINAL'/('_solver_'+level+'.py')).read_bytes()).hexdigest()
    return expected

def checked_result(modelroot, point, provenance):
    d = modelroot / point.get('level', 'fine') / point.get('model_id', point['id'])
    provenance = expected_provenance(provenance, point)
    s, p = read(d / 'summary.json'), read(d / 'provenance.json')
    if not (d / 'response.csv').is_file():
        raise ValueError('Missing saved response: ' + str(d))
    if (s['config'] != point['config'] or s['numerics'] != provenance['numerics']
            or p['hashes'] != provenance['hashes'] or s['provenance_sha256'] != p['sha256']
            or p.get('numerics') != provenance['numerics'] or p.get('config') != point['config']
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
            'purpose': point['purpose'], 'level': point.get('level','fine'),
            'chi2_fixed_reference_t0': response_chi2(t,f,obs,57065.24758756976),
            'numerics': s['numerics'], 'max_temperature_K': s.get('max_temperature_K'),
            'energy': s.get('energy'), 'static_opacity_gate_passed': s['static_opacity_gate_passed']}

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
    rows={r['name']:r for r in results}
    obs=np.genfromtxt(ROOT/'data/UGC11487_WISE_TDE_model_input_22epochs.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors=dict(zip(('fine','angular','time','radial','spectral'),('k','tab:blue','tab:orange','tab:green','tab:red')))
    for j,band in enumerate(('W1','W2')):
        ax=axes[0,j]
        ax.errorbar(obs['mjd'],1000*obs['F_'+band+'_Jy'],yerr=1000*obs['sigma_'+band+'_Jy'],fmt='o',ms=3,color='0.5',label='Data')
        for point in plan['points']:
            level=point['level']
            if level not in rows: continue
            path=ROOT/'results/v48_paper_final/models'/level/point['model_id']/'response.csv'
            r=np.genfromtxt(path,delimiter=',',names=True)
            ax.plot(r['time_obs_days']+plan['reference']['t0_mjd'],1000*r[band+'_Jy']+np.atleast_1d(obs['baseline_'+band+'_mJy'])[0],color=colors[level],label=level,lw=1)
        ax.set(xlabel='MJD',ylabel=band+' total flux [mJy]',xlim=(min(obs['mjd'])-150,max(obs['mjd'])+150),title='Common reference t0')
        ax.legend(fontsize=8)
    names=[p['level'] for p in plan['points'] if p['level'] in rows]
    x=np.arange(len(names))
    ax=axes[1,0]
    for field,label,marker in [('chi2','Profiled t0','o'),('chi2_fixed_reference_t0','Fixed reference t0','s')]:
        ax.plot(x,[rows[n][field]-plan['reference']['chi2'] for n in names],marker+'-',label=label)
    ax.axhline(0,color='0.6',lw=.8);ax.set_xticks(x,names,rotation=15)
    ax.set(ylabel='chi2 - baseline chi2',title='Numerical sensitivity at fixed physical parameters');ax.legend()
    ax=axes[1,1]
    ax.plot(x,[rows[n]['fit']['t0_mjd']-plan['reference']['t0_mjd'] for n in names],'o-')
    ax.axhline(0,color='0.6',lw=.8);ax.set_xticks(x,names,rotation=15)
    ax.set(ylabel='Fitted t0 shift [days]',title='Timing compensation')
    fig.suptitle('Case B reference: individual numerical refinements')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('case_b_resolution485_diagnostic.'+ext),dpi=170)
    plt.close(fig)


def review_summary(plan, results):
    return {'scope':plan['scope'], 'resolution_convergence_certified':False,
            'comparisons':[{'level':r['level'], 'delta_chi2_profiled_t0':r['chi2']-plan['reference']['chi2'],
            'delta_chi2_fixed_t0':r['chi2_fixed_reference_t0']-plan['reference']['chi2'],
            't0_shift_days':r['fit']['t0_mjd']-plan['reference']['t0_mjd']} for r in results]}


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
    out = root / 'results/v48_case_b_resolution485'
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        print(json.dumps(read(out/'report.json'), indent=2) if (out/'report.json').exists() else 'Not started')
        return
    if a.stop:
        (out/'STOP').touch(); print('STOP requested for this validation only'); return
    plan = json.loads(PLAN_JSON)
    points = plan['points']
    if len(points) != 5 or len({p['id'] for p in points}) != 5:
        raise RuntimeError('Expected exactly 5 distinct fixed configurations')
    if any(ident(p['config']) != p['model_id'] for p in points):
        raise RuntimeError('Configuration hash mismatch')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    locks, finished = [], {}
    modelroot = root/'results/v48_paper_final/models'
    reason = 'preflight'
    error_details = None
    started = time.monotonic()
    try:
        for directory in ('v48_uncertainties', 'v48_parallel_final', 'v48_targeted_profiles',
                          'v48_profile_continue', 'v48_uncertainty_repair', 'v48_response_validation',
                          'v48_profile_final_check', 'v48_case_b_validation', 'v48_case_b_support',
                          'v48_case_b_two_checks', 'v48_case_b_profile_checks', 'v48_case_b_refine438', 'v48_case_b_refine453', 'v48_case_b_targeted468', 'v48_case_b_resolution485'):
            locks.append(acquire(root/'results'/directory/'coordinator.lock'))
        binding = out/'plan_identity.json'
        if binding.exists() and read(binding)['sha256'] != signature:
            raise RuntimeError('Saved plan differs; preserve results for review')
        write(binding, {'sha256': signature, 'fixed_models': 5})
        sys.path.insert(0, str(root/'V48_PAPER_FINAL'))
        import run_final as rf
        if rf.ROOT.resolve() != root.resolve(): raise RuntimeError('Wrong project root')
        print('PREFLIGHT unchanged solver/data/runtime', flush=True)
        if rf.LEVELS != {'fine': {}, 'angular': {'nmu': 12, 'nrays': 512, 'nobserver': 1024}, 'time': {'dt': 15.0}, 'radial': {'nr': 12}, 'spectral': {'nwave': 288}}: raise RuntimeError('Numerical refinement definitions changed')
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
                           'config': read(modelroot/'fine'/plan['reference']['id']/'config.json'),
                           'predicted_chi2': plan['reference']['chi2'],
                           'inside_training_convex_hull': True, 'purpose': 'reference'}
        reference_result = checked_result(modelroot, reference_point, provenance)
        if ident(reference_point['config']) != reference_point['id'] or abs(reference_result['chi2']-plan['reference']['chi2']) > 1e-6:
            raise RuntimeError('Reference configuration or chi-square differs')
        # Scan ONLY the 5 selected directories, not the thousands of archived models.
        for point in points:
            d = modelroot/point['level']/point['model_id']
            if (d/'summary.json').exists():
                finished[point['id']] = checked_result(modelroot, point, provenance)
        print('FIXED MODELS 5; CACHED', len(finished), '; TO RUN', 5-len(finished), flush=True)
        if a.check_only:
            reason = 'preflight_passed'; return
        budget = Budget(out/'total_budget.json', 8)
        print('TOTAL ATTEMPTS REMAINING', budget.remaining, '(4 expected new models + 4 reserve attempts; cap persists)', flush=True)
        (out/'STOP').unlink(missing_ok=True)
        # Rebuild the checkpoint from validated files; never trust stale completion flags.
        write(out/'checkpoint.json', {'finished': finished})
        def command(point):
            d = modelroot/point['level']/point['model_id']; d.mkdir(parents=True, exist_ok=True)
            write(d/'config.json', point['config'])
            return [sys.executable, '-u', str(root/'V48_PAPER_FINAL'/('_solver_'+point['level']+'.py')),
                    '--resolution', 'fine', '--config', str(d/'config.json'), '--output', str(d)]
        def collect(point): return checked_result(modelroot, point, provenance)
        remaining = a.minutes*60-(time.monotonic()-started)
        if remaining <= 0:
            reason = 'time_limit_before_RT'
        else:
            finished, reason = schedule(points, out, command, collect, a.workers,
                                        remaining, 5, budget=budget)
    except Exception:
        error_details = traceback.format_exc()
        reason = 'error_requires_review'
        raise
    finally:
        # A failed lock acquisition must not overwrite another running coordinator's report.
        if len(locks) == 15:
            if 'provenance' in locals():
                for point in points:
                    try: finished[point['id']] = checked_result(modelroot, point, provenance)
                    except (OSError, ValueError, KeyError): pass
            center = next((r['chi2'] for r in finished.values() if r['name']=='fine'), None)
            for r in finished.values():
                r['delta_from_archived_reference'] = r['chi2']-plan['reference']['chi2']
                r['delta_from_validated_center'] = None if center is None else r['chi2']-center
                r['delta_from_best_checked_reference'] = r['chi2']-min(plan['reference']['chi2'], center if center is not None else plan['reference']['chi2'])
            report = {'status': reason, 'completed': len(finished), 'planned': 5,
                      'validation_complete': len(finished)==5, 'confidence_intervals_certified': False,
                      'error_details': error_details, 'elapsed_minutes': (time.monotonic()-started)/60, 'results': list(finished.values()),
                      'note': plan['scope'], 'local_review': review_summary(plan, list(finished.values()))}
            try:
                make_diagnostic(out, plan, list(finished.values()))
            except Exception:
                report['figure_error'] = traceback.format_exc()
            write(out/'report.json', report)
            dest = out/'SEND_CASE_B_RESOLUTION485.zip'
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('validation_plan.json', json.dumps(plan, indent=2))
                for filename in ('report.json', 'checkpoint.json', 'total_budget.json', 'plan_identity.json'):
                    p = out/filename
                    if p.exists(): z.write(p, filename)
                for p in out.glob('*.log'): z.write(p, 'logs/'+p.name)
                z.write(Path(__file__), 'run_case_b_resolution485.py')
                for ext in ('png','pdf'):
                    p=out/('case_b_resolution485_diagnostic.'+ext)
                    if p.exists(): z.write(p,p.name)
                for point in points:
                    for filename in ('summary.json', 'provenance.json', 'config.json', 'response.csv'):
                        p = modelroot/point['level']/point['model_id']/filename
                        if p.exists(): z.write(p, 'models/'+point['id']+'/'+filename)
            print('FINISHED', reason, 'SAVED', len(finished), '/5', flush=True)
            print('SEND', dest, flush=True)
        for lock in reversed(locks): lock.close()

PLAN_JSON = '{"reference": {"id": "3152034933561190c5b5", "chi2": 58.923309117058764, "t0_mjd": 57065.24758756976}, "points": [{"name": "fine", "id": "fine_3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "fixed physical reference: numerical resolution check", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true, "level": "fine", "override": {}, "model_id": "3152034933561190c5b5"}, {"name": "angular", "id": "angular_3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "fixed physical reference: numerical resolution check", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true, "level": "angular", "override": {"nmu": 12, "nrays": 512, "nobserver": 1024}, "model_id": "3152034933561190c5b5"}, {"name": "time", "id": "time_3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "fixed physical reference: numerical resolution check", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true, "level": "time", "override": {"dt": 15.0}, "model_id": "3152034933561190c5b5"}, {"name": "radial", "id": "radial_3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "fixed physical reference: numerical resolution check", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true, "level": "radial", "override": {"nr": 12}, "model_id": "3152034933561190c5b5"}, {"name": "spectral", "id": "spectral_3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "fixed physical reference: numerical resolution check", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true, "level": "spectral", "override": {"nwave": 288}, "model_id": "3152034933561190c5b5"}], "scope": "Fixed physical reference; individual resolution sensitivity checks. Not endpoint convergence or confidence certification."}'

if __name__ == "__main__": main()
