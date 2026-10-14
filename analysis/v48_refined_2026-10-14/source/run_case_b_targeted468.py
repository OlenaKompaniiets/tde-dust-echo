"""Evaluate one immutable list of 19 RT configurations (2 cached, 17 new expected); no parameter search."""
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
    rows = {r['name']:r for r in results}
    lookup = {p['name']:p for p in plan['points']}
    ref = plan['reference']['chi2']
    center = lookup['center']['config']
    keys = ['eta','tau_visc','tau_C','tau_E','theta','inc']
    fig,axes = plt.subplots(3,3,figsize=(13,10))
    for ax,key in zip(axes.flat,keys):
        samples = [(center[key],0.)]
        for side in ('minus','plus'):
            name='poll_'+key+'_'+side
            if name in rows: samples.append((lookup[name]['config'][key],rows[name]['chi2']-ref))
        samples.sort()
        ax.plot([v[0] for v in samples],[v[1] for v in samples],'o-',color='tab:blue')
        ax.axhline(0,color='0.5',ls='--',lw=.8)
        ax.set(xlabel=key,ylabel=r'$\chi^2-\chi^2_{ref}$',title='Local coordinate probes')
    ax=axes.flat[6]
    samples=[]
    for name,x in [('valley_previous',-1),('center',0),('valley_forward',1)]:
        if name in rows:samples.append((x,rows[name]['chi2']-ref))
    if samples: ax.plot([v[0] for v in samples],[v[1] for v in samples],'o-')
    ax.axhline(0,color='0.5',ls='--',lw=.8)
    ax.set(xlabel='Previous (-1), current (0), forward (+1)',ylabel=r'$\chi^2-\chi^2_{ref}$',title='Correlated direction')
    for ax,name in zip(list(axes.flat)[7:],['eta_upper','inc_lower']):
        old=plan['original_endpoints'][name]
        xs=[0];ys=[old['chi2']-ref]
        for x,suffix in [(1,'_nuisance'),(2,'_targeted')]:
            if name+suffix in rows:xs.append(x);ys.append(rows[name+suffix]['chi2']-ref)
        ax.scatter(xs,ys,s=45)
        ax.set_xticks([0,1,2],['Previous RT','New nuisance','New crossing'],rotation=20)
        ax.axhline(1,color='k',ls='--',lw=.8)
        ax.set(ylabel=r'$\chi^2-\chi^2_{ref}$',title=name)
    fig.suptitle('Direct RT checks: local probes and targeted endpoints\nFinite probes do not certify a minimum or confidence intervals.')
    fig.tight_layout(rect=[0,0,1,.94])
    for ext in ('png','pdf'):fig.savefig(out/('case_b_targeted468_diagnostic.'+ext),dpi=170)
    plt.close(fig)

def review_summary(plan, results):
    rows={r['name']:r for r in results}
    ref=plan['reference']['chi2']
    names=[p['name'] for p in plan['points'] if p['name'].startswith('poll_') or p['name'].startswith('valley_')]
    probes=[rows[n] for n in names if n in rows]
    tolerance=plan['improvement_review_tolerance_chi2']
    lower=[{'name':r['name'],'chi2':r['chi2'],'improvement':ref-r['chi2']} for r in probes if r['chi2']<ref-tolerance]
    endpoints={}
    for name,old in plan['original_endpoints'].items():
        item={'previous_chi2':old['chi2']}
        if name+'_nuisance' in rows:
            item['nuisance_reoptimized_chi2']=rows[name+'_nuisance']['chi2']
            item['nuisance_improvement_at_same_parameter']=old['chi2']-rows[name+'_nuisance']['chi2']
        if name+'_targeted' in rows:
            item['targeted_delta_chi2']=rows[name+'_targeted']['chi2']-ref
            item['targeted_delta_minus_one']=item['targeted_delta_chi2']-1
        endpoints[name]=item
    return {'probes_completed':len(probes),'probes_planned':len(names),
            'improvement_review_tolerance_chi2':tolerance,
            'lower_probe_points':lower,'targeted_endpoints':endpoints,
            'interpretation':'A 0.01 chi2 improvement is a review trigger, not a significance threshold. No local/global minimum or intervals certified.'}


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
    out = root / 'results/v48_case_b_targeted468'
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        print(json.dumps(read(out/'report.json'), indent=2) if (out/'report.json').exists() else 'Not started')
        return
    if a.stop:
        (out/'STOP').touch(); print('STOP requested for this validation only'); return
    plan = json.loads(PLAN_JSON)
    points = plan['points']
    if len(points) != 19 or len({p['id'] for p in points}) != 19:
        raise RuntimeError('Expected exactly 19 distinct fixed configurations')
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
                          'v48_case_b_two_checks', 'v48_case_b_profile_checks', 'v48_case_b_refine438', 'v48_case_b_refine453', 'v48_case_b_targeted468'):
            locks.append(acquire(root/'results'/directory/'coordinator.lock'))
        binding = out/'plan_identity.json'
        if binding.exists() and read(binding)['sha256'] != signature:
            raise RuntimeError('Saved plan differs; preserve results for review')
        write(binding, {'sha256': signature, 'fixed_models': 19})
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
        # Scan ONLY the 19 selected directories, not the thousands of archived models.
        for point in points:
            d = modelroot/point['id']
            if (d/'summary.json').exists():
                finished[point['id']] = checked_result(modelroot, point, provenance)
        print('FIXED MODELS 19; CACHED', len(finished), '; TO RUN', 19-len(finished), flush=True)
        if a.check_only:
            reason = 'preflight_passed'; return
        budget = Budget(out/'total_budget.json', 21)
        print('TOTAL ATTEMPTS REMAINING', budget.remaining, '(17 expected new models + 4 reserve attempts; cap persists)', flush=True)
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
                                        remaining, 19, budget=budget)
    except Exception:
        error_details = traceback.format_exc()
        reason = 'error_requires_review'
        raise
    finally:
        # A failed lock acquisition must not overwrite another running coordinator's report.
        if len(locks) == 14:
            if 'provenance' in locals():
                for point in points:
                    try: finished[point['id']] = checked_result(modelroot, point, provenance)
                    except (OSError, ValueError, KeyError): pass
            center = next((r['chi2'] for r in finished.values() if r['name']=='center'), None)
            for r in finished.values():
                r['delta_from_archived_reference'] = r['chi2']-plan['reference']['chi2']
                r['delta_from_validated_center'] = None if center is None else r['chi2']-center
                r['delta_from_best_checked_reference'] = r['chi2']-min(plan['reference']['chi2'], center if center is not None else plan['reference']['chi2'])
            report = {'status': reason, 'completed': len(finished), 'planned': 19,
                      'validation_complete': len(finished)==19, 'confidence_intervals_certified': False,
                      'error_details': error_details, 'elapsed_minutes': (time.monotonic()-started)/60, 'results': list(finished.values()),
                      'note': plan['scope'], 'local_review': review_summary(plan, list(finished.values()))}
            try:
                make_diagnostic(out, plan, list(finished.values()))
            except Exception:
                report['figure_error'] = traceback.format_exc()
            write(out/'report.json', report)
            dest = out/'SEND_CASE_B_TARGETED468.zip'
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('validation_plan.json', json.dumps(plan, indent=2))
                for filename in ('report.json', 'checkpoint.json', 'total_budget.json', 'plan_identity.json'):
                    p = out/filename
                    if p.exists(): z.write(p, filename)
                for p in out.glob('*.log'): z.write(p, 'logs/'+p.name)
                z.write(Path(__file__), 'run_case_b_targeted468.py')
                for ext in ('png','pdf'):
                    p=out/('case_b_targeted468_diagnostic.'+ext)
                    if p.exists(): z.write(p,p.name)
                for point in points:
                    for filename in ('summary.json', 'provenance.json', 'config.json', 'response.csv'):
                        p = modelroot/point['id']/filename
                        if p.exists(): z.write(p, 'models/'+point['id']+'/'+filename)
            print('FINISHED', reason, 'SAVED', len(finished), '/19', flush=True)
            print('SEND', dest, flush=True)
        for lock in reversed(locks): lock.close()

PLAN_JSON = '{"reference": {"id": "3152034933561190c5b5", "chi2": 58.923309117058764}, "points": [{"name": "center", "id": "3152034933561190c5b5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "cached best direct RT reference", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_eta_minus", "id": "658eddfaa2b38259964b", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1803643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "eta", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_eta_plus", "id": "875c5073667c9fc76cc1", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1843643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "eta", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_visc_minus", "id": "fbe0719003655381251e", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 240.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_visc", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_visc_plus", "id": "e7fb8bba4a44cf006584", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 250.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_visc", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_C_minus", "id": "aee4d5252f716c03a903", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38010950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_C", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_C_plus", "id": "29ada3f71628855ddbac", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.39010950854710486, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_C", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_E_minus", "id": "142ad0bfc2b3efdaf2af", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.9519514078090119, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_E", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_tau_E_plus", "id": "fa22bd0f17fb4bf66464", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 2.0119514078090117, "theta": 46.90535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "tau_E", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_theta_minus", "id": "307ed481b417a848c9fc", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.40535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "theta", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_theta_plus", "id": "c8003ba4607ca75ef3cd", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 47.40535162653177, "inc": 42.662643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "theta", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_inc_minus", "id": "97b736d2545c8c5ab0d6", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 42.162643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "inc", "side": "minus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "poll_inc_plus", "id": "48e665a972f42b9377e2", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 245.24509766270944, "eta": 0.1823643760379557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.38510950854710485, "tau_E": 1.981951407809012, "theta": 46.90535162653177, "inc": 43.162643087378456, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "local coordinate probe; t0 optimized by solver", "parameter": "inc", "side": "plus", "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "valley_previous", "id": "6dc9df8686595bfdf766", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 236.6166417997312, "eta": 0.17999108814818693, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3789466566676859, "tau_E": 1.8269627667339157, "theta": 46.419649752174, "inc": 42.509672086276325, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "cached previous RT center; correlated direction probe", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "valley_forward", "id": "39f014c0b811a0b0ee1d", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 253.87355352568767, "eta": 0.18473766392772445, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3912723604265238, "tau_E": 2.1369400488841084, "theta": 47.391053500889534, "inc": 42.81561408848059, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "correlated probe continuing previous-to-current center displacement", "parameter": null, "side": null, "predicted_chi2": null, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "eta_upper_nuisance", "id": "b1681bb4c57d090f4600", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 260.239703477158, "eta": 0.19138098436683557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.401759221839645, "tau_E": 2.4727730088669744, "theta": 48.057200156355286, "inc": 43.7635156247388, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "nuisance reoptimization at the previously RT-checked fixed parameter", "parameter": "eta", "side": "upper", "predicted_chi2": 59.753081313952386, "fixed_t0": null, "inside_training_convex_hull": false}, {"name": "eta_upper_targeted", "id": "80d247e54420591e1dc5", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 262.86737222009833, "eta": 0.19231848436683557, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.40368608559477737, "tau_E": 2.5030415411994333, "theta": 48.20682356477562, "inc": 43.96116777063333, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "targeted emulator profile crossing at direct reference chi2+1", "parameter": "eta", "side": "upper", "predicted_chi2": 59.926282565422255, "fixed_t0": null, "inside_training_convex_hull": false}, {"name": "inc_lower_nuisance", "id": "2cb061b07b349ef5df81", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 233.65068513171886, "eta": 0.17485156566625726, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.3639998517884276, "tau_E": 1.6309323112800813, "theta": 45.0400463874, "inc": 40.248774924880905, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "nuisance reoptimization at the previously RT-checked fixed parameter", "parameter": "inc", "side": "lower", "predicted_chi2": 59.75721036186801, "fixed_t0": null, "inside_training_convex_hull": true}, {"name": "inc_lower_targeted", "id": "0f556880ea72929f57ba", "config": {"stars": "input/m0.7_t1.0/0.800.dat", "tau_visc": 233.81014656679233, "eta": 0.1745122289074341, "compact": [0.25, 0.5, 0], "extended": [1, 2, 2], "tau_C": 0.36293266418182835, "tau_E": 1.6301342185546537, "theta": 44.94954411511382, "inc": 39.990962424880905, "composition": "graphite", "amin": 0.1, "amax": 1.0, "q": 3.5, "density": 1000000000.0, "dt": 10.0, "source_end_days": 6500.0}, "purpose": "targeted emulator profile crossing at direct reference chi2+1", "parameter": "inc", "side": "lower", "predicted_chi2": 59.92100230105673, "fixed_t0": null, "inside_training_convex_hull": true}], "n_models_training": 468, "coordinate_steps": {"eta": 0.002, "tau_visc": 5.0, "tau_C": 0.005000000000000001, "tau_E": 0.03, "theta": 0.5, "inc": 0.5}, "targeted_diagnostics": {"eta_upper": {"original_value": 0.19138098436683557, "original_delta_chi2": 0.846149935293262, "new_value": 0.19231848436683557, "predicted_delta_from_RT_reference": 1.0029734483634911, "inside_hull": false, "fit": {"v": [0.9954108328879877, 0.7048909822955564, 0.7430630819069017, 3.4739342226028094, 0.5205887752975394, 0.5194098733019513, -0.7515563708435704], "chi2": 59.926282565422255, "success": true, "optimality": 1.9962009944428154e-05, "nfev": 25}, "nodes": [{"v": [0.9016608328879877, 0.5997842325779416, 0.6659885317016073, 3.272144007053084, 0.46073941192940787, 0.4403490149441393, -0.658048868565001], "chi2": 59.753081313952386, "success": true, "optimality": 5.698125130228815e-05, "nfev": 23}, {"v": [0.8016608328879877, 0.4891595732029742, 0.6052585792489594, 3.08681113114429, 0.3787922036197795, 0.3684181901542597, -0.5621000422737796], "chi2": 59.610419076086686, "success": true, "optimality": 1.2953661413732551e-05, "nfev": 32}, {"v": [1.0016608328879877, 0.7169162037142549, 0.7511653288665407, 3.4984417695246792, 0.5269549138682689, 0.5222786961268313, -0.7588668025721124], "chi2": 59.938081027755814, "success": true, "optimality": 3.4223354962001314e-05, "nfev": 25}, {"v": [0.9516608328879876, 0.6381516066557892, 0.6944635567925677, 3.349440435697665, 0.48816092375218906, 0.4871475338841891, -0.699760482938674], "chi2": 59.84240250279575, "success": true, "optimality": 1.6647144761612364e-05, "nfev": 20}, {"v": [0.9766608328879877, 0.6745958295335308, 0.7211615008020502, 3.4113652526608833, 0.5041772598996156, 0.5077804731211943, -0.7311753362421224], "chi2": 59.890500700987374, "success": true, "optimality": 0.00016311482148249673, "nfev": 33}, {"v": [0.9891608328879877, 0.6938868010364843, 0.7353630847803073, 3.4513110408646055, 0.5146972066173156, 0.5160020757322097, -0.7445216089532161], "chi2": 59.91439971478986, "success": true, "optimality": 0.00030994223216012333, "nfev": 41}, {"v": [0.9954108328879877, 0.7048909822955564, 0.7430630819069017, 3.4739342226028094, 0.5205887752975394, 0.5194098733019513, -0.7515563708435704], "chi2": 59.926282565422255, "success": true, "optimality": 1.9962009944428154e-05, "nfev": 25}]}, "inc_lower": {"original_value": 40.248774924880905, "original_delta_chi2": 1.1448339426118395, "new_value": 39.990962424880905, "predicted_delta_from_RT_reference": 0.997693183997967, "inside_hull": true, "fit": {"v": [-0.7852147130521603, -0.45739804383668475, -0.8870737746110605, -2.345447928362389, -0.7823230045671812, -1.0686722649990203, 0.5517040626584085], "chi2": 59.92100230105673, "success": true, "optimality": 2.9599223287891085e-05, "nfev": 27}, "nodes": [{"v": [-0.7512810371698422, -0.4637765012396231, -0.8443862703470901, -2.340127310192871, -0.7461220956527097, -0.9655472649990202, 0.5463818728824034], "chi2": 59.75721036186801, "success": true, "optimality": 2.0543840977604984e-05, "nfev": 24}, {"v": [-0.5526530005322361, -0.3498976771237111, -0.6376864766768419, -1.1699946366622778, -0.47706436196854285, -0.8655472649990202, 0.42873572544325905], "chi2": 59.65084449575927, "success": true, "optimality": 0.00010204765714636835, "nfev": 19}, {"v": [-0.7842483407747187, -0.4576260095425836, -0.8857975254683939, -2.3456696363089264, -0.7812731628401388, -1.0655472649990203, 0.5515899540864844], "chi2": 59.91565660138503, "success": true, "optimality": 4.3228797447775416e-05, "nfev": 25}, {"v": [-0.49646626768418717, -0.3274499335383153, -0.575812599982521, -1.065512950992451, -0.4094904116189953, -0.7655472649990203, 0.39999947003706243], "chi2": 59.492970165163804, "success": true, "optimality": 1.618491445197736e-05, "nfev": 22}, {"v": [-0.8156964135401986, -0.45218501479099055, -0.928033951729317, -2.341520685053416, -0.817338170217708, -1.1655472649990202, 0.5562585700648377], "chi2": 60.09675132126273, "success": true, "optimality": 1.6581467314942093e-06, "nfev": 24}, {"v": [-0.7999085926660057, -0.454535995008887, -0.9066052925058093, -2.3431170207219307, -0.7988347183055843, -1.1155472649990203, 0.5537366570980945], "chi2": 60.00368613477931, "success": true, "optimality": 2.3505011157775695e-05, "nfev": 28}, {"v": [-0.7920408919350233, -0.4559513076865854, -0.8961030905335764, -2.344188389022292, -0.7898759710820775, -1.0905472649990204, 0.5525943374821759], "chi2": 59.959015775014514, "success": true, "optimality": 1.2992183956940598e-05, "nfev": 22}, {"v": [-0.7881315359381343, -0.45675158374799557, -0.8909232199077859, -2.344863517582912, -0.7855203544691352, -1.0780472649990203, 0.5520719014460715], "chi2": 59.937167925663964, "success": true, "optimality": 5.330487832574667e-06, "nfev": 26}, {"v": [-0.7861848218499563, -0.4571774795446622, -0.8883527951735706, -2.34524289680436, -0.7833788603919348, -1.0717972649990202, 0.5518244257036642], "chi2": 59.926369542977966, "success": true, "optimality": 4.335353219213834e-06, "nfev": 27}, {"v": [-0.7852147130521603, -0.45739804383668475, -0.8870737746110605, -2.345447928362389, -0.7823230045671812, -1.0686722649990203, 0.5517040626584085], "chi2": 59.92100230105673, "success": true, "optimality": 2.9599223287891085e-05, "nfev": 27}]}}, "original_endpoints": {"eta_upper": {"id": "c3a02c18d29bd054047c", "name": "eta_upper", "chi2": 59.769459052352026, "chi2_profiled_t0": 59.769459052352026, "fit": {"chi2_total": 59.769459052352026, "chi2_w1": 19.827011487577618, "chi2_w2": 39.942447564774405, "t0_mjd": 57051.884238301485, "t0_boundary": false}, "fixed_t0": null, "parameter": "eta", "side": "upper", "prediction_error": -0.08113270978078901, "inside_training_convex_hull": false, "purpose": "provisional profile endpoint; direct RT check", "delta_from_archived_reference": 0.8419121418801936, "delta_from_validated_center": 0.846149935293262, "delta_from_best_checked_reference": 0.846149935293262}, "inc_lower": {"id": "a5849a0286009a366954", "name": "inc_lower", "chi2": 60.068143059670604, "chi2_profiled_t0": 60.068143059670604, "fit": {"chi2_total": 60.068143059670604, "chi2_w1": 18.9682990005807, "chi2_w2": 41.099844059089904, "t0_mjd": 57067.100435085835, "t0_boundary": false}, "fixed_t0": null, "parameter": "inc", "side": "lower", "prediction_error": 0.21519984865428654, "inside_training_convex_hull": false, "purpose": "provisional profile endpoint; direct RT check", "delta_from_archived_reference": 1.140596149198771, "delta_from_validated_center": 1.1448339426118395, "delta_from_best_checked_reference": 1.1448339426118395}}, "improvement_review_tolerance_chi2": 0.01, "scope": "Finite local probes and two RT endpoint checks. No global minimum or confidence certification."}'

if __name__ == "__main__": main()
