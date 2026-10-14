"""Evaluate one immutable list of 13 RT configurations; no parameter search."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
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
    return {'id': point['id'], 'name': point['name'], 'chi2': chi, 'fit': s['fit'],
            'purpose': point.get('purpose'), 'prediction_error': None if point['predicted_chi2'] is None else chi-point['predicted_chi2'],
            'inside_training_convex_hull': point['inside_training_convex_hull']}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, choices=range(1, 13), default=4)
    ap.add_argument('--minutes', type=float, default=60)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--stop', action='store_true')
    a = ap.parse_args()
    if a.minutes <= 0: ap.error('--minutes must be positive')
    root = HERE.parent
    out = root / 'results/v48_case_b_validation'
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        print(json.dumps(read(out/'report.json'), indent=2) if (out/'report.json').exists() else 'Not started')
        return
    if a.stop:
        (out/'STOP').touch(); print('STOP requested for this validation only'); return
    plan = read(HERE/'validation_plan.json')
    points = plan['points']
    if len(points) != 13 or len({p['id'] for p in points}) != 13:
        raise RuntimeError('Expected exactly 13 distinct fixed configurations')
    if any(ident(p['config']) != p['id'] for p in points):
        raise RuntimeError('Configuration hash mismatch')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    locks, finished = [], {}
    modelroot = root/'results/v48_paper_final/models/fine'
    reason = 'preflight'
    started = time.monotonic()
    try:
        for directory in ('v48_uncertainties', 'v48_parallel_final', 'v48_targeted_profiles',
                          'v48_profile_continue', 'v48_uncertainty_repair', 'v48_response_validation',
                          'v48_profile_final_check', 'v48_case_b_validation'):
            locks.append(acquire(root/'results'/directory/'coordinator.lock'))
        binding = out/'plan_identity.json'
        if binding.exists() and read(binding)['sha256'] != signature:
            raise RuntimeError('Saved plan differs; preserve results for review')
        write(binding, {'sha256': signature, 'fixed_models': 13})
        sys.path.insert(0, str(root/'V48_PAPER_FINAL'))
        import run_final as rf
        if rf.ROOT.resolve() != root.resolve(): raise RuntimeError('Wrong project root')
        print('PREFLIGHT unchanged solver/data/runtime', flush=True)
        rf.prepare()
        provenance = read(HERE/'seed.json')['provenance']
        for name, expected in provenance['hashes'].items():
            pieces = name.replace('\\', '/').split('/UGC11487_V4_FIXED/', 1)
            if len(pieces) != 2: raise RuntimeError('Unknown provenance path: '+name)
            path = root/pieces[1]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Source/data mismatch: '+str(path))
        if platform.python_version() != provenance['python'] or np.__version__ != provenance['numpy']:
            raise RuntimeError('Python/NumPy differ from the archived calculation; no RT launched')
        # Scan ONLY the 13 selected directories, not the thousands of archived models.
        for point in points:
            d = modelroot/point['id']
            if (d/'summary.json').exists():
                finished[point['id']] = checked_result(modelroot, point, provenance)
        print('FIXED MODELS 13; CACHED', len(finished), '; TO RUN', 13-len(finished), flush=True)
        if a.check_only:
            reason = 'preflight_passed'; return
        budget = Budget(out/'total_budget.json', 18)
        print('TOTAL ATTEMPTS REMAINING', budget.remaining, '(13 targets + at most 5 retry attempts)', flush=True)
        (out/'STOP').unlink(missing_ok=True)
        # Rebuild the checkpoint from validated files; never trust stale completion flags.
        write(out/'checkpoint.json', {'finished': finished})
        def command(point):
            d = modelroot/point['id']; d.mkdir(parents=True, exist_ok=True)
            write(d/'config.json', point['config'])
            return [sys.executable, '-u', str(root/'V48_PAPER_FINAL/_solver_fine.py'),
                    '--resolution', 'fine', '--config', str(d/'config.json'), '--output', str(d)]
        def collect(point): return checked_result(modelroot, point, provenance)
        finished, reason = schedule(points, out, command, collect, a.workers,
                                    max(.01, a.minutes*60-(time.monotonic()-started)),
                                    13, budget=budget)
    except Exception:
        reason = 'error_requires_review'
        raise
    finally:
        # A failed lock acquisition must not overwrite another running coordinator's report.
        if len(locks) == 8:
            if 'provenance' in locals():
                for point in points:
                    try: finished[point['id']] = checked_result(modelroot, point, provenance)
                    except (OSError, ValueError, KeyError): pass
            center = next((r['chi2'] for r in finished.values() if r['name']=='center'), None)
            for r in finished.values():
                r['delta_from_archived_reference'] = r['chi2']-plan['reference']['chi2']
                r['delta_from_validated_center'] = None if center is None else r['chi2']-center
            report = {'status': reason, 'completed': len(finished), 'planned': 13,
                      'validation_complete': len(finished)==13, 'confidence_intervals_certified': False,
                      'elapsed_minutes': (time.monotonic()-started)/60, 'results': list(finished.values()),
                      'note': 'Fixed-point RT checks only. Nuisance-minimum stationarity, profile crossings and numerical-resolution effects require assessment; no automatic confidence certification.'}
            write(out/'report.json', report)
            try:
                from make_figures import main as draw_figures
                draw_figures()
            except Exception as exc:
                print('PLOTS WARNING', str(exc), flush=True)
            dest = out/'SEND_CASE_B_CHECK_FOR_REVIEW.zip'
            with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(HERE/'validation_plan.json', 'validation_plan.json')
                for filename in ('report.json', 'checkpoint.json', 'total_budget.json', 'plan_identity.json'):
                    p = out/filename
                    if p.exists(): z.write(p, filename)
                for p in out.glob('*.log'): z.write(p, 'logs/'+p.name)
                for pattern in ('*.png','*.pdf'):
                    for p in out.glob(pattern): z.write(p, 'figures/'+p.name)
                for point in points:
                    for filename in ('summary.json', 'provenance.json', 'config.json', 'response.csv'):
                        p = modelroot/point['id']/filename
                        if p.exists(): z.write(p, 'models/'+point['id']+'/'+filename)
            print('FINISHED', reason, 'SAVED', len(finished), '/13', flush=True)
            print('SEND', dest, flush=True)
        for lock in reversed(locks): lock.close()

if __name__ == '__main__': main()
