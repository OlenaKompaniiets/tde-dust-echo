"""Read-only installation check; does not run radiative transfer."""
from pathlib import Path
import importlib, json, hashlib, sys
root=Path(__file__).resolve().parents[1]
problems=[]
for name in ('numpy','scipy','pandas','matplotlib','astropy'):
 try:
  m=importlib.import_module(name); print(name,getattr(m,'__version__','installed'))
 except ImportError: problems.append('Missing package: '+name)
expected=json.loads((root/'release_manifest.json').read_text())['inputs']
for rel,h in expected.items():
 p=root/rel
 if not p.exists(): problems.append('Missing file: '+rel)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=h: problems.append('Changed input (review before reproduction): '+rel)
print('STARS histories:',len(list((root/'STARS_library-master/input').rglob('*.dat'))))
for p in problems:print('ERROR',p)
print('PASS' if not problems else 'FAILED')
sys.exit(bool(problems))
