"""Reproduce Case A using the unchanged archived fine solver, in a new folder."""
from pathlib import Path
import subprocess,sys,argparse,datetime,json
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plots',action='store_true');a=ap.parse_args()
subprocess.run([sys.executable,str(root/'tools/check_install.py')],check=True)
out=root/'results'/('case_a_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
cmd=[sys.executable,'-u',str(root/'V48_LOCAL_PAIR/solver.py'),'--resolution','fine','--config',str(root/'configs/case_a.json'),'--output',str(out)]
if a.plots:cmd+=['--plots']
subprocess.run(cmd,cwd=root,check=True)
s=json.loads((out/'summary.json').read_text());ref=json.loads((root/'examples/case_a/summary.json').read_text())
print('Output:',out); print('chi2:',s['fit']['chi2_total']);print('Difference from archive:',s['fit']['chi2_total']-ref['fit']['chi2_total'])
print('This reports reproduction, not a new optimization. Review numerical differences before accepting a changed solution.')
if a.plots:
 subprocess.run([sys.executable,str(root/'tools/plot_thermal.py'),'--folder',str(out)],cwd=root,check=True)
