"""Cache-only local curvature audit. No RT calls; no certified intervals."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, hashlib, csv
from pathlib import Path
import numpy as np
KEYS=['eta','tau_visc','tau_C','tau_E','theta','inc']
SCALE=np.array([.01,25,.025,.15,2.5,2.5])
BOUNDS=np.array([[.1,.22],[100,400],[.2,.8],[1.2,3.4],[30,55],[30,55]])
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w',encoding='utf-8') as f: json.dump(o,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def design(u): return np.column_stack([np.ones(len(u)),u]+[u[:,i]*u[:,j] for i in range(6) for j in range(i,6)])
def fit(u,y):
 D=design(u);b,_,rank,_=np.linalg.lstsq(D,y,rcond=None);H=np.zeros((6,6));n=7
 for i in range(6):
  for j in range(i,6): H[i,j]=H[j,i]=b[n]*(2 if i==j else 1);n+=1
 return b,H,int(rank),float(np.linalg.cond(D))