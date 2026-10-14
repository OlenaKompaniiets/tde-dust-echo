"""Persistent conservative cap: reserve on disk BEFORE starting each RT process."""
import json,os
from pathlib import Path

def atomic_write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

class Budget:
 def __init__(self,path,limit):
  self.path=Path(path)
  if self.path.exists():
   self.data=json.loads(self.path.read_text(encoding='utf-8'))
   if self.data['limit']!=limit:raise RuntimeError('Stored total budget differs; it cannot be reset by a restart.')
  else:self.data={'version':1,'limit':limit,'reservations':[]};atomic_write(self.path,self.data)
 @property
 def remaining(self):return max(0,self.data['limit']-len(self.data['reservations']))
 def reserve(self,model_id):
  if not self.remaining:return False
  self.data['reservations'].append({'attempt':len(self.data['reservations'])+1,'model_id':model_id})
  atomic_write(self.path,self.data);return True
