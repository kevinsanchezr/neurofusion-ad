#!/usr/bin/env python3
"""Freeze accepted Stage D without regenerating splits or normalized inputs."""
import csv,hashlib,json,stat,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/derived/adni/stage_d/v1';F=D/'FROZEN.json';L=D/'frozen_stage_d_sha256.csv'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if F.exists():
  rows=list(csv.DictReader(L.open()));bad=[r['path'] for r in rows if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']]
  if bad:raise RuntimeError(f'Stage D frozen hash failure: {bad[:5]}')
  print(json.dumps({'status':'ALREADY_FROZEN_VALID','files':len(rows)},indent=2));return
 state=json.loads((D/'COMPLETE.json').read_text())
 if state.get('status')!='COMPLETE_NO_TRAINING' or state.get('training_started') is not False:raise RuntimeError('Stage D is not eligible to freeze')
 files=[p for p in sorted(D.rglob('*')) if p.is_file() and p.name not in ('FROZEN.json','frozen_stage_d_sha256.csv')]
 files += [R/'configs/adni_stage_d.yaml',R/'scripts/run_adni_stage_d.py',R/'scripts/report_adni_stage_d.py',R/'reports/adni_stage_d_plan.md',R/'reports/adni_stage_d_report.md',R/'TODO_MRI_PET_ADNI.md']
 rows=[{'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size,'mode_before':oct(stat.S_IMODE(p.stat().st_mode))} for p in files]
 with L.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for p in files:
  if D in p.parents:p.chmod(stat.S_IMODE(p.stat().st_mode)&~0o222)
 marker={'stage':'D','status':'FROZEN_COMPLETE_NO_TRAINING','subjects':54,'normalized_nifti':216,'split_design':'3 outer x 3 inner x 3 repeats','seeds':[20260910,20261007,20261103],'files_in_ledger':len(rows),'ledger':str(L.relative_to(R)),'ledger_sha256':sha(L),'training_started':False,'stage_e_started':False,'write_protection':'All files under stage_d/v1 present at freeze have write bits removed; Stage D pipeline refuses execution while FROZEN.json exists','frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 F.write_text(json.dumps(marker,indent=2)+'\n');L.chmod(stat.S_IMODE(L.stat().st_mode)&~0o222);print(json.dumps(marker,indent=2))
if __name__=='__main__':main()
