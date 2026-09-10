#!/usr/bin/env python3
"""Freeze accepted ADNI C2 MRI/PET pairs without recomputing images."""
import csv,hashlib,json,stat,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'data/derived/adni/stage_c2';M=R/'data/derived/adni/adni_c2_manifest.csv';Q=R/'reports/adni_stage_c2_qc.csv';L=B/'frozen_pairs_sha256.csv';F=B/'FROZEN.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 if F.exists():
  s=json.loads(F.read_text());rows=list(csv.DictReader(L.open()));bad=[r['path'] for r in rows if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']]
  if bad:raise RuntimeError(f'Frozen C2 integrity failure: {bad[:5]}')
  print(json.dumps({'status':'ALREADY_FROZEN_VALID','pairs':s['pairs'],'files':len(rows)},indent=2));return
 m=list(csv.DictReader(M.open()));q=list(csv.DictReader(Q.open()))
 if len(m)!=54 or len({r['subject_id'] for r in m})!=54 or len(q)!=54:raise RuntimeError('Expected 54 subjects and QC rows')
 if any(r['status']=='FAIL' for r in q):raise RuntimeError('Cannot freeze with QC FAIL')
 rows=[]
 for r in sorted(m,key=lambda x:x['subject_id']):
  for mod,key in [('MRI','mri_mni_path'),('PET','pet_mni_path')]:
   p=Path(r[key])
   if not p.is_file():raise FileNotFoundError(p)
   rows.append({'subject_id':r['subject_id'],'safe_subject_id':r['safe_subject_id'],'group':r['group'],'modality':mod,'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size,'mode_before':oct(stat.S_IMODE(p.stat().st_mode))})
 if len(rows)!=108:raise RuntimeError(f'Expected 108 files, got {len(rows)}')
 with L.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for r in rows:
  p=Path(r['path']);p.chmod(stat.S_IMODE(p.stat().st_mode)&~(stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH))
 report=R/'reports/adni_stage_c2_complete_report.md';pipeline=R/'scripts/run_adni_stage_c2.py'
 s={'stage':'C2','status':'FROZEN_QC_ACCEPTED','subjects':54,'pairs':54,'files':108,'modalities':{'MRI':54,'PET':54},'qc_counts':{x:sum(r['status']==x for r in q) for x in ('PASS','WARNING_ACCEPTED','WARNING','FAIL')},'ledger':str(L.relative_to(R)),'ledger_sha256':sha(L),'manifest_sha256':sha(M),'qc_sha256':sha(Q),'report_sha256':sha(report),'pipeline_sha256':sha(pipeline),'write_protection':'NIfTI write bits removed; C2 pipeline refuses execution while FROZEN.json exists','next_stage_started':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 F.write_text(json.dumps(s,indent=2)+'\n');L.chmod(stat.S_IMODE(L.stat().st_mode)&~(stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
