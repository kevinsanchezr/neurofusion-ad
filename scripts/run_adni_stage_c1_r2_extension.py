#!/usr/bin/env python3
"""Independent six-subject extension for the frozen C1-R2 SyNRA strategy."""
import csv, json
import run_adni_stage_c1_r2 as r2

def main():
 r2.OUT.mkdir(parents=True,exist_ok=True);r2.FIG.mkdir(parents=True,exist_ok=True)
 # Prepare all inputs before registration to keep SynthStrip independent of retained ANTs memory.
 prepared={}
 for sub,g in r2.EXT:
  print('PREPARE',sub,flush=True);prepared[sub]=r2.prepare(sub,True)
 fixed=r2.ants.image_read(str(r2.BRAIN));fmask=r2.ants.image_read(str(r2.MASK));rows=[]
 for sub,g in r2.EXT:
  print('START EXTENSION',sub,'standard_synra',flush=True)
  rows.append(r2.run_strategy(sub,g,'standard_synra',fixed,fmask,*prepared[sub],True))
  print('DONE EXTENSION',sub,rows[-1]['status'],rows[-1]['syn_dice'],flush=True)
 path=r2.R/'reports/adni_stage_c1_r2_extension_qc.csv'
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))));w.writeheader();w.writerows(rows)
 print(json.dumps({'extension_subjects':len(rows),'pass':sum(x['status']=='PASS' for x in rows),'fail':sum(x['status']=='FAIL' for x in rows),'parameters_adjusted_after_selection':False},indent=2))
if __name__=='__main__':main()
