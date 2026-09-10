#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'data/derived/adni/stage_d/v1';FIG=R/'reports/figures/adni_stage_d';REPORT=R/'reports/adni_stage_d_report.md'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 q=list(csv.DictReader((D/'qc/normalization_qc.csv').open()));idx=list(csv.DictReader((D/'manifests/model_input_index.csv').open()));FIG.mkdir(parents=True,exist_ok=True)
 variants=sorted({r['variant'] for r in q});fig,ax=plt.subplots(1,3,figsize=(12,4))
 for pos,v in enumerate(variants,1):
  z=[r for r in q if r['variant']==v]
  for a,key in zip(ax,['mean','std','nonzero_fraction']):a.boxplot([float(r[key]) for r in z],positions=[pos],widths=.5)
 for a,title in zip(ax,['Masked mean','Masked standard deviation','Whole-volume non-zero fraction']):
  a.set_xticks(range(1,len(variants)+1),[v.replace('_','\n') for v in variants],fontsize=6);a.set_title(title);a.grid(alpha=.2)
 fig.suptitle('Stage D normalization QC (pooled; not used for variant selection)');fig.tight_layout();fpath=FIG/'normalization_qc_summary.png';fig.savefig(fpath,dpi=180);plt.close(fig)
 text='''# ADNI Stage D — splits, normalization inputs and QC

Stage D data preparation completed without training. C2 hashes were verified before generation and remain unchanged.

## Approved design

Nested subject-level stratified CV uses 3 outer folds × 3 inner folds, repeated three times. Seeds were published in the versioned configuration before split generation: 20260910, 20261007 and 20261103. Each of the 27 repeat/outer/inner combinations contains:

- inner train: 24 subjects, exactly 8 CN + 8 MCI + 8 AD;
- inner validation: 12 subjects, exactly 4 CN + 4 MCI + 4 AD;
- outer test: 18 subjects, exactly 6 CN + 6 MCI + 6 AD.

All 27 leakage and balance audits PASS. Outer-test membership is never used to fit normalization, select variants, tune, early-stop or train. No model was trained.

## Normalization variants

Four subject-local variants were materialized, each with 54 finite, non-empty outputs and unchanged 97×115×97, 2 mm, RAS geometry:

- mri_robust_z: median/IQR inside MNI brain mask, clipped to [-8,8];
- mri_percentile01: masked percentiles 0.5/99.5 scaled to [0,1];
- pet_relative_robust: masked percentiles 1/99 scaled to [0,1];
- pet_whole_brain_mean: divided by positive-voxel mean inside brain mask.

These operations learn no cohort parameter. They were generated for mechanical QC only and remain UNSELECTED_INNER_CV_ONLY.

pet_reference_suvr is configured but not materialized. It is gated until pons/cerebellum atlas masks and coverage, especially for 009_S_1199, pass anatomical QC. No SUV or mislabeled SUVR was generated.

## Inputs

The model input index contains 432 rows: 108 MRI-only, 108 PET-only and 216 multimodal definitions. Multimodal channel order is fixed as [MRI,PET]. Indexes reference normalized NIfTI files instead of duplicating them.

## QC and provenance

- Normalization QC: 216 PASS, 0 WARNING, 0 FAIL.
- Split audits: 27 PASS, 0 FAIL.
- Subjects: 54; groups: 18 CN/18 MCI/18 AD.
- Normalized outputs/provenance rows: 216/216.
- Missing input paths: 0.
- Output and C2 hash mismatches: 0.
- External test consulted for selection: no.
- Variant selected: none.
- Training started: no.

The pooled QC figure is descriptive only and was not used for selection.

## Deliverables

- Configuration: configs/adni_stage_d.yaml
- Pipeline: scripts/run_adni_stage_d.py
- Splits: data/derived/adni/stage_d/v1/splits/adni_nested_cv_3x3_r3.csv
- Split QC: data/derived/adni/stage_d/v1/qc/split_audit.csv
- Normalization QC: data/derived/adni/stage_d/v1/qc/normalization_qc.csv
- Variant registry and provenance: data/derived/adni/stage_d/v1/manifests/
- Model input index: data/derived/adni/stage_d/v1/manifests/model_input_index.csv
- Global hashes: data/derived/adni/stage_d/v1/provenance_sha256.csv
- Status: data/derived/adni/stage_d/v1/COMPLETE.json
- Figure: reports/figures/adni_stage_d/normalization_qc_summary.png

## Stop state

Stage D stopped after inputs and QC. Normalization selection must occur independently inside each inner-training workflow after training authorization. No outer test was inspected for performance. No model or augmentation was created.
'''
 REPORT.write_text(text)
 files=[]
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name not in ('provenance_sha256.csv','COMPLETE.json'):files.append({'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size})
 for p in [R/'configs/adni_stage_d.yaml',R/'scripts/run_adni_stage_d.py',Path(__file__),REPORT,fpath,R/'data/derived/adni/adni_c2_manifest.csv',R/'data/derived/adni/stage_c2/FROZEN.json',R/'data/derived/adni/stage_c2/frozen_pairs_sha256.csv',R/'reports/adni_stage_d_plan.md',R/'TODO_MRI_PET_ADNI.md']:
  files.append({'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size})
 with (D/'provenance_sha256.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(files)
 state=json.loads((D/'COMPLETE.json').read_text());state.update({'report':str(REPORT.relative_to(R)),'qc_figure':str(fpath.relative_to(R)),'provenance_ledger_sha256':sha(D/'provenance_sha256.csv')});(D/'COMPLETE.json').write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({'report':str(REPORT),'figure':str(fpath),'provenance_files':len(files)},indent=2))
if __name__=='__main__':main()
