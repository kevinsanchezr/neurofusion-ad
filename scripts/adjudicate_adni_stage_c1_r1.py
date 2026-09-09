#!/usr/bin/env python3
"""Final visual/quantitative adjudication and report for Stage C1-R1."""
from pathlib import Path
import csv, hashlib, json
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
BASE=R/'data/derived/adni/stage_c1_r1'; QC=R/'reports/adni_stage_c1_r1_qc.csv'
FIG=R/'reports/figures/adni_stage_c1_r1'; TPL=R/'data/derived/adni/stage_c1/templateflow/tpl-MNI152NLin2009cAsym'
TMRI=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'
SUBS=['005_S_0223','005_S_0222','005_S_0221','009_S_1199','153_S_4172']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def norm(a):
 v=a[np.isfinite(a)&(a!=0)]; lo,hi=np.percentile(v,[1,99]) if v.size else (0,1)
 return np.clip((a-lo)/(hi-lo+1e-12),0,1)
def regfig(sub):
 sd=BASE/'subjects'/f'sub-{sub}'
 t=norm(nib.load(TMRI).get_fdata()); m=norm(nib.load(sd/f'sub-{sub}_T1w_MNI2mm.nii.gz').get_fdata())
 fig,axs=plt.subplots(3,3,figsize=(9,9))
 for col,axis in enumerate([2,1,0]):
  idx=t.shape[axis]//2; a=np.rot90(np.take(t,idx,axis)); b=np.rot90(np.take(m,idx,axis))
  axs[0,col].imshow(a,cmap='gray');axs[0,col].imshow(b,cmap='magma',alpha=.35)
  checker=np.where((np.indices(a.shape).sum(0)//8)%2,a,b);axs[1,col].imshow(checker,cmap='gray')
  axs[2,col].imshow(np.abs(a-b),cmap='inferno',vmin=0,vmax=1)
  for row,title in enumerate(['template/subject overlay','checkerboard','absolute difference']):axs[row,col].set_title(title,fontsize=7);axs[row,col].axis('off')
 fig.suptitle(f'{sub}: MRI→MNI template comparison');fig.tight_layout();fig.savefig(FIG/f'{sub}_mri_mni_template_qc.png',dpi=150);plt.close(fig)
def coverage(sub):
 sd=BASE/'subjects'/f'sub-{sub}'
 mask=nib.load(sd/f'sub-{sub}_SynthStrip_mask.nii.gz').get_fdata()>.5
 before=nib.load(sd/(f'sub-{sub}_PET_frame0_before_motion.nii.gz' if sub!='153_S_4172' else f'sub-{sub}_PET_3D_BRAIN2_native.nii.gz'))
 after=nib.load(sd/f'sub-{sub}_PET_in_T1w.nii.gz').get_fdata()
 # Only post-registration coverage is in the MRI grid and scientifically interpretable.
 nz=after>0
 return float(np.logical_and(mask,nz).sum()/max(mask.sum(),1)), float(np.logical_and(mask,nz).sum()/max(nz.sum(),1))
def main():
 q=pd.read_csv(QC)
 visual={
 '005_S_0223':('FAIL','SynthStrip mask visually acceptable, but MRI→MNI validation fails: transformed-mask Dice 0.341; quantitative and visual evidence are not concordant enough for PASS.'),
 '005_S_0222':('PASS','Mask excludes skull/eyes/neck and preserves frontal/temporal cortex, cerebellum and pons; MRI→MNI and PET→MRI plausible.'),
 '005_S_0221':('PASS','Mask excludes skull/eyes/neck and preserves frontal/temporal cortex, cerebellum and pons; MRI→MNI and PET→MRI plausible.'),
 '009_S_1199':('PASS','Mask and MRI→MNI plausible; limited 47-slice PET coverage remains visible but is not further truncated and PET→MRI improves.'),
 '153_S_4172':('PASS','Mask and registrations plausible. Native/canonical/final orientations are traceable RAS, rigid determinant is positive, and no left-right inversion is detected; protocol warning retained.'),
 }
 for sub in SUBS: regfig(sub)
 q['pet_brain_coverage_after'],q['pet_nonzero_precision_in_brain'] = zip(*[coverage(s) for s in q.subject_id])
 q['visual_status']=[visual[s][0] for s in q.subject_id]
 q['visual_reason']=[visual[s][1] for s in q.subject_id]
 q['preprocessing_status']=['FAIL' if visual[s][0]=='FAIL' or r.quantitative_status=='FAIL' else 'PASS' for s,r in zip(q.subject_id,q.itertuples())]
 q['status']=['FAIL' if p=='FAIL' else ('WARNING' if s=='153_S_4172' else 'PASS') for s,p in zip(q.subject_id,q.preprocessing_status)]
 q['regional_visual_check']='frontal cortex; temporal lobes; cerebellum; pons retained; eyes/skull/neck excluded'
 q['laterality_check']=['not exceptional' if s!='153_S_4172' else 'PASS: no detectable LR inversion; RAS chain and positive rigid determinant' for s in q.subject_id]
 q.to_csv(QC,index=False)
 report=f'''# ADNI Stage C1-R1 — SynthStrip replacement validation

## Decision

**STOP. Stage C2 is not authorized or recommended yet.** Four subjects pass preprocessing; `005_S_0223` fails MRI→MNI validation despite an acceptable SynthStrip mask. Thus the required 5/5 gate is not met. No full-cohort processing, splitting, intensity normalization, or training was run.

## Fixed scope and result

| Subject | Group / role | SynthStrip mask | MRI→MNI | PET→MRI | Final |
|---|---|---|---|---|---|
| 005_S_0223 | CN, standard ECAT7 | PASS | **FAIL** (MNI mask Dice 0.341) | PASS | **FAIL** |
| 005_S_0222 | MCI, standard ECAT7 | PASS | PASS | PASS | PASS |
| 005_S_0221 | AD, standard ECAT7 | PASS | PASS | PASS | PASS |
| 009_S_1199 | MCI, PET 47 slices | PASS | PASS | PASS; limited native coverage retained | PASS |
| 153_S_4172 | AD, ADNI2 PET-DICOM/LAS | PASS | PASS | PASS | PASS + protocol WARNING |

## Official SynthStrip distribution

- Origin: official FreeSurfer `freesurfer/synthstrip:1.8` Docker Hub image.
- Immutable image digest: `sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06`.
- Image ID: `sha256:d1755c427edd40d9b500c1793e96bb559b1d2ab0636e3c060613ffa4079a392b`.
- Model: `/freesurfer/models/synthstrip.1.pt`; SHA-256 `{MODEL_SHA if False else '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33'}`.
- Container: Python 3.10.18, PyTorch 2.1.2+cpu, Surfa 0.6.1, NiBabel 5.3.2.
- Host pipeline: ANTsPyX 0.6.1 (unchanged), Python 3.11.8, NiBabel {nib.__version__}.
- Per-subject command: `docker run --rm --network=none -v <subject-dir>:/data freesurfer/synthstrip:1.8 -i /data/<T1_RAS> -o /data/<stripped> -m /data/<mask> -b 1`.
- The exact commands and stdout/stderr are stored under each subject directory. No `--no-csf` and no subject-specific morphology were used.

## Reprocessing and invalidation

Only native MRI/PET inputs were reused. PET motion correction was recomputed. Previous template-propagated masks, MRI→MNI registrations, PET-in-MNI images, and dependent transforms remain solely in `stage_c1/` as failed-validation artifacts and were not reused. All R1 outputs live in `stage_c1_r1/`.

Pipeline: native MRI → canonical RAS → N4 → SynthStrip on RAS → original binary mask applied to N4 → SyN to MNI152NLin2009cAsym 2 mm. ECAT PET motion was recomputed rigidly and averaged only afterward; the ADNI2 BRAIN 2 PET remained 3D. PET was rigidly registered to the N4/SynthStrip MRI, then PET→MRI and MRI→MNI transforms were applied together in one linear interpolation. Masks used nearest-neighbor interpolation. Output remains 97×115×97 at 2 mm without PET normalization.

## Mask QC

All five SynthStrip masks are finite, have valid affines, one connected component, no isolated voxels, no slice gaps, and brain volumes 1438–1732 ml. Visual review found no evident skull, eyes, neck, or air inclusion and no loss of frontal cortex, temporal lobes, cerebellum, or pons. The MNI mask is not treated as individual ground truth; its overlap is one non-circular registration diagnostic among topology, extent, regional visual review, PET metrics, and transform validity.

## Registration QC and exceptions

PET→MRI NMI and Dice improved in every subject. All rigid determinants are positive. MRI warp Jacobians are finite and strictly positive (non-positive fraction 0). `009_S_1199` retains its lower 47-slice PET coverage without evidence of new truncation. For `153_S_4172`, native-to-RAS provenance, a positive rigid determinant, overlays, and labeled RAS review show no detectable left-right inversion; its ADNI2/PET-DICOM/LAS warning remains independent of preprocessing success.

`005_S_0223` is the stopping condition: SynthStrip itself is visually and topologically acceptable, but the transformed mask has MNI Dice 0.341 versus 0.959–0.972 for the other four. The isolated warped MRI appears centered, but the template overlay/quantitative discordance prevents PASS. ANTsPyNet was not run because the requested fallback is triggered by SynthStrip failure; here the downstream MRI→MNI registration failed.

## Reproducibility and outputs

- Configuration: `configs/adni_stage_c1_r1.yaml`
- Pipeline: `scripts/run_adni_stage_c1_r1.py`
- Adjudication: `scripts/adjudicate_adni_stage_c1_r1.py`
- QC table: `reports/adni_stage_c1_r1_qc.csv`
- Figures: `reports/figures/adni_stage_c1_r1/`
- Masks/transforms/provenance: `data/derived/adni/stage_c1_r1/subjects/`
- Previous-mask comparison: `data/derived/adni/stage_c1_r1/previous_mask_vs_synthstrip.csv`
- Hash inventory: `data/derived/adni/stage_c1_r1/sha256sums.csv`

## Recommendation

Do not start C2. Diagnose and revalidate the general MRI→MNI initialization/registration strategy on the same five cases, without subject-specific tuning. SynthStrip replacement itself is validated on all five, so an ANTsPyNet comparison is not scientifically indicated by this run.
'''
 (R/'reports/adni_stage_c1_r1_skullstrip_validation.md').write_text(report)
 # Refresh hashes after final adjudication.
 files=[]
 for p in sorted(BASE.rglob('*')):
  if p.is_file() and p != BASE/'sha256sums.csv': files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 for p in [QC,R/'configs/adni_stage_c1_r1.yaml',R/'scripts/run_adni_stage_c1_r1.py',Path(__file__),R/'reports/adni_stage_c1_r1_skullstrip_validation.md']:
  files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 with open(BASE/'sha256sums.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['path','sha256','bytes']);w.writerows(files)
 print(q[['subject_id','preprocessing_status','status','visual_status','mni_mask_dice','pet_mri_nmi_before','pet_mri_nmi_after']].to_string(index=False))
 print('Stage C2 started: NO; recommendation: DO NOT START')
if __name__=='__main__':main()
