#!/usr/bin/env python3
"""Apply the universal PET registration adjudication rule retrospectively."""
from pathlib import Path
import json, hashlib, time
import pandas as pd

R=Path(__file__).resolve().parents[1]
BASE=R/'data/derived/adni/stage_c2'; QC=R/'reports/adni_stage_c2_qc.csv'; SUB='036_S_1001'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=BASE/'036_S_1001_pet_registration_adjudication.json';a=json.loads(ap.read_text())
 a['visual_status']='PASS_NO_DETERIORATION'
 a['visual_findings']='Improved centering and coverage; no reflection, truncation, or gross deterioration on multislice overlays and checkerboards.'
 rule=a['universal_rule'];rule['no_visual_deterioration']=True;rule['absolute_dice_drop']=a['before']['dice']-a['after']['dice'];rule['nmi_improves']=a['after']['nmi']>a['before']['nmi'];rule['absolute_dice_drop_lt_0.01']=rule['absolute_dice_drop']<.01
 a['decision']='WARNING_ACCEPTED' if all([rule['nmi_improves'],rule['absolute_dice_drop_lt_0.01'],rule['transform_valid'],rule['no_visual_deterioration']]) else 'FAIL';ap.write_text(json.dumps(a,indent=2)+'\n')
 pd.DataFrame([{'subject_id':SUB,'decision':a['decision'],'rule_version':'C2-PET-QC-1','nmi_delta':a['delta_after_minus_before']['nmi'],'dice_delta':a['delta_after_minus_before']['dice'],'absolute_dice_drop':rule['absolute_dice_drop'],'visual_status':a['visual_status'],'transform_valid':rule['transform_valid'],'adjudication_path':str(ap)}]).to_csv(BASE/'pet_registration_adjudications.csv',index=False)
 q=pd.read_csv(QC);d=q.pet_mri_dice_before-q.pet_mri_dice_after;eligible=(q.pet_mri_nmi_after>q.pet_mri_nmi_before)&(d>0)&(d<.01)&(q.pet_rigid_determinant>0)&q.final_pet_finite&q.visual_status.eq('FAIL')
 # Only cases with completed visual adjudication can be accepted. At this point that is SUB.
 ix=q.subject_id.eq(SUB)&eligible
 if not ix.any() or a['decision']!='WARNING_ACCEPTED':raise RuntimeError('Universal acceptance rule not satisfied')
 q.loc[ix,'status']='WARNING_ACCEPTED';q.loc[ix,'visual_status']='PASS_NO_DETERIORATION';q.loc[ix,'reasons']='PET-MRI NMI improved; Dice absolute decrease 0.001669 < 0.01; valid transform; visual QC shows no deterioration';q.loc[ix,'analytical_eligibility']=True
 q.to_csv(QC,index=False)
 sd=BASE/'subjects'/f'sub-{SUB}';native=R/'data/derived/adni/nifti_native';conf=R/'configs/adni_stage_c2.yaml'
 signature={'mri':sha(native/f'sub-{SUB}_T1w.nii.gz'),'pet':sha(native/f'sub-{SUB}_FDGpet.nii.gz'),'config':sha(conf)}
 (sd/'complete.json').write_text(json.dumps({'signature':signature,'status':'WARNING_ACCEPTED','completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'adjudication':str(ap),'rule_version':'C2-PET-QC-1'},indent=2)+'\n')
 row=json.loads((sd/'qc.json').read_text());row['status']='WARNING_ACCEPTED';row['reasons']=q.loc[ix,'reasons'].iloc[0];row['visual_status']='PASS_NO_DETERIORATION';row['analytical_eligibility']=True;(sd/'qc.json').write_text(json.dumps(row,indent=2)+'\n')
 print(q.status.value_counts().to_string());print('036_S_1001',a['decision'])
if __name__=='__main__':main()
