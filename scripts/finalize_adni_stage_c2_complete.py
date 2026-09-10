#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,time
import pandas as pd
R=Path(__file__).resolve().parents[1];BASE=R/'data/derived/adni/stage_c2';QC=R/'reports/adni_stage_c2_qc.csv';MAN=R/'data/derived/adni/adni_native_manifest.csv'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 q=pd.read_csv(QC);q['visual_status']='PASS';q.loc[q.subject_id.eq('036_S_1001'),'visual_status']='PASS_NO_DETERIORATION';q.loc[q.subject_id.eq('153_S_4172'),'visual_status']='PASS_NO_LR_INVERSION';q['analytical_eligibility']=~q.status.eq('FAIL');q.to_csv(QC,index=False)
 n=pd.read_csv(MAN);base=n.drop_duplicates('subject_id')[['safe_subject_id','subject_id','group','mri_date','pet_date','delta_days','protocol_exception']].copy()
 for mod,pfx in [('MRI','mri'),('PET','pet')]:
  x=n[n.modality.eq(mod)][['subject_id','image_id','series_id','visit','acquisition_date','original_format','nifti_path']].rename(columns={c:f'{pfx}_{c}' for c in ['image_id','series_id','visit','acquisition_date','original_format','nifti_path']});base=base.merge(x,on='subject_id',how='left')
 cols=['subject_id','status','reasons','visual_status','analytical_eligibility','mri_mni_path','pet_mni_path','mask_path','pet_motion_corrected_path','pet_mean_path','mri_mni_dice','mri_mni_coverage','mask_volume_ml','mask_components','mri_affine_determinant','jacobian_min','jacobian_max','jacobian_nonpositive_fraction','pet_frames','pet_motion_fd_mm','pet_mri_nmi_before','pet_mri_nmi_after','pet_mri_dice_before','pet_mri_dice_after','pet_rigid_determinant','pet_to_mni_single_resampling','final_pet_shape','final_pet_spacing','final_pet_orientation','final_pet_finite','mri_forward_transforms','mri_inverse_transforms','pet_forward_transform','pet_inverse_transform']
 out=base.merge(q[cols],on='subject_id',how='left');out['c2_status']=out.status;out['notes']='';out.loc[out.subject_id.eq('009_S_1199'),'notes']='47-slice native PET coverage exception retained';out.loc[out.subject_id.eq('036_S_1001'),'notes']='WARNING_ACCEPTED by universal C2-PET-QC-1 rule';out.loc[out.subject_id.eq('153_S_4172'),'notes']='ADNI2 PET-DICOM/LAS; BRAIN 2 only; no LR inversion detected';out.to_csv(R/'data/derived/adni/adni_c2_manifest.csv',index=False)
 state={'stage':'C2','status':'COMPLETE_QC_ACCEPTED','subjects':54,'counts':q.status.value_counts().to_dict(),'groups':out.groupby('group').size().to_dict(),'adjudication_rule':'C2-PET-QC-1','next_stage_started':False,'pet_normalization':False,'splits':False,'training':False,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(BASE/'COMPLETE.json').write_text(json.dumps(state,indent=2)+'\n')
 files=[]
 for p in sorted(BASE.rglob('*')):
  if p.is_file() and p.name!='sha256sums.csv':files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 for p in [R/'configs/adni_stage_c2.yaml',R/'scripts/run_adni_stage_c2.py',R/'scripts/adjudicate_adni_c2_pet_registration.py',R/'scripts/apply_adni_c2_pet_adjudication_rule.py',Path(__file__),QC,R/'data/derived/adni/adni_c2_manifest.csv',R/'reports/adni_stage_c2_complete_report.md']:
  if p.exists():files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 with open(BASE/'sha256sums.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['path','sha256','bytes']);w.writerows(files)
 print(json.dumps(state,indent=2));print('hashes',len(files))
if __name__=='__main__':main()
