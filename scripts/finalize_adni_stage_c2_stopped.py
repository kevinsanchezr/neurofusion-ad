#!/usr/bin/env python3
"""Finalize metadata after the mandatory C2 stop gate; never resumes processing."""
from pathlib import Path
import csv,hashlib,json
import pandas as pd
R=Path(__file__).resolve().parents[1];BASE=R/'data/derived/adni/stage_c2';QC=R/'reports/adni_stage_c2_qc.csv';MAN=R/'data/derived/adni/adni_native_manifest.csv'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 q=pd.read_csv(QC);q['visual_status']='PASS';q.loc[q.subject_id.eq('036_S_1001'),'visual_status']='FAIL';q['analytical_eligibility']=q.status.eq('PASS');q.loc[q.subject_id.eq('009_S_1199'),'analytical_eligibility']=True;q.to_csv(QC,index=False)
 n=pd.read_csv(MAN);base=n.drop_duplicates('subject_id')[['safe_subject_id','subject_id','group','mri_date','pet_date','delta_days','protocol_exception']].copy()
 for mod,pfx in [('MRI','mri'),('PET','pet')]:
  x=n[n.modality.eq(mod)][['subject_id','image_id','series_id','visit','acquisition_date','original_format','nifti_path']].rename(columns={c:f'{pfx}_{c}' for c in ['image_id','series_id','visit','acquisition_date','original_format','nifti_path']});base=base.merge(x,on='subject_id',how='left')
 cols=['subject_id','status','reasons','visual_status','analytical_eligibility','mri_mni_path','pet_mni_path','mask_path','pet_motion_corrected_path','pet_mean_path','mri_mni_dice','jacobian_min','jacobian_max','jacobian_nonpositive_fraction','pet_mri_nmi_before','pet_mri_nmi_after','pet_mri_dice_before','pet_mri_dice_after','pet_to_mni_single_resampling']
 out=base.merge(q[cols],on='subject_id',how='left');out['c2_status']=out.status.fillna('NOT_PROCESSED_STOP_GATE');out.loc[out.subject_id.eq('036_S_1001'),'c2_status']='FAIL_EXCLUDED';out.loc[out.subject_id.eq('009_S_1199'),'c2_status']='WARNING_EXCEPTION_APPROVED_BEFORE_STOP';out['analytical_eligibility']=out.analytical_eligibility.fillna(False);out['notes']='';out.loc[out.c2_status.eq('NOT_PROCESSED_STOP_GATE'),'notes']='Not processed because C2 stopped at 036_S_1001';out.loc[out.subject_id.eq('036_S_1001'),'notes']='PET-to-MRI Dice decreased 0.663983 to 0.662315 despite NMI improvement; mandatory stop gate';out.loc[out.subject_id.eq('009_S_1199'),'notes']='47-slice PET native coverage exception retained';out.loc[out.subject_id.eq('153_S_4172'),'notes']='ADNI2 PET-DICOM/LAS exception; not reached before stop';out.to_csv(R/'data/derived/adni/adni_c2_manifest.csv',index=False)
 files=[]
 for p in sorted(BASE.rglob('*')):
  if p.is_file() and p.name!='sha256sums.csv':files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 for p in [R/'configs/adni_stage_c2.yaml',R/'scripts/run_adni_stage_c2.py',Path(__file__),QC,R/'data/derived/adni/adni_c2_manifest.csv']:
  files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 with open(BASE/'sha256sums.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['path','sha256','bytes']);w.writerows(files)
 (BASE/'STOPPED.json').write_text(json.dumps({'stage':'C2','status':'STOPPED_ON_FAIL','failed_subject':'036_S_1001','completed_subjects':35,'failed_outputs_excluded':True,'not_processed':18,'next_stage_started':False},indent=2)+'\n')
 print(out.c2_status.value_counts().to_string());print('C2 STOPPED; NEXT STAGE NOT STARTED')
if __name__=='__main__':main()
