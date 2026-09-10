#!/usr/bin/env python3
"""Create Stage D splits, normalization variants, input indexes and QC; never train."""
from pathlib import Path
import csv,hashlib,json,math,stat,time
from collections import Counter
import numpy as np,nibabel as nib
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];D=R/'data/derived/adni/stage_d/v1';C2=R/'data/derived/adni/stage_c2';MAN=R/'data/derived/adni/adni_c2_manifest.csv';CONF=R/'configs/adni_stage_d.yaml';REPORTS=R/'reports'
SEEDS=[20260910,20261007,20261103];SHAPE=(97,115,97)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_csv(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a,mask):
 x=a[mask];return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)),p01=float(np.percentile(x,1)),p99=float(np.percentile(x,99)),nonzero_fraction=float(np.count_nonzero(a)/a.size),nan_count=int(np.isnan(a).sum()),inf_count=int(np.isinf(a).sum()))
def save_like(src,a,p):
 im=nib.load(src);h=im.header.copy();h.set_data_dtype(np.float32);out=nib.Nifti1Image(a.astype(np.float32),im.affine,h);out.set_qform(im.get_qform(),int(im.header['qform_code']));out.set_sform(im.get_sform(),int(im.header['sform_code']));p.parent.mkdir(parents=True,exist_ok=True);nib.save(out,p)
def main():
 frozen=json.loads((C2/'FROZEN.json').read_text())
 if frozen['status']!='FROZEN_QC_ACCEPTED' or frozen['subjects']!=54:raise RuntimeError('C2 is not frozen/complete')
 ledger=list(csv.DictReader((C2/'frozen_pairs_sha256.csv').open()))
 bad=[r['path'] for r in ledger if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']]
 if bad:raise RuntimeError(f'C2 hash failure: {bad[:3]}')
 subjects=sorted(list(csv.DictReader(MAN.open())),key=lambda r:r['subject_id'])
 if len(subjects)!=54 or Counter(r['group'] for r in subjects)!=Counter({'CN':18,'MCI':18,'AD':18}):raise RuntimeError('Cohort mismatch')
 D.mkdir(parents=True,exist_ok=True)
 # Seeds are published in config before this deterministic split creation.
 ids=np.array([r['subject_id'] for r in subjects]);y=np.array([r['group'] for r in subjects]);split_rows=[];audit=[]
 for rep,seed in enumerate(SEEDS,1):
  outer=StratifiedKFold(3,shuffle=True,random_state=seed)
  for of,(dev,test) in enumerate(outer.split(ids,y),1):
   inner_seed=seed+1000*of;inner=StratifiedKFold(3,shuffle=True,random_state=inner_seed)
   for inf,(tr0,val0) in enumerate(inner.split(ids[dev],y[dev]),1):
    tr=dev[tr0];val=dev[val0]
    roles={int(i):'outer_test' for i in test};roles.update({int(i):'inner_train' for i in tr});roles.update({int(i):'inner_validation' for i in val})
    if len(roles)!=54 or set(tr)&set(val) or set(tr)&set(test) or set(val)&set(test):raise RuntimeError('Split leakage')
    counts=Counter((roles[i],y[i]) for i in range(54))
    expected={('outer_test',g):6 for g in ('CN','MCI','AD')}|{('inner_train',g):8 for g in ('CN','MCI','AD')}|{('inner_validation',g):4 for g in ('CN','MCI','AD')}
    if any(counts[k]!=v for k,v in expected.items()):raise RuntimeError(f'Imbalance {rep}/{of}/{inf}: {counts}')
    for i in range(54):split_rows.append({'repeat':rep,'repeat_seed':seed,'outer_fold':of,'inner_fold':inf,'inner_seed':inner_seed,'subject_id':ids[i],'safe_subject_id':subjects[i]['safe_subject_id'],'group':y[i],'role':roles[i]})
    audit.append({'repeat':rep,'outer_fold':of,'inner_fold':inf,'seed':seed,'inner_seed':inner_seed,'inner_train_n':len(tr),'inner_validation_n':len(val),'outer_test_n':len(test),'train_per_class':8,'validation_per_class':4,'test_per_class':6,'subject_intersections':0,'status':'PASS'})
 write_csv(D/'splits/adni_nested_cv_3x3_r3.csv',split_rows);write_csv(D/'qc/split_audit.csv',audit)
 # Normalize with only subject-local mask statistics. No variant selection.
 qc=[];prov=[];index=[]
 for n,r in enumerate(subjects,1):
  sid=r['subject_id'];safe=r['safe_subject_id'];mri=Path(r['mri_mni_path']);pet=Path(r['pet_mni_path']);maskp=mri.parent/f'sub-{sid}_SynthStrip_mask_MNI2mm.nii.gz'
  if not maskp.is_file():raise FileNotFoundError(maskp)
  mi=nib.load(mri);pi=nib.load(pet);ki=nib.load(maskp);ma=np.asarray(ki.dataobj)>0.5;mx=np.asarray(mi.dataobj,dtype=np.float32);px=np.asarray(pi.dataobj,dtype=np.float32)
  for im,name in [(mi,'MRI'),(pi,'PET'),(ki,'mask')]:
   if im.shape!=SHAPE or ''.join(nib.aff2axcodes(im.affine))!='RAS' or not np.allclose(im.affine,mi.affine,atol=1e-4):raise RuntimeError(f'{sid} invalid {name} geometry')
  if not np.isfinite(mx).all() or not np.isfinite(px).all() or not ma.any():raise RuntimeError(f'{sid} invalid values/mask')
  mx=np.where(ma,mx,0);px=np.where(ma,px,0);mvals=mx[ma];pvals=px[ma];positive=pvals[pvals>0]
  med=float(np.median(mvals));iqr=float(np.percentile(mvals,75)-np.percentile(mvals,25));mlo,mhi=np.percentile(mvals,[.5,99.5]);plo,phi=np.percentile(pvals,[1,99]);pmean=float(positive.mean()) if positive.size else 0
  if min(iqr,mhi-mlo,phi-plo,pmean)<=0:raise RuntimeError(f'{sid} invalid normalization denominator')
  variants={
   'mri_robust_z':np.where(ma,np.clip((mx-med)/iqr,-8,8),0),
   'mri_percentile01':np.where(ma,np.clip((mx-mlo)/(mhi-mlo),0,1),0),
   'pet_relative_robust':np.where(ma,np.clip((px-plo)/(phi-plo),0,1),0),
   'pet_whole_brain_mean':np.where(ma,px/pmean,0)}
  paths={}
  for variant,a in variants.items():
   modality='MRI' if variant.startswith('mri_') else 'PET';p=D/'normalized'/variant/f'{safe}_{modality}.nii.gz'
   if p.exists():
    old=nib.load(p)
    if old.shape!=SHAPE or not np.allclose(old.affine,mi.affine):raise RuntimeError(f'Existing invalid output {p}')
   else:save_like(mri if modality=='MRI' else pet,a,p)
   if not np.isfinite(a).all():raise RuntimeError(f'Nonfinite {sid}/{variant}')
   paths[variant]=str(p.resolve());s=stats(a,ma);q={'subject_id':sid,'group':r['group'],'variant':variant,'modality':modality,'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size,'shape':'97x115x97','spacing':'2x2x2','orientation':'RAS','affine_match':True,'mask_voxels':int(ma.sum()),'status':'PASS',**s};qc.append(q)
   prov.append({'subject_id':sid,'variant':variant,'source_path':str((mri if modality=='MRI' else pet).resolve()),'source_sha256':sha(mri if modality=='MRI' else pet),'mask_path':str(maskp.resolve()),'mask_sha256':sha(maskp),'output_path':str(p.resolve()),'output_sha256':sha(p),'parameters':json.dumps({'median':med,'iqr':iqr} if variant=='mri_robust_z' else {'p0.5':float(mlo),'p99.5':float(mhi)} if variant=='mri_percentile01' else {'p1':float(plo),'p99':float(phi)} if variant=='pet_relative_robust' else {'positive_brain_mean':pmean},sort_keys=True)})
  for mv in ('mri_robust_z','mri_percentile01'):index.append({'subject_id':sid,'safe_subject_id':safe,'group':r['group'],'input_mode':'MRI-only','variant':mv,'channel_order':'MRI','channel_1_path':paths[mv],'channel_2_path':''})
  for pv in ('pet_relative_robust','pet_whole_brain_mean'):index.append({'subject_id':sid,'safe_subject_id':safe,'group':r['group'],'input_mode':'PET-only','variant':pv,'channel_order':'PET','channel_1_path':paths[pv],'channel_2_path':''})
  for mv in ('mri_robust_z','mri_percentile01'):
   for pv in ('pet_relative_robust','pet_whole_brain_mean'):index.append({'subject_id':sid,'safe_subject_id':safe,'group':r['group'],'input_mode':'multimodal','variant':mv+'__'+pv,'channel_order':'MRI,PET','channel_1_path':paths[mv],'channel_2_path':paths[pv]})
  print(f'STAGE_D_INPUTS {n}/54 {sid}',flush=True)
 write_csv(D/'qc/normalization_qc.csv',qc);write_csv(D/'manifests/normalization_provenance.csv',prov);write_csv(D/'manifests/model_input_index.csv',index)
 # Explicitly unselected variant registry; SUVR remains a gated configuration.
 variants=[{'modality':'MRI','variant':'mri_robust_z','materialized':True,'selection_status':'UNSELECTED_INNER_CV_ONLY'},{'modality':'MRI','variant':'mri_percentile01','materialized':True,'selection_status':'UNSELECTED_INNER_CV_ONLY'},{'modality':'PET','variant':'pet_relative_robust','materialized':True,'selection_status':'UNSELECTED_INNER_CV_ONLY'},{'modality':'PET','variant':'pet_whole_brain_mean','materialized':True,'selection_status':'UNSELECTED_INNER_CV_ONLY'},{'modality':'PET','variant':'pet_reference_suvr','materialized':False,'selection_status':'GATED_PENDING_ROI_COVERAGE_QC'}];write_csv(D/'manifests/normalization_variants.csv',variants)
 files=[]
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name not in ('provenance_sha256.csv','COMPLETE.json'):files.append({'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size})
 for p in (CONF,MAN,C2/'FROZEN.json',C2/'frozen_pairs_sha256.csv',Path(__file__)):
  files.append({'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size})
 write_csv(D/'provenance_sha256.csv',files)
 state={'stage':'D','status':'COMPLETE_NO_TRAINING','subjects':54,'split_design':'nested 3 outer x 3 inner x 3 repeats','seeds':SEEDS,'split_rows':len(split_rows),'normalization_qc_rows':len(qc),'input_index_rows':len(index),'materialized_variants':4,'suvr_status':'GATED_PENDING_ROI_COVERAGE_QC','variant_selection':'NOT_PERFORMED; inner-training-only when training is authorized','external_test_consulted':False,'training_started':False,'c2_hashes_verified':True,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(D/'COMPLETE.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
if __name__=='__main__':main()
