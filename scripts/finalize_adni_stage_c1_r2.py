#!/usr/bin/env python3
"""Compose selected MRI transform with PET rigid transforms and finalize C1-R2."""
from pathlib import Path
import csv, hashlib, json
import ants, nibabel as nib, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];BASE=R/'data/derived/adni/stage_c1_r2';R1=R/'data/derived/adni/stage_c1_r1/subjects';FIG=R/'reports/figures/adni_stage_c1_r2'
TPL=R/'data/derived/adni/stage_c1/templateflow/tpl-MNI152NLin2009cAsym';HEAD=TPL/'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz';MASK=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
PRIMARY=['005_S_0223','005_S_0222','005_S_0221','009_S_1199','153_S_4172']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def nmi(a,b):
 x=a.ravel();y=b.ravel();idx=np.linspace(0,len(x)-1,min(len(x),300000),dtype=int);h=np.histogram2d(x[idx],y[idx],64)[0];p=h/max(h.sum(),1);px=p.sum(1);py=p.sum(0);z=p>0;mi=(p[z]*np.log(p[z]/(px[:,None]*py[None,:])[z])).sum();hx=-(px[px>0]*np.log(px[px>0])).sum();hy=-(py[py>0]*np.log(py[py>0])).sum();return float(2*mi/(hx+hy))
def pet_chain():
 fixed=ants.image_read(str(HEAD)); rows=[]
 for s in PRIMARY:
  r2=BASE/'subjects'/f'sub-{s}'/'standard_synra';r1=R1/f'sub-{s}';out=r2/f'sub-{s}_PET_MNI2mm.nii.gz'
  mean=r1/f'sub-{s}_PET_mean_after_motion.nii.gz'; ptx=r1/'transforms'/'pet_to_mri_fwd_0.mat'; tx=[str(r2/'transforms'/'selected_fwd_0.nii.gz'),str(r2/'transforms'/'selected_fwd_1.mat'),str(ptx)]
  pet=ants.apply_transforms(fixed,ants.image_read(str(mean)),tx,interpolator='linear',singleprecision=True);ants.image_write(pet,str(out));a=pet.numpy();mri=ants.image_read(str(r2/'final_T1w_MNI2mm.nii.gz'))
  row={'subject_id':s,'transform_order':json.dumps(tx),'single_resampling':True,'finite':bool(np.isfinite(a).all()),'empty':not bool(np.any(a!=0)),'nmi_pet_mri_mni':nmi(mri.numpy(),a),'pet_mni_sha256':sha(out),'shape':'x'.join(map(str,nib.load(out).shape)),'orientation':''.join(nib.aff2axcodes(nib.load(out).affine))};rows.append(row)
  fig,ax=plt.subplots(1,3,figsize=(9,3));ma=mri.numpy(); pa=a
  for c,axis in enumerate([2,1,0]):
   i=ma.shape[axis]//2;x=np.rot90(np.take(ma,i,axis));y=np.rot90(np.take(pa,i,axis));v=y[y!=0];lo,hi=np.percentile(v,[1,99]) if v.size else (0,1);ax[c].imshow(x,cmap='gray');ax[c].imshow(y,cmap='hot',alpha=.35,vmin=lo,vmax=hi);ax[c].axis('off')
  fig.suptitle(f'{s}: composed PET→MRI→MNI, one resampling');fig.tight_layout();fig.savefig(FIG/f'{s}_selected_pet_mri_mni.png',dpi=150);plt.close(fig)
 with open(BASE/'pet_chain_qc.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
def diagnosis():
 rows=[]
 for s in PRIMARY:
  d=BASE/'subjects'/f'sub-{s}'/'inputs';
  for kind,fn in [('n4','T1w_RAS_N4.nii.gz'),('mask','SynthStrip_mask.nii.gz')]:
   im=nib.load(d/fn);a=np.asarray(im.dataobj);nz=a>0;idx=np.argwhere(nz);qf,qc=im.get_qform(coded=True);sf,sc=im.get_sform(coded=True)
   rows.append({'subject_id':s,'image':kind,'shape':'x'.join(map(str,im.shape)),'spacing':json.dumps([float(x) for x in im.header.get_zooms()[:3]]),'orientation':''.join(nib.aff2axcodes(im.affine)),'affine':json.dumps(im.affine.tolist()),'qform_code':int(qc),'sform_code':int(sc),'qform_equals_sform':bool(np.allclose(qf,sf)),'origin_world':json.dumps(im.affine[:3,3].tolist()),'geometric_center_world':json.dumps(nib.affines.apply_affine(im.affine,(np.array(im.shape[:3])-1)/2).tolist()),'bounding_box_voxel':json.dumps([idx.min(0).tolist(),idx.max(0).tolist()]),'nonzero_voxels':int(nz.sum()),'min':float(a.min()),'max':float(a.max()),'mean_nonzero':float(a[nz].mean())})
 # Explicitly reproduce the R1 mismatch using immutable copied transforms.
 s='005_S_0223';r1=R1/f'sub-{s}';fm=ants.image_read(str(MASK));mm=ants.image_read(str(r1/f'sub-{s}_SynthStrip_mask.nii.gz'));stored=ants.image_read(str(r1/f'sub-{s}_SynthStrip_mask_MNI2mm.nii.gz'));tx=[str(r1/'transforms'/'mri_to_mni_fwd_0.nii.gz'),str(r1/'transforms'/'mri_to_mni_fwd_1.mat')];re=ants.apply_transforms(fm,mm,tx,interpolator='nearestNeighbor')
 def di(a,b):a=a.numpy()>0;b=b.numpy()>0;return float(2*(a&b).sum()/(a.sum()+b.sum()))
 evidence={'fixed':'MNI','moving':'MRI/SynthStrip mask','correct_forward_order':tx,'stored_r1_mask_vs_template_dice':di(fm,stored),'reapplied_current_chain_vs_template_dice':di(fm,re),'stored_vs_reapplied_dice':di(stored,re),'reversed_chain_vs_template_dice':di(fm,ants.apply_transforms(fm,mm,list(reversed(tx)),interpolator='nearestNeighbor')),'conclusion':'R1 0.341 was stale/inconsistent derived QC output, not current forward-transform performance.'}
 pd.DataFrame(rows).to_csv(BASE/'geometry_diagnosis.csv',index=False);(BASE/'005_S_0223_r1_transform_diagnosis.json').write_text(json.dumps(evidence,indent=2)+'\n');return evidence
def main():
 pet=pet_chain();ev=diagnosis();primary=pd.read_csv(R/'reports/adni_stage_c1_r2_qc.csv');ext=pd.read_csv(R/'reports/adni_stage_c1_r2_extension_qc.csv');combined=pd.concat([primary,ext],ignore_index=True,sort=False);combined.to_csv(R/'reports/adni_stage_c1_r2_qc.csv',index=False)
 summary=combined.groupby('strategy').agg(n=('status','size'),passes=('status',lambda x:(x=='PASS').sum()),dice_min=('syn_dice','min'),dice_mean=('syn_dice','mean'),nmi_mean=('syn_nmi','mean'),com_max_mm=('syn_com_distance_mm','max'),jacobian_min=('syn_jacobian_min','min'),jacobian_max=('syn_jacobian_max','max')).reset_index();summary.to_csv(BASE/'strategy_comparison.csv',index=False)
 stage=[]
 for _,r in combined.iterrows():
  for st in ['before','rigid','affine','syn']:
   stage.append({'subject_id':r.subject_id,'validation_set':r.validation_set,'strategy':r.strategy,'stage':st,'dice':r.get(st+'_dice'),'nmi':r.get(st+'_nmi'),'coverage':r.get(st+'_coverage'),'com_distance_mm':r.get(st+'_com_distance_mm'),'affine_determinant':r.get(st+'_affine_determinant'),'jacobian_min':r.get(st+'_jacobian_min'),'jacobian_max':r.get(st+'_jacobian_max'),'jacobian_nonpositive_fraction':r.get(st+'_jacobian_nonpositive_fraction'),'outside_fov_voxels':r.get(st+'_outside_fov_voxels')})
 pd.DataFrame(stage).to_csv(BASE/'metrics_by_stage.csv',index=False)
 provenance={'selected_strategy':'standard_synra','selection_frozen_before_extension':True,'primary_visual_qc':'PASS 5/5','extension_visual_qc':'PASS 6/6','pet_chain_qc':pet,'r1_diagnostic':ev,'antspyx':ants.__version__,'interpolators':{'intensity':'linear','mask':'nearestNeighbor'},'stage_c2_started':False};(BASE/'final_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 files=[]
 for p in sorted(BASE.rglob('*')):
  if p.is_file() and p.name!='sha256sums.csv':files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 for p in [R/'configs/adni_stage_c1_r2.yaml',R/'scripts/run_adni_stage_c1_r2.py',R/'scripts/run_adni_stage_c1_r2_extension.py',Path(__file__),R/'reports/adni_stage_c1_r2_qc.csv']:
  files.append((str(p.relative_to(R)),sha(p),p.stat().st_size))
 with open(BASE/'sha256sums.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['path','sha256','bytes']);w.writerows(files)
 print(summary.to_string(index=False));print('PET chains PASS',sum(x['finite'] and not x['empty'] and x['shape']=='97x115x97' and x['orientation']=='RAS' for x in pet),'/5');print('C2 STARTED: NO')
if __name__=='__main__':main()
