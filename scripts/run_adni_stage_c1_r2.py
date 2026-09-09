#!/usr/bin/env python3
"""C1-R2 MRI-to-MNI strategy comparison. Never processes the full cohort."""
from pathlib import Path
import csv, hashlib, json, os, shutil, subprocess, sys, time
import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
R=Path(__file__).resolve().parents[1]; NATIVE=R/'data/derived/adni/nifti_native'
R1=R/'data/derived/adni/stage_c1_r1/subjects'; BASE=R/'data/derived/adni/stage_c1_r2'
OUT=BASE/'subjects'; FIG=R/'reports/figures/adni_stage_c1_r2'; QC=R/'reports/adni_stage_c1_r2_qc.csv'
TPL=R/'data/derived/adni/stage_c1/templateflow/tpl-MNI152NLin2009cAsym'
HEAD=TPL/'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'; BRAIN=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'; MASK=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
PRIMARY=[('005_S_0223','CN'),('005_S_0222','MCI'),('005_S_0221','AD'),('009_S_1199','MCI'),('153_S_4172','AD')]
EXT=[('011_S_0005','CN'),('024_S_0985','CN'),('011_S_0326','MCI'),('024_S_1400','MCI'),('011_S_0003','AD'),('094_S_1164','AD')]
STRATS=['baseline_syn','staged_com_rigid_affine_syn','standard_synra']
IMAGE='freesurfer/synthstrip:1.8'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dice(a,b):
 a=np.asarray(a)>0;b=np.asarray(b)>0;d=a.sum()+b.sum();return float(2*(a&b).sum()/d) if d else 0
def nmi(a,b,bins=64):
 x=np.asarray(a).ravel();y=np.asarray(b).ravel();ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)>300000: idx=np.linspace(0,len(x)-1,300000,dtype=int);x=x[idx];y=y[idx]
 h=np.histogram2d(x,y,bins=bins)[0];p=h/max(h.sum(),1);px=p.sum(1);py=p.sum(0);z=p>0
 mi=(p[z]*np.log(p[z]/(px[:,None]*py[None,:])[z])).sum(); hx=-(px[px>0]*np.log(px[px>0])).sum();hy=-(py[py>0]*np.log(py[py>0])).sum();return float(2*mi/(hx+hy)) if hx+hy else 0
def com_world(img, binary=False):
 a=img.numpy(); w=(a>0) if binary else np.maximum(a,0); idx=np.array(center_of_mass(w)); return np.array(img.origin)+np.array(img.direction).reshape(3,3)@(idx*np.array(img.spacing))
def affine_info(path):
 t=ants.read_transform(str(path));p=np.asarray(t.parameters,float);m=p[:9].reshape(3,3)
 return float(np.linalg.det(m)),p[9:12].tolist(),Rotation.from_matrix(m).as_euler('xyz',degrees=True).tolist()
def copytx(paths,d,prefix):
 out=[]
 for i,x in enumerate(paths):
  ext='nii.gz' if str(x).endswith('.nii.gz') else 'mat';q=d/f'{prefix}_{i}.{ext}';shutil.copy2(x,q);out.append(str(q))
 return out
def stage_metrics(fixed,fmask,moving,mmask,warped,warpedmask,affine_path=None,warp_path=None):
 fm=fmask.numpy()>0;wm=warpedmask.numpy()>0; nz=warped.numpy()!=0
 row={'dice':dice(fm,wm),'nmi':nmi(fixed.numpy(),warped.numpy()),'coverage':float((fm&wm).sum()/max(fm.sum(),1)),
      'com_distance_mm':float(np.linalg.norm(com_world(fmask,True)-com_world(warpedmask,True))),
      'outside_fov_voxels':int(np.logical_and(fm,~nz).sum()),'finite':bool(np.isfinite(warped.numpy()).all()),'empty':not bool(np.any(nz))}
 row['affine_determinant']=np.nan
 if affine_path: row['affine_determinant']=affine_info(affine_path)[0]
 row.update(jacobian_min=np.nan,jacobian_max=np.nan,jacobian_nonpositive_fraction=np.nan)
 if warp_path:
  jac=ants.create_jacobian_determinant_image(fixed,str(warp_path),do_log=False,geom=False);a=jac.numpy();row.update(jacobian_min=float(a.min()),jacobian_max=float(a.max()),jacobian_nonpositive_fraction=float(np.mean(a<=0)))
 return row
def apply_stage(fixed,fmask,moving,mmask,tx,sd,name,aff=None,warp=None):
 w=ants.apply_transforms(fixed,moving,tx,interpolator='linear');wm=ants.apply_transforms(fmask,mmask,tx,interpolator='nearestNeighbor')
 ants.image_write(w,str(sd/f'{name}_T1w.nii.gz'));ants.image_write(wm,str(sd/f'{name}_mask.nii.gz'))
 return w,wm,stage_metrics(fixed,fmask,moving,mmask,w,wm,aff,warp)
def reg(fixed,moving,fmask,mmask,typ,prefix,initial=None):
 return ants.registration(fixed=fixed,moving=moving,type_of_transform=typ,initial_transform=initial,outprefix=str(prefix),mask=fmask,moving_mask=mmask,mask_all_stages=True,aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=.2,syn_metric='CC',syn_sampling=4,reg_iterations=(40,20,0),random_seed=20260909,singleprecision=True,verbose=True)
def prepare(sub,extension=False):
 sd=OUT/f'sub-{sub}'/'inputs';sd.mkdir(parents=True,exist_ok=True)
 if not extension:
  src=R1/f'sub-{sub}'; pairs=[(src/f'sub-{sub}_T1w_RAS_N4.nii.gz',sd/'T1w_RAS_N4.nii.gz'),(src/f'sub-{sub}_SynthStrip_mask.nii.gz',sd/'SynthStrip_mask.nii.gz')]
  for a,b in pairs:
   if not b.exists(): shutil.copy2(a,b)
 else:
  ras=sd/'T1w_RAS.nii.gz';n4p=sd/'T1w_RAS_N4.nii.gz';maskp=sd/'SynthStrip_mask.nii.gz';strip=sd/'T1w_RAS_SynthStrip.nii.gz'
  if not ras.exists():nib.save(nib.as_closest_canonical(nib.load(NATIVE/f'sub-{sub}_T1w.nii.gz')),ras)
  if not n4p.exists():ants.image_write(ants.n4_bias_field_correction(ants.image_read(str(ras)),shrink_factor=4,convergence={'iters':[50,50,30,20],'tol':1e-7},rescale_intensities=False),str(n4p))
  if not maskp.exists():
   cmd=['docker','run','--rm','--network=none','-v',f'{sd.resolve()}:/data',IMAGE,'-i','/data/T1w_RAS.nii.gz','-o','/data/T1w_RAS_SynthStrip.nii.gz','-m','/data/SynthStrip_mask.nii.gz','-b','1'];(sd/'synthstrip_command.txt').write_text(' '.join(cmd)+'\n');subprocess.run(cmd,check=True)
 n4=ants.image_read(str(sd/'T1w_RAS_N4.nii.gz'));mask=ants.image_read(str(sd/'SynthStrip_mask.nii.gz'));brain=n4*mask;ants.image_write(brain,str(sd/'T1w_RAS_N4_brain.nii.gz'));return brain,mask
def qcfig(sub,strat,sd,fixed,fmask,stages):
 fig,axs=plt.subplots(len(stages),3,figsize=(9,3*len(stages)))
 for r,(name,img,mask) in enumerate(stages):
  a=img.numpy();m=mask.numpy();f=fixed.numpy();v=a[a!=0];lo,hi=np.percentile(v,[1,99]) if v.size else (0,1)
  for c,axis in enumerate([2,1,0]):
   idx=a.shape[axis]//2;x=np.rot90(np.take(a,idx,axis));mm=np.rot90(np.take(m,idx,axis));ff=np.rot90(np.take(f,idx,axis));axs[r,c].imshow(ff,cmap='gray');axs[r,c].imshow(x,cmap='magma',alpha=.3,vmin=lo,vmax=hi);axs[r,c].contour(mm,levels=[.5],colors='lime',linewidths=.5);axs[r,c].set_title(name,fontsize=7);axs[r,c].axis('off')
 fig.suptitle(f'{sub} — {strat}: template overlay + transformed mask');fig.tight_layout();fig.savefig(FIG/f'{sub}_{strat}_stages.png',dpi=130);plt.close(fig)
def run_strategy(sub,group,strat,fixed,fmask,moving,mmask,extension):
 sd=OUT/f'sub-{sub}'/strat;td=sd/'transforms';sd.mkdir(parents=True,exist_ok=True);td.mkdir(exist_ok=True)
 before=ants.resample_image_to_target(moving,fixed,interp_type='linear');bm=ants.resample_image_to_target(mmask,fmask,interp_type='nearestNeighbor');metrics={'before':stage_metrics(fixed,fmask,moving,mmask,before,bm)};stages=[('before',before,bm)]
 if strat=='baseline_syn':
  # Affine diagnostic stage, followed by the baseline SyN configuration.
  ar=reg(fixed,moving,fmask,mmask,'Affine',td/'diag_affine_'); aw,am,metrics['affine']=apply_stage(fixed,fmask,moving,mmask,ar['fwdtransforms'],sd,'affine',ar['fwdtransforms'][-1]);stages.append(('affine',aw,am))
  rr=reg(fixed,moving,fmask,mmask,'SyN',td/'final_'); tx=rr['fwdtransforms']; metrics['rigid']={'not_applicable':True}; final=rr['warpedmovout']; finalm=ants.apply_transforms(fmask,mmask,tx,interpolator='nearestNeighbor')
 elif strat=='staged_com_rigid_affine_syn':
  rigid=reg(fixed,moving,fmask,mmask,'Rigid',td/'rigid_');rw,rm,metrics['rigid']=apply_stage(fixed,fmask,moving,mmask,rigid['fwdtransforms'],sd,'rigid',rigid['fwdtransforms'][-1]);stages.append(('rigid',rw,rm))
  aff=reg(fixed,moving,fmask,mmask,'Affine',td/'affine_',rigid['fwdtransforms']);aw,am,metrics['affine']=apply_stage(fixed,fmask,moving,mmask,aff['fwdtransforms'],sd,'affine',aff['fwdtransforms'][-1]);stages.append(('affine',aw,am))
  syn=reg(fixed,moving,fmask,mmask,'SyNOnly',td/'syn_',aff['fwdtransforms']);tx=syn['fwdtransforms'];final=syn['warpedmovout'];finalm=ants.apply_transforms(fmask,mmask,tx,interpolator='nearestNeighbor')
 else:
  rigid=reg(fixed,moving,fmask,mmask,'Rigid',td/'diag_rigid_');rw,rm,metrics['rigid']=apply_stage(fixed,fmask,moving,mmask,rigid['fwdtransforms'],sd,'rigid',rigid['fwdtransforms'][-1]);stages.append(('rigid',rw,rm))
  aff=reg(fixed,moving,fmask,mmask,'Affine',td/'diag_affine_',rigid['fwdtransforms']);aw,am,metrics['affine']=apply_stage(fixed,fmask,moving,mmask,aff['fwdtransforms'],sd,'affine',aff['fwdtransforms'][-1]);stages.append(('affine',aw,am))
  rr=reg(fixed,moving,fmask,mmask,'SyNRA',td/'final_');tx=rr['fwdtransforms'];final=rr['warpedmovout'];finalm=ants.apply_transforms(fmask,mmask,tx,interpolator='nearestNeighbor')
 warp=next((x for x in tx if str(x).endswith('.nii.gz')),None);affine=next((x for x in reversed(tx) if str(x).endswith('.mat')),None);metrics['syn']=stage_metrics(fixed,fmask,moving,mmask,final,finalm,affine,warp);stages.append(('SyN final',final,finalm))
 ants.image_write(final,str(sd/'final_T1w_MNI2mm.nii.gz'));ants.image_write(finalm,str(sd/'final_mask_MNI2mm.nii.gz'));copied=copytx(tx,td,'selected_fwd')
 (sd/'metrics_by_stage.json').write_text(json.dumps(metrics,indent=2)+'\n');(sd/'transform_order.json').write_text(json.dumps({'fixed':'MNI','moving':sub,'ANTs_forward_order':copied},indent=2)+'\n');qcfig(sub,strat,sd,fixed,fmask,stages)
 row={'subject_id':sub,'group':group,'validation_set':'extension' if extension else 'primary','strategy':strat,'input_mri_sha256':sha(NATIVE/f'sub-{sub}_T1w.nii.gz'),'input_mask_sha256':sha(OUT/f'sub-{sub}'/'inputs'/'SynthStrip_mask.nii.gz')}
 for stage,val in metrics.items():
  for k,v in val.items():row[f'{stage}_{k}']=v
 row['transform_order']=json.dumps(copied);row['status']='PASS' if metrics['syn']['dice']>=.8 and metrics['syn']['finite'] and not metrics['syn']['empty'] and metrics['syn']['affine_determinant']>0 and metrics['syn']['jacobian_nonpositive_fraction']==0 else 'FAIL';return row
def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);fixed=ants.image_read(str(BRAIN));fmask=ants.image_read(str(MASK));rows=[]
 for sub,g in PRIMARY:
  moving,mmask=prepare(sub,False)
  for st in STRATS:
   print('START',sub,st,flush=True);rows.append(run_strategy(sub,g,st,fixed,fmask,moving,mmask,False));print('DONE',sub,st,rows[-1]['status'],rows[-1].get('syn_dice'),flush=True)
 with open(QC,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
 print(json.dumps({'primary_rows':len(rows),'status':{x:sum(r['status']==x for r in rows) for x in ['PASS','FAIL']},'extension_run':False},indent=2))
if __name__=='__main__':main()
