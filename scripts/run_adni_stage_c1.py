#!/usr/bin/env python3
"""Five-subject ADNI C1 preprocessing validation; never processes the full cohort."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, time
from collections import Counter
from pathlib import Path
import ants, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parents[1]; NATIVE=R/'data/derived/adni/nifti_native'; BASE=R/'data/derived/adni/stage_c1'
OUT=BASE/'subjects'; FIG=R/'reports/figures/adni_stage_c1'; QC=R/'reports/adni_stage_c1_qc.csv'
TPL=BASE/'templateflow/tpl-MNI152NLin2009cAsym'; TPL_HEAD=TPL/'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'; TPL_BRAIN=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz'; TPL_MASK=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
SUBJECTS=[('005_S_0223','CN','standard ECAT7'),('005_S_0222','MCI','standard ECAT7'),('005_S_0221','AD','standard ECAT7'),('009_S_1199','MCI','47-slice ECAT7 exception'),('153_S_4172','AD','ADNI2 PET-DICOM/LAS exception')]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save_canonical(src,dst):
 im=nib.load(src); can=nib.as_closest_canonical(im); nib.save(can,dst); return im,can
def vals(path):
 i=nib.load(path);a=np.asanyarray(i.dataobj);return dict(shape='x'.join(map(str,i.shape)),spacing='x'.join(f'{z:.6g}' for z in i.header.get_zooms()),orientation=''.join(nib.aff2axcodes(i.affine)),affine=json.dumps(i.affine.tolist(),separators=(',',':')),finite=bool(np.isfinite(a).all()),minimum=float(a.min()),maximum=float(a.max()),mean=float(a.mean()),std=float(a.std()),nonzero_fraction=float(np.count_nonzero(a)/a.size))
def nmi(a,b,bins=64):
 x=np.asarray(a).ravel();y=np.asarray(b).ravel();ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)>300000:
  ix=np.linspace(0,len(x)-1,300000,dtype=int);x=x[ix];y=y[ix]
 h=np.histogram2d(x,y,bins=bins)[0];p=h/h.sum();px=p.sum(1);py=p.sum(0);nz=p>0
 mi=float((p[nz]*np.log(p[nz]/(px[:,None]*py[None,:])[nz])).sum());hx=float(-(px[px>0]*np.log(px[px>0])).sum());hy=float(-(py[py>0]*np.log(py[py>0])).sum());return 2*mi/(hx+hy) if hx+hy else 0
def dice(a,b):
 a=np.asarray(a)>0;b=np.asarray(b)>0;return float(2*np.logical_and(a,b).sum()/(a.sum()+b.sum())) if a.sum()+b.sum() else 0
def mask_metrics(fixed_mask, moving):
 ma=ants.get_mask(moving);return dice(fixed_mask.numpy(),ma.numpy()),float(np.logical_and(fixed_mask.numpy()>0,ma.numpy()>0).sum()/max(1,(fixed_mask.numpy()>0).sum()))
def affine_stats(path):
 t=ants.read_transform(path);p=np.asarray(t.parameters,float);m=p[:9].reshape(3,3);det=float(np.linalg.det(m));rot=Rotation.from_matrix(m).as_euler('xyz',degrees=True);trans=p[9:12];return det,trans.tolist(),rot.tolist()
def copy_transforms(paths,dest,prefix):
 out=[]
 for j,p in enumerate(paths):
  ext='nii.gz' if str(p).endswith('.nii.gz') else Path(p).suffix.lstrip('.') or 'mat';q=dest/f'{prefix}_{j}.{ext}';shutil.copy2(p,q);out.append(str(q))
 return out
def ortho(axs,a,title,mask=None):
 pos=a[a>0];lo,hi=np.percentile(pos,[1,99]) if pos.size else (0,1);x,y,z=[s//2 for s in a.shape]
 for ax,v,mv,n in zip(axs,[a[:,:,z],a[:,y,:],a[x,:,:]],[None if mask is None else mask[:,:,z],None if mask is None else mask[:,y,:],None if mask is None else mask[x,:,:]],['axial','coronal','sagittal']):
  ax.imshow(np.rot90(v),cmap='gray',vmin=lo,vmax=hi,origin='lower');
  if mv is not None:ax.contour(np.rot90(mv),levels=[.5],colors='lime',linewidths=.5)
  ax.set_title(title+' '+n,fontsize=7);ax.axis('off')
def qc_figure(sub,paths):
 fig,ax=plt.subplots(4,3,figsize=(10,11));ortho(ax[0],nib.load(paths['native']).get_fdata(),'MRI native');ortho(ax[1],nib.load(paths['n4']).get_fdata(),'MRI N4 + mask',nib.load(paths['mask']).get_fdata());ortho(ax[2],nib.load(paths['pet_before']).get_fdata(),'PET before');ortho(ax[3],nib.load(paths['pet_after']).get_fdata(),'PET motion-corrected');fig.suptitle(sub);fig.tight_layout();fig.savefig(FIG/f'{sub}_n4_mask_motion.png',dpi=130);plt.close(fig)
 fig,ax=plt.subplots(2,3,figsize=(10,6));m=nib.load(paths['mri_mni']).get_fdata();p=nib.load(paths['pet_mni']).get_fdata();ortho(ax[0],m,'MRI MNI');ortho(ax[1],p,'PET MNI');fig.suptitle(sub+' MNI aligned');fig.tight_layout();fig.savefig(FIG/f'{sub}_mni.png',dpi=130);plt.close(fig)
 # PET-MRI overlay and checkerboard in native MRI grid
 m=nib.load(paths['brain']).get_fdata();p=nib.load(paths['pet_mri']).get_fdata();pm=(p-p.min())/(p.max()-p.min()+1e-12);mm=(m-m.min())/(m.max()-m.min()+1e-12);x,y,z=[s//2 for s in m.shape];fig,ax=plt.subplots(2,3,figsize=(10,6))
 for col,(a,b) in enumerate([(mm[:,:,z],pm[:,:,z]),(mm[:,y,:],pm[:,y,:]),(mm[x,:,:],pm[x,:,:])]):
  a=np.rot90(a);b=np.rot90(b);ax[0,col].imshow(a,cmap='gray');ax[0,col].imshow(b,cmap='hot',alpha=.35);cb=np.where((np.indices(a.shape).sum(0)//12)%2,a,b);ax[1,col].imshow(cb,cmap='gray');ax[0,col].axis('off');ax[1,col].axis('off')
 fig.suptitle(sub+' PET→MRI overlay / checkerboard');fig.tight_layout();fig.savefig(FIG/f'{sub}_pet_to_mri.png',dpi=130);plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 th=ants.image_read(str(TPL_HEAD));tb=ants.image_read(str(TPL_BRAIN));tm=ants.image_read(str(TPL_MASK));rows=[];versions={'ants':ants.__version__,'nibabel':nib.__version__,'numpy':np.__version__}
 for sub,group,selection in SUBJECTS:
  print('START',sub,flush=True);sd=OUT/f'sub-{sub}';tr=sd/'transforms';sd.mkdir(parents=True,exist_ok=True);tr.mkdir(exist_ok=True)
  native_mri=NATIVE/f'sub-{sub}_T1w.nii.gz';native_pet=NATIVE/f'sub-{sub}_FDGpet.nii.gz';ras=sd/f'sub-{sub}_T1w_RAS.nii.gz';save_canonical(native_mri,ras)
  mri=ants.image_read(str(ras));n4=ants.n4_bias_field_correction(mri,shrink_factor=4,convergence={'iters':[50,50,30,20],'tol':1e-7},rescale_intensities=False);n4p=sd/f'sub-{sub}_T1w_RAS_N4.nii.gz';ants.image_write(n4,str(n4p))
  # Validated ANTs template-mask propagation: affine only for initial subject-space mask.
  init=ants.registration(th,n4,type_of_transform='AffineFast',outprefix=str(tr/'extract_init_'),random_seed=20260906,verbose=False);initf=copy_transforms(init['fwdtransforms'],tr,'extract_fwd');initi=copy_transforms(init['invtransforms'],tr,'extract_inv')
  mask=ants.apply_transforms(n4,tm,init['invtransforms'],interpolator='nearestNeighbor');mask=ants.threshold_image(mask,.5,1,1,0);maskp=sd/f'sub-{sub}_brain_mask.nii.gz';ants.image_write(mask,str(maskp));brain=n4*mask;brainp=sd/f'sub-{sub}_T1w_RAS_N4_brain.nii.gz';ants.image_write(brain,str(brainp))
  reg=ants.registration(tb,brain,type_of_transform='SyN',outprefix=str(tr/'mri_to_mni_'),reg_iterations=(40,20,0),random_seed=20260906,singleprecision=True,verbose=False);mf=copy_transforms(reg['fwdtransforms'],tr,'mri_to_mni_fwd');mi=copy_transforms(reg['invtransforms'],tr,'mri_to_mni_inv');mrip=sd/f'sub-{sub}_T1w_MNI2mm.nii.gz';ants.image_write(reg['warpedmovout'],str(mrip))
  # Warp subject mask for overlap QC and nonlinear Jacobian validity.
  wm=ants.apply_transforms(tm,mask,reg['fwdtransforms'],interpolator='nearestNeighbor');wmp=sd/f'sub-{sub}_brain_mask_MNI2mm.nii.gz';ants.image_write(wm,str(wmp));warp=next((p for p in reg['fwdtransforms'] if str(p).endswith('.nii.gz')),None);jacmin=jacmax=jacbad=''
  if warp:
   jac=ants.create_jacobian_determinant_image(tb,warp,do_log=False,geom=False);ja=jac.numpy();jacmin=float(ja.min());jacmax=float(ja.max());jacbad=float(np.mean(ja<=0));ants.image_write(jac,str(sd/f'sub-{sub}_mri_to_mni_jacobian.nii.gz'))
  # PET motion: keep 4D corrected, average only afterward. 3D exception bypasses motion.
  pet=ants.image_read(str(native_pet));motion_fd=[];motion_det=[];translations=[];rotations=[]
  if pet.dimension==4:
   before=ants.slice_image(pet,axis=3,idx=0);beforep=sd/f'sub-{sub}_PET_frame0_before_motion.nii.gz';ants.image_write(before,str(beforep));mc=ants.motion_correction(pet,type_of_transform='Rigid',fdOffset=50,outprefix=str(tr/'pet_motion_'),random_seed=20260906,verbose=False);pet4=mc['motion_corrected'];pet4p=sd/f'sub-{sub}_PET_motion_corrected_4D.nii.gz';ants.image_write(pet4,str(pet4p));motion_fd=np.asarray(mc['FD'],float).tolist()
   for pp in mc['motion_parameters']:
    if pp=='NA' or pp is None: motion_det.append(1.);translations.append([0,0,0]);rotations.append([0,0,0])
    else:
     f=pp[0] if isinstance(pp,list) else pp;det,t,r=affine_stats(f);motion_det.append(det);translations.append(t);rotations.append(r)
   petmean=ants.get_average_of_timeseries(pet4);pet_after=ants.slice_image(pet4,axis=3,idx=0)
  else:
   before=pet;pet_after=pet;beforep=sd/f'sub-{sub}_PET_3D_native.nii.gz';ants.image_write(before,str(beforep));pet4p='';petmean=pet
  meanp=sd/f'sub-{sub}_PET_mean_after_motion.nii.gz';ants.image_write(petmean,str(meanp));afterp=sd/f'sub-{sub}_PET_frame0_after_motion.nii.gz';ants.image_write(pet_after,str(afterp))
  before_grid=ants.resample_image_to_target(petmean,brain,interp_type='linear');mi_before=nmi(brain.numpy(),before_grid.numpy());dice_before,cov_before=mask_metrics(mask,before_grid)
  preg=ants.registration(brain,petmean,type_of_transform='Rigid',outprefix=str(tr/'pet_to_mri_'),random_seed=20260906,verbose=False);pf=copy_transforms(preg['fwdtransforms'],tr,'pet_to_mri_fwd');pi=copy_transforms(preg['invtransforms'],tr,'pet_to_mri_inv');petmrip=sd/f'sub-{sub}_PET_in_T1w.nii.gz';ants.image_write(preg['warpedmovout'],str(petmrip));petdet,pett,petrot=affine_stats(preg['fwdtransforms'][-1]);mi_after=nmi(brain.numpy(),preg['warpedmovout'].numpy());dice_after,cov_after=mask_metrics(mask,preg['warpedmovout'])
  chain=reg['fwdtransforms']+preg['fwdtransforms'];petmni=ants.apply_transforms(tb,petmean,chain,interpolator='linear',imagetype=0,singleprecision=True);petmnip=sd/f'sub-{sub}_PET_MNI2mm.nii.gz';ants.image_write(petmni,str(petmnip));(tr/'pet_to_mni_chain.json').write_text(json.dumps({'ordered_ANTs_transformlist':[str(x) for x in chain],'application':'single ants.apply_transforms interpolation'},indent=2)+'\n')
  mdice=dice(tm.numpy(),wm.numpy());mnmi=nmi(tb.numpy(),reg['warpedmovout'].numpy());mcoverage=float(np.logical_and(tm.numpy()>0,wm.numpy()>0).sum()/max(1,(tm.numpy()>0).sum()))
  v=vals(petmnip); reasons=[]
  if not v['finite'] or v['nonzero_fraction']==0:reasons.append('invalid/empty final PET')
  if jacbad!='' and jacbad>0:reasons.append('non-positive MRI warp Jacobian')
  if petdet<=0 or any(x<=0 for x in motion_det):reasons.append('invalid rigid determinant')
  if mi_after<=mi_before:reasons.append('PET→MRI NMI did not improve')
  if dice_after<dice_before:reasons.append('PET→MRI overlap did not improve')
  warnings=[]
  if sub=='009_S_1199' and cov_after<.75:warnings.append('limited PET coverage after registration')
  if sub=='153_S_4172':warnings.append('ADNI2 PET-DICOM/LAS exception; visually verify laterality')
  if mdice<.80 or mcoverage<.80:warnings.append('MRI-MNI brain overlap below 0.80')
  status='FAIL' if reasons else ('WARNING' if warnings else 'PASS')
  row={'subject_id':sub,'group':group,'selection_reason':selection,'status':status,'reasons':'; '.join(reasons+warnings),'mri_native_sha256':sha(native_mri),'pet_native_sha256':sha(native_pet),'motion_framewise_displacement_mm':json.dumps(motion_fd),'motion_translation_mm':json.dumps(translations),'motion_rotation_deg':json.dumps(rotations),'motion_determinants':json.dumps(motion_det),'pet_rigid_translation_mm':json.dumps(pett),'pet_rigid_rotation_deg':json.dumps(petrot),'pet_rigid_determinant':petdet,'pet_mri_nmi_before':mi_before,'pet_mri_nmi_after':mi_after,'pet_mri_dice_before':dice_before,'pet_mri_dice_after':dice_after,'pet_brain_coverage_before':cov_before,'pet_brain_coverage_after':cov_after,'mri_mni_nmi':mnmi,'mri_mni_dice':mdice,'mri_mni_coverage':mcoverage,'warp_jacobian_min':jacmin,'warp_jacobian_max':jacmax,'warp_nonpositive_fraction':jacbad,**v,'mri_fwd_transforms':json.dumps(mf),'mri_inverse_transforms':json.dumps(mi),'pet_fwd_transform':json.dumps(pf),'pet_inverse_transform':json.dumps(pi)};rows.append(row)
  qc_figure(sub,{'native':native_mri,'n4':n4p,'mask':maskp,'brain':brainp,'pet_before':beforep,'pet_after':afterp,'pet_mri':petmrip,'mri_mni':mrip,'pet_mni':petmnip});print('DONE',sub,status,flush=True)
 with open(QC,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (BASE/'tool_versions.json').write_text(json.dumps(versions,indent=2)+'\n');print(json.dumps({'counts':Counter(x['status'] for x in rows),'subjects':[x['subject_id'] for x in rows]},default=dict,indent=2))
if __name__=='__main__':main()
