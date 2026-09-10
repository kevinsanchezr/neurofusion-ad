#!/usr/bin/env python3
"""Universal PET-to-MRI adjudication metrics; does not resume C2."""
from pathlib import Path
import json, hashlib
import ants, nibabel as nib, numpy as np, pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, center_of_mass
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];BASE=R/'data/derived/adni/stage_c2';SUB='036_S_1001';SD=BASE/'subjects'/f'sub-{SUB}';FIG=R/'reports/figures/adni_stage_c2/adjudication_036_S_1001';FIG.mkdir(parents=True,exist_ok=True)
def dice(a,b):a=np.asarray(a)>0;b=np.asarray(b)>0;return float(2*(a&b).sum()/(a.sum()+b.sum()))
def nmi(a,b,bins=64):
 x=a.ravel();y=b.ravel();ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)>300000:i=np.linspace(0,len(x)-1,300000,dtype=int);x=x[i];y=y[i]
 h=np.histogram2d(x,y,bins)[0];p=h/max(h.sum(),1);px=p.sum(1);py=p.sum(0);z=p>0;mi=(p[z]*np.log(p[z]/(px[:,None]*py[None,:])[z])).sum();hx=-(px[px>0]*np.log(px[px>0])).sum();hy=-(py[py>0]*np.log(py[py>0])).sum();return float(2*mi/(hx+hy))
def surfaces(a,b,spacing):
 a=np.asarray(a,bool);b=np.asarray(b,bool);sa=a^binary_erosion(a);sb=b^binary_erosion(b);db=distance_transform_edt(~sb,sampling=spacing);da=distance_transform_edt(~sa,sampling=spacing);ab=db[sa];ba=da[sb];both=np.r_[ab,ba]
 return {'hd95_mm':float(np.percentile(both,95)),'mean_surface_distance_mm':float(both.mean()),'asd_a_to_b_mm':float(ab.mean()),'asd_b_to_a_mm':float(ba.mean()),'max_hausdorff_mm':float(both.max())}
def com_mm(mask,aff):return nib.affines.apply_affine(aff,np.array(center_of_mass(mask)))
def metrics(mri,pet,brain):
 pm=ants.get_mask(pet).numpy()>0;bm=brain.numpy()>0;aff=nib.load(SD/f'sub-{SUB}_T1w_RAS_N4.nii.gz').affine;inter=pm&bm
 return {'nmi':nmi(mri.numpy(),pet.numpy()),'dice':dice(bm,pm),'brain_coverage':float(inter.sum()/bm.sum()),'pet_precision_in_brain':float(inter.sum()/pm.sum()),'com_distance_mm':float(np.linalg.norm(com_mm(bm,aff)-com_mm(pm,aff))),**surfaces(bm,pm,brain.spacing),'pet_mask_voxels':int(pm.sum()),'brain_mask_voxels':int(bm.sum())},pm
def main():
 brain=ants.image_read(str(SD/f'sub-{SUB}_T1w_RAS_N4_brain.nii.gz'));mask=ants.image_read(str(SD/f'sub-{SUB}_SynthStrip_mask.nii.gz'));mean=ants.image_read(str(SD/f'sub-{SUB}_PET_mean_after_motion.nii.gz'));before=ants.resample_image_to_target(mean,brain,interp_type='linear');after=ants.image_read(str(SD/f'sub-{SUB}_PET_in_T1w.nii.gz'))
 pre,pre_m=metrics(brain,before,mask);post,post_m=metrics(brain,after,mask)
 tx=SD/'transforms'/'pet_to_mri_fwd_0.mat';t=ants.read_transform(str(tx));p=np.asarray(t.parameters,float);det=float(np.linalg.det(p[:9].reshape(3,3)))
 geometry={'brain_shape':brain.shape,'before_shape':before.shape,'after_shape':after.shape,'same_spacing':before.spacing==after.spacing==brain.spacing,'same_origin':np.allclose(before.origin,after.origin) and np.allclose(after.origin,brain.origin),'same_direction':np.allclose(before.direction,after.direction) and np.allclose(after.direction,brain.direction),'fixed':'MRI N4 multiplied by SynthStrip mask','moving':'PET mean after motion correction','pet_mask_method':'ants.get_mask on PET resampled into MRI grid, identically before and after','transform':str(tx),'transform_determinant':det,'transform_parameters':p.tolist()}
 delta={k:post[k]-pre[k] for k in ['nmi','dice','brain_coverage','pet_precision_in_brain','com_distance_mm','hd95_mm','mean_surface_distance_mm']}
 out={'subject_id':SUB,'before':pre,'after':post,'delta_after_minus_before':delta,'geometry_and_transform_validation':geometry,'visual_status':'PENDING','universal_rule':{'nmi_improves':post['nmi']>pre['nmi'],'absolute_dice_drop_lt_0.01':pre['dice']-post['dice']<.01,'transform_valid':det>0 and geometry['same_spacing'] and geometry['same_origin'] and geometry['same_direction']},'decision':'PENDING_VISUAL'}
 (BASE/'036_S_1001_pet_registration_adjudication.json').write_text(json.dumps(out,indent=2)+'\n')
 rows=[]
 for stage,d in [('before',pre),('after',post)]:rows.append({'subject_id':SUB,'stage':stage,**d})
 pd.DataFrame(rows).to_csv(R/'reports/adni_stage_c2_036_S_1001_adjudication_metrics.csv',index=False)
 # Multi-slice MRI/PET overlays and PET-mask contours.
 ma=brain.numpy();fig,ax=plt.subplots(4,6,figsize=(18,11))
 for r,(name,pet,pm) in enumerate([('before',before,pre_m),('after',after,post_m)]):
  pa=pet.numpy();v=pa[pa!=0];lo,hi=np.percentile(v,[1,99]);
  for c,(axis,frac) in enumerate([(2,.35),(2,.5),(2,.65),(1,.4),(1,.55),(0,.5)]):
   i=int(ma.shape[axis]*frac);m=np.rot90(np.take(ma,i,axis));x=np.rot90(np.take(pa,i,axis));ax[r*2,c].imshow(m,cmap='gray');ax[r*2,c].imshow(x,cmap='hot',alpha=.35,vmin=lo,vmax=hi);ax[r*2,c].set_title(f'{name} overlay a{axis} {i}',fontsize=7);ax[r*2,c].axis('off');ax[r*2+1,c].imshow(m,cmap='gray');ax[r*2+1,c].contour(np.rot90(np.take(mask.numpy(),i,axis)),[.5],colors='lime',linewidths=.6);ax[r*2+1,c].contour(np.rot90(np.take(pm,i,axis)),[.5],colors='red',linewidths=.6);ax[r*2+1,c].set_title('MRI mask green / PET mask red',fontsize=7);ax[r*2+1,c].axis('off')
 fig.suptitle(f'{SUB} PET→MRI adjudication');fig.tight_layout();fig.savefig(FIG/'before_after_multislice.png',dpi=150);plt.close(fig)
 # Checkerboard before/after using normalized MRI and PET.
 fig,ax=plt.subplots(2,3,figsize=(10,6))
 for r,(name,pet) in enumerate([('before',before),('after',after)]):
  for c,axis in enumerate([2,1,0]):
   i=ma.shape[axis]//2;m=np.rot90(np.take(ma,i,axis));x=np.rot90(np.take(pet.numpy(),i,axis));m=(m-m.min())/(m.max()-m.min()+1e-9);x=(x-x.min())/(x.max()-x.min()+1e-9);cb=np.where((np.indices(m.shape).sum(0)//10)%2,m,x);ax[r,c].imshow(cb,cmap='gray');ax[r,c].set_title(name+' checkerboard',fontsize=8);ax[r,c].axis('off')
 fig.tight_layout();fig.savefig(FIG/'before_after_checkerboard.png',dpi=150);plt.close(fig)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
