#!/usr/bin/env python3
"""Stage C2 native-to-MNI preprocessing for exactly the audited 54 subjects."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
import ants,nibabel as nib,numpy as np,pandas as pd
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];NATIVE=R/'data/derived/adni/nifti_native';MAN=R/'data/derived/adni/adni_native_manifest.csv'
BASE=R/'data/derived/adni/stage_c2';OUT=BASE/'subjects';FIG=R/'reports/figures/adni_stage_c2';QC=R/'reports/adni_stage_c2_qc.csv';CONF=R/'configs/adni_stage_c2.yaml'
TPL=R/'data/derived/adni/stage_c1/templateflow/tpl-MNI152NLin2009cAsym';HEAD=TPL/'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz';BRAIN=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz';MASK=TPL/'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
IMAGE='freesurfer/synthstrip:1.8';DIGEST='sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dice(a,b):
 a=np.asarray(a)>0;b=np.asarray(b)>0;d=a.sum()+b.sum();return float(2*(a&b).sum()/d) if d else 0
def nmi(a,b,bins=64):
 x=np.asarray(a).ravel();y=np.asarray(b).ravel();ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if x.size>300000:i=np.linspace(0,x.size-1,300000,dtype=int);x=x[i];y=y[i]
 h=np.histogram2d(x,y,bins)[0];p=h/max(h.sum(),1);px=p.sum(1);py=p.sum(0);z=p>0;mi=(p[z]*np.log(p[z]/(px[:,None]*py[None,:])[z])).sum();hx=-(px[px>0]*np.log(px[px>0])).sum();hy=-(py[py>0]*np.log(py[py>0])).sum();return float(2*mi/(hx+hy)) if hx+hy else 0
def affstats(path):
 p=np.asarray(ants.read_transform(str(path)).parameters,float);m=p[:9].reshape(3,3);return float(np.linalg.det(m)),p[9:12].tolist(),Rotation.from_matrix(m).as_euler('xyz',degrees=True).tolist()
def cptrans(paths,d,prefix):
 out=[]
 for i,x in enumerate(paths):
  q=d/f'{prefix}_{i}.{"nii.gz" if str(x).endswith(".nii.gz") else "mat"}';shutil.copy2(x,q);out.append(str(q))
 return out
def stats(path):
 im=nib.load(path);a=np.asarray(im.dataobj);return {'shape':'x'.join(map(str,im.shape)),'spacing':'x'.join(f'{x:.6g}' for x in im.header.get_zooms()),'orientation':''.join(nib.aff2axcodes(im.affine)),'affine':json.dumps(im.affine.tolist(),separators=(',',':')),'finite':bool(np.isfinite(a).all()),'nan_count':int(np.isnan(a).sum()),'inf_count':int(np.isinf(a).sum()),'min':float(np.nanmin(a)),'max':float(np.nanmax(a)),'mean':float(np.nanmean(a)),'std':float(np.nanstd(a)),'nonzero_fraction':float(np.count_nonzero(a)/a.size)}
def synth(ras,strip,mask,sd):
 if mask.exists() and strip.exists():return
 cmd=['docker','run','--rm','--network=none','-v',f'{sd.resolve()}:/data',IMAGE,'-i',f'/data/{ras.name}','-o',f'/data/{strip.name}','-m',f'/data/{mask.name}','-b','1'];(sd/'synthstrip_command.txt').write_text(' '.join(cmd)+'\n');res=subprocess.run(cmd,text=True,capture_output=True);(sd/'synthstrip.stdout.log').write_text(res.stdout);(sd/'synthstrip.stderr.log').write_text(res.stderr)
 if res.returncode:raise RuntimeError(f'SynthStrip exit {res.returncode}: {res.stderr[-1000:]}')
def fig(sub,mri,mask,mrimni,petmri,petmni):
 items=[('N4 + mask',mri,mask),('MRI MNI',mrimni,None),('PET in MRI',petmri,None),('PET MNI',petmni,None)];f,ax=plt.subplots(4,3,figsize=(10,12))
 for r,(title,img,ma) in enumerate(items):
  a=img.numpy();v=a[np.isfinite(a)&(a!=0)];lo,hi=np.percentile(v,[1,99]) if v.size else (0,1)
  for c,axis in enumerate([2,1,0]):
   i=a.shape[axis]//2;ax[r,c].imshow(np.rot90(np.take(a,i,axis)),cmap='gray' if 'MRI' in title or 'N4' in title else 'hot',vmin=lo,vmax=hi)
   if ma is not None:ax[r,c].contour(np.rot90(np.take(ma.numpy(),i,axis)),levels=[.5],colors='lime',linewidths=.5)
   ax[r,c].set_title(title,fontsize=7);ax[r,c].axis('off')
 f.suptitle(sub);f.tight_layout();f.savefig(FIG/f'{sub}_c2_qc.png',dpi=140);plt.close(f)
 # Explicit MNI PET/MRI overlay.
 a=mrimni.numpy();p=petmni.numpy();f,ax=plt.subplots(1,3,figsize=(9,3))
 for c,axis in enumerate([2,1,0]):
  i=a.shape[axis]//2;x=np.rot90(np.take(a,i,axis));y=np.rot90(np.take(p,i,axis));v=y[y!=0];lo,hi=np.percentile(v,[1,99]) if v.size else (0,1);ax[c].imshow(x,cmap='gray');ax[c].imshow(y,cmap='hot',alpha=.35,vmin=lo,vmax=hi);ax[c].axis('off')
 f.suptitle(f'{sub} PET→MRI→MNI single-resampling overlay');f.tight_layout();f.savefig(FIG/f'{sub}_c2_pet_mri_mni.png',dpi=140);plt.close(f)
def process(sub):
 m=pd.read_csv(MAN);rows=m[m.subject_id.eq(sub)];mr=rows[rows.modality.eq('MRI')].iloc[0];pr=rows[rows.modality.eq('PET')].iloc[0];sd=OUT/f'sub-{sub}';td=sd/'transforms';sd.mkdir(parents=True,exist_ok=True);td.mkdir(exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 nm=Path(mr.nifti_path);np_pet=Path(pr.nifti_path);stamp=sd/'complete.json';signature={'mri':sha(nm),'pet':sha(np_pet),'config':sha(CONF)}
 if stamp.exists() and json.loads(stamp.read_text()).get('signature')==signature:
  print('SKIP_VALID',sub);return json.loads((sd/'qc.json').read_text())
 ras=sd/f'sub-{sub}_T1w_RAS.nii.gz';nib.save(nib.as_closest_canonical(nib.load(nm)),ras);mri=ants.image_read(str(ras));n4=ants.n4_bias_field_correction(mri,shrink_factor=4,convergence={'iters':[50,50,30,20],'tol':1e-7},rescale_intensities=False);n4p=sd/f'sub-{sub}_T1w_RAS_N4.nii.gz';ants.image_write(n4,str(n4p))
 maskp=sd/f'sub-{sub}_SynthStrip_mask.nii.gz';stripp=sd/f'sub-{sub}_T1w_RAS_SynthStrip.nii.gz';synth(ras,stripp,maskp,sd);mask=ants.image_read(str(maskp));ma=mask.numpy()>0.5;cc,nc=label(ma,np.ones((3,3,3)));sizes=np.bincount(cc.ravel())[1:];vol=float(ma.sum()*np.prod(mask.spacing)/1000);brain=n4*mask;brainp=sd/f'sub-{sub}_T1w_RAS_N4_brain.nii.gz';ants.image_write(brain,str(brainp))
 fixed=ants.image_read(str(BRAIN));fmask=ants.image_read(str(MASK));reg=ants.registration(fixed,brain,type_of_transform='SyNRA',outprefix=str(td/'mri_to_mni_'),mask=fmask,moving_mask=mask,mask_all_stages=True,aff_metric='mattes',aff_sampling=32,aff_random_sampling_rate=.2,syn_metric='CC',syn_sampling=4,reg_iterations=(40,20,0),random_seed=20260909,singleprecision=True,verbose=False);mrifwd=cptrans(reg['fwdtransforms'],td,'mri_to_mni_fwd');mriinv=cptrans(reg['invtransforms'],td,'mri_to_mni_inv');mrimni=reg['warpedmovout'];mrimnip=sd/f'sub-{sub}_T1w_MNI2mm.nii.gz';ants.image_write(mrimni,str(mrimnip));wm=ants.apply_transforms(fmask,mask,reg['fwdtransforms'],interpolator='nearestNeighbor');wmp=sd/f'sub-{sub}_SynthStrip_mask_MNI2mm.nii.gz';ants.image_write(wm,str(wmp));warp=next(x for x in reg['fwdtransforms'] if str(x).endswith('.nii.gz'));aff=next(x for x in reg['fwdtransforms'] if str(x).endswith('.mat'));jac=ants.create_jacobian_determinant_image(fixed,warp,do_log=False,geom=False);ja=jac.numpy();jacp=sd/f'sub-{sub}_mri_to_mni_jacobian.nii.gz';ants.image_write(jac,str(jacp));adet,atrans,arot=affstats(aff)
 pet=ants.image_read(str(np_pet));fd=[];mdet=[]
 if pet.dimension==4:
  mc=ants.motion_correction(pet,type_of_transform='Rigid',fdOffset=50,outprefix=str(td/'pet_motion_'),random_seed=20260909,verbose=False);pet4=mc['motion_corrected'];pet4p=sd/f'sub-{sub}_PET_motion_corrected_4D.nii.gz';ants.image_write(pet4,str(pet4p));fd=np.asarray(mc['FD'],float).tolist();petmean=ants.get_average_of_timeseries(pet4)
  for x in mc['motion_parameters']:
   if x=='NA' or x is None:mdet.append(1.)
   else:mdet.append(affstats(x[0] if isinstance(x,list) else x)[0])
 else:petmean=pet;pet4p='';fd=[];mdet=[]
 meanp=sd/f'sub-{sub}_PET_mean_after_motion.nii.gz';ants.image_write(petmean,str(meanp));pre=ants.resample_image_to_target(petmean,brain,interp_type='linear');pren=nmi(brain.numpy(),pre.numpy());pred=dice(mask.numpy(),ants.get_mask(pre).numpy());preg=ants.registration(brain,petmean,type_of_transform='Rigid',outprefix=str(td/'pet_to_mri_'),random_seed=20260909,singleprecision=True,verbose=False);petfwd=cptrans(preg['fwdtransforms'],td,'pet_to_mri_fwd');petinv=cptrans(preg['invtransforms'],td,'pet_to_mri_inv');petmri=preg['warpedmovout'];petmrip=sd/f'sub-{sub}_PET_in_T1w.nii.gz';ants.image_write(petmri,str(petmrip));postn=nmi(brain.numpy(),petmri.numpy());postd=dice(mask.numpy(),ants.get_mask(petmri).numpy());pdet,ptrans,prot=affstats(preg['fwdtransforms'][-1]);chain=reg['fwdtransforms']+preg['fwdtransforms'];petmni=ants.apply_transforms(ants.image_read(str(HEAD)),petmean,chain,interpolator='linear',singleprecision=True);petmnip=sd/f'sub-{sub}_PET_MNI2mm.nii.gz';ants.image_write(petmni,str(petmnip));(td/'pet_to_mni_chain.json').write_text(json.dumps({'ANTs_transformlist':mrifwd+petfwd,'application_order':'as listed','resampling_count':1,'interpolator':'linear'},indent=2)+'\n')
 reasons=[];warnings=[];mdice=dice(fmask.numpy(),wm.numpy());mcover=float(((fmask.numpy()>0)&(wm.numpy()>0)).sum()/max((fmask.numpy()>0).sum(),1));mq=stats(maskp);fq=stats(petmnip)
 if nc!=1 or (sizes.size and sizes.max()/ma.sum()<.995) or not 700<=vol<=1900:reasons.append('invalid SynthStrip mask topology/volume')
 if mdice<.8:reasons.append(f'MRI-MNI Dice {mdice:.3f}<0.80')
 if adet<=0 or pdet<=0 or any(x<=0 for x in mdet):reasons.append('non-positive rigid/affine determinant')
 if not np.isfinite(ja).all() or ja.min()<=0:reasons.append('non-positive/invalid Jacobian')
 if postn<pren or postd<pred:reasons.append('PET-MRI registration metric worsened')
 if not fq['finite'] or fq['nonzero_fraction']==0 or fq['shape']!='97x115x97' or fq['orientation']!='RAS':reasons.append('invalid final PET')
 if sub=='009_S_1199':warnings.append('47-slice native PET coverage exception retained')
 if sub=='153_S_4172':warnings.append('ADNI2 PET-DICOM/LAS protocol exception retained; laterality visual QC required')
 status='FAIL' if reasons else ('WARNING' if warnings else 'PASS');row={'subject_id':sub,'group':mr.group,'status':status,'reasons':'; '.join(reasons+warnings),'mri_image_id':mr.image_id,'pet_image_id':pr.image_id,'mri_series_id':mr.series_id,'pet_series_id':pr.series_id,'mri_native_sha256':signature['mri'],'pet_native_sha256':signature['pet'],'mask_volume_ml':vol,'mask_components':int(nc),'mask_largest_component_fraction':float(sizes.max()/ma.sum()),'mri_mni_dice':mdice,'mri_mni_coverage':mcover,'mri_affine_determinant':adet,'mri_affine_translation_mm':json.dumps(atrans),'mri_affine_rotation_deg':json.dumps(arot),'jacobian_min':float(ja.min()),'jacobian_max':float(ja.max()),'jacobian_nonpositive_fraction':float(np.mean(ja<=0)),'pet_frames':int(pet.shape[3]) if pet.dimension==4 else 1,'pet_motion_fd_mm':json.dumps(fd),'pet_motion_determinants':json.dumps(mdet),'pet_mri_nmi_before':pren,'pet_mri_nmi_after':postn,'pet_mri_dice_before':pred,'pet_mri_dice_after':postd,'pet_rigid_determinant':pdet,'pet_rigid_translation_mm':json.dumps(ptrans),'pet_rigid_rotation_deg':json.dumps(prot),'mri_mni_path':str(mrimnip),'pet_mni_path':str(petmnip),'mask_path':str(maskp),'pet_motion_corrected_path':str(pet4p),'pet_mean_path':str(meanp),'mri_forward_transforms':json.dumps(mrifwd),'mri_inverse_transforms':json.dumps(mriinv),'pet_forward_transform':json.dumps(petfwd),'pet_inverse_transform':json.dumps(petinv),'pet_to_mni_single_resampling':True,**{f'final_pet_{k}':v for k,v in fq.items()}}
 (sd/'qc.json').write_text(json.dumps(row,indent=2)+'\n');fig(sub,n4,mask,mrimni,petmri,petmni)
 if status=='FAIL':raise RuntimeError('QC FAIL '+sub+': '+'; '.join(reasons))
 stamp.write_text(json.dumps({'signature':signature,'status':status,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n');return row
def collect():
 rows=[]
 for p in sorted(OUT.glob('sub-*/qc.json')):rows.append(json.loads(p.read_text()))
 if rows:
  with open(QC,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))));w.writeheader();w.writerows(rows)
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--subject');args=ap.parse_args();m=pd.read_csv(MAN);subs=sorted(m.subject_id.unique())
 if len(subs)!=54:raise RuntimeError('manifest is not 54 subjects')
 if args.subject:
  if args.subject not in subs:raise ValueError(args.subject)
  row=process(args.subject);print(json.dumps({'subject':args.subject,'status':row['status']}));return
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 for i,s in enumerate(subs,1):
  print(f'C2 {i}/54 START {s}',flush=True);cmd=[sys.executable,str(Path(__file__).resolve()),'--subject',s];r=subprocess.run(cmd)
  collect()
  if r.returncode:print('STOP_ON_FAIL',s,flush=True);raise SystemExit(r.returncode)
  print(f'C2 {i}/54 DONE {s}',flush=True)
 rows=collect();print(json.dumps({'subjects':len(rows),'counts':pd.Series([x['status'] for x in rows]).value_counts().to_dict(),'C2_complete':len(rows)==54,'next_stage_started':False},indent=2))
if __name__=='__main__':main()
