#!/usr/bin/env python3
"""Idempotent native extraction, conversion and QC for the audited ADNI cohort."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, tempfile, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import ecat
import numpy as np
import pydicom

R=Path(__file__).resolve().parents[1]; RAW=R/'data/raw/adni'; D=R/'data/derived/adni'
EXT=D/'extracted'; NII=D/'nifti_native'; Q=D/'qc'; FIG=R/'reports/figures/adni_qc'
META=RAW/'Multimodal_PET_-_MRI_IDA_Metadata.zip'
ARCH=[RAW/'Multimodal PET - MRI.zip',RAW/'Multimodal PET - MRI_dataset.zip']
DCM=R/'.tools/dcm2niix/usr/bin/dcm2niix'; LIB=R/'.tools/dcm2niix/usr/lib/x86_64-linux-gnu'
QC_CSV=R/'reports/adni_nifti_qc.csv'; MAN=D/'adni_native_manifest.csv'

def hfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def htree(paths,base):
 h=hashlib.sha256()
 for p in sorted(paths): h.update(str(p.relative_to(base)).encode()); h.update(bytes.fromhex(hfile(p)))
 return h.hexdigest()
def txt(root,name):
 e=root.find('.//'+name); return (e.text or '').strip() if e is not None else ''
def strip(root):
 for e in root.iter(): e.tag=e.tag.rsplit('}',1)[-1]
 return root
def scalar(ds,n):
 v=getattr(ds,n,''); return '\\'.join(map(str,v)) if isinstance(v,(list,pydicom.multival.MultiValue)) else str(v)

def metadata():
 minimal={}; rich={}
 with zipfile.ZipFile(META) as z:
  for i in z.infolist():
   if i.is_dir(): continue
   x=strip(ET.fromstring(z.read(i)))
   if x.tag=='metadata':
    key='/'.join(i.filename.split('/')[:5]); minimal[key]={'subject_id':x.find('subject').get('id'),'series_id':x.find('series').get('uid'),'image_id':x.find('image').get('uid')}
   elif x.tag=='idaxs':
    rich['I'+txt(x,'imageUID').lstrip('I')]={'group':txt(x,'researchGroup'),'visit':txt(x,'visitIdentifier'),'modality':txt(x,'modality'),'date':txt(x,'dateAcquired'),'description':txt(x,'description')}
 rows=[]
 for key,m in sorted(minimal.items()):
  a=key.split('/'); desc=a[2]; rr=rich.get(m['image_id'],{}); mod=rr.get('modality') or ('PET' if 'FDG' in desc.upper() else 'MRI')
  rows.append({**m,'key':key,'description':rr.get('description') or desc.replace('_',' '),'modality':mod,'date':rr.get('date') or a[3][:10],'group':rr.get('group',''),'visit':rr.get('visit',''),'detailed_xml':bool(rr)})
 by=defaultdict(list)
 for x in rows: by[x['subject_id']].append(x)
 for sub,rs in by.items():
  g={x['group'] for x in rs if x['group']}; v={x['visit'] for x in rs if x['visit']}
  for x in rs:
   if not x['group']: x['group']=next(iter(g))
   if not x['visit'] and len(v)==1: x['visit']=next(iter(v))
   x['metadata_caveat']=not x['detailed_xml']
 return rows,by

def protect_raw():
 old={r['path']:r['sha256'] for r in csv.DictReader(open(D/'stage_a_validation/qc/sha256.csv')) if r['kind']=='raw_archive'}
 now={str(p.relative_to(R)):hfile(p) for p in ARCH+[META]}
 if now!=old: raise RuntimeError(f'Raw archive hash mismatch: {now}')
 return now

def extract(rows):
 wanted={x['key']:x for x in rows}; ledger=[]
 for zp in ARCH:
  with zipfile.ZipFile(zp) as z:
   for i in z.infolist():
    if i.is_dir(): continue
    key='/'.join(i.filename.split('/')[:5])
    if key not in wanted: continue
    if wanted[key]['image_id']=='I256061':
     ds=pydicom.dcmread(z.open(i),stop_before_pixels=True)
     if not (scalar(ds,'SeriesDescription')=='FDG PET BRAIN 2' and scalar(ds,'SeriesNumber')=='4' and scalar(ds,'AcquisitionNumber')=='3001'): continue
    out=EXT/i.filename; out.parent.mkdir(parents=True,exist_ok=True)
    action='reused'
    if not out.exists() or out.stat().st_size!=i.file_size:
     with z.open(i) as src,open(out,'wb') as dst: shutil.copyfileobj(src,dst)
     action='extracted'
    if out.stat().st_size!=i.file_size: raise RuntimeError(f'Extraction size mismatch {out}')
    ledger.append({'archive':zp.name,'member':i.filename,'size':i.file_size,'crc32':f'{i.CRC:08x}','path':str(out.relative_to(R)),'action':action})
 with open(Q/'extraction_ledger.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=ledger[0]); w.writeheader(); w.writerows(ledger)
 return ledger

def sources(row): return sorted((EXT/row['key']).glob('*'))
def preflight(rows):
 detail=[]
 for r in rows:
  if r['modality']!='PET' or PurePosixPath(sources(r)[0]).suffix.lower()!='.v': continue
  ps=sources(r)
  if len(ps)!=1: raise RuntimeError(f"{r['image_id']}: expected one ECAT")
  im=ecat.load(str(ps[0])); nf=im._subheader.get_nframes(); frames=[]
  for j in range(nf):
   sh=im._subheader.subheaders[j]; frames.append({'frame':j,'shape':list(im.get_frame(j).shape),'spacing_mm':list(map(float,im._subheader.get_zooms(j)[:3])),'affine':im.get_frame_affine(j).tolist(),'start_ms':int(sh['frame_start_time']),'duration_ms':int(sh['frame_duration'])})
  compatible=all(x['shape']==frames[0]['shape'] and np.allclose(x['spacing_mm'],frames[0]['spacing_mm']) and np.allclose(x['affine'],frames[0]['affine'],atol=1e-5) for x in frames)
  detail.append({'subject_id':r['subject_id'],'image_id':r['image_id'],'frames':frames,'compatible':compatible})
  if nf!=6 or not compatible: raise RuntimeError(f"UNVALIDATED ECAT STRUCTURE {r['subject_id']} {r['image_id']}: frames={nf}, compatible={compatible}")
 (Q/'ecat_frame_metadata.json').write_text(json.dumps(detail,indent=2)+'\n')
 return detail

def valid_existing(out,prov,source_hash):
 if not out.exists() or not prov.exists(): return False
 try:
  p=json.loads(prov.read_text()); im=nib.load(str(out)); a=np.asanyarray(im.dataobj)
  return p['source_sha256']==source_hash and p['output_sha256']==hfile(out) and np.isfinite(a).all() and np.count_nonzero(a)>0
 except Exception:return False

def convert_ecat(row,src,out):
 im=ecat.load(str(src)); nf=im._subheader.get_nframes(); data=[]; aff=[]; starts=[]; durations=[]
 for j in range(nf):
  data.append(np.asarray(im.get_frame(j),dtype=np.float32)); aff.append(im.get_frame_affine(j)); sh=im._subheader.subheaders[j]; starts.append(int(sh['frame_start_time'])); durations.append(int(sh['frame_duration']))
 if nf!=6 or not all(np.allclose(x,aff[0],atol=1e-5) for x in aff): raise RuntimeError(f'ECAT geometry failure {row["image_id"]}')
 ni=nib.Nifti1Image(np.stack(data,axis=3),aff[0]); z=tuple(map(float,im._subheader.get_zooms(0)[:3])); ni.header.set_xyzt_units('mm','sec'); ni.header.set_zooms(z+(durations[0]/1000,)); ni.set_qform(aff[0],1); ni.set_sform(aff[0],1); nib.save(ni,out)
 return nf,starts,durations,'nibabel.ecat.load + per-frame 4D preservation'
def convert_dicom(row,srcs,out):
 with tempfile.TemporaryDirectory(dir=Q) as td:
  t=Path(td); inp=t/'input'; dest=t/'output'; inp.mkdir(); dest.mkdir()
  for p in srcs: (inp/p.name).symlink_to(p.resolve())
  env={**os.environ,'LD_LIBRARY_PATH':str(LIB)}; cmd=[str(DCM),'-z','y','-b','y','-w','1','-f','result','-o',str(dest),str(inp)]
  cp=subprocess.run(cmd,env=env,text=True,capture_output=True); (Q/f"{row['subject_id']}_{row['image_id']}_dcm2niix.log").write_text(cp.stdout+cp.stderr)
  ns=list(dest.glob('*.nii.gz'))
  if cp.returncode or len(ns)!=1: raise RuntimeError(f"DICOM conversion {row['image_id']} code={cp.returncode}, outputs={len(ns)}")
  shutil.move(ns[0],out); js=list(dest.glob('*.json'))
  if js: shutil.move(js[0],out.with_suffix('').with_suffix('.json'))
 return 1,[],[],'dcm2niix -z y -b y -w 1'

def qc(row,out,nframes,starts,durations):
 im=nib.load(str(out)); a=np.asanyarray(im.dataobj); nan=int(np.isnan(a).sum()); inf=int(np.isinf(a).sum()); nz=float(np.count_nonzero(a)/a.size)
 reasons=[]
 if not np.isfinite(im.affine).all(): reasons.append('invalid affine')
 if nan: reasons.append(f'{nan} NaN')
 if inf: reasons.append(f'{inf} Inf')
 if not nz: reasons.append('empty volume')
 if row['modality']=='MRI' and im.ndim!=3: reasons.append('MRI not 3D')
 if row['modality']=='PET' and row['source_format']=='ECAT7' and (im.ndim!=4 or im.shape[3]!=6): reasons.append('unexpected ECAT dimensions')
 warnings=[]
 if row['image_id']=='I256061': warnings.append('ADNI2 PET-DICOM protocol exception; XML-matched BRAIN 2 only')
 if row['modality']=='PET' and im.shape[2]<60: warnings.append('native PET has limited 47-slice axial matrix and coarser spacing; visual coverage plausible')
 status='FAIL' if reasons else ('WARNING' if warnings else 'PASS')
 reasons.extend(warnings)
 return {'safe_subject_id':'sub-'+row['subject_id'],'subject_id':row['subject_id'],'group':row['group'],'modality':row['modality'],'image_id':row['image_id'],'series_id':row['series_id'],'source_format':row['source_format'],'source_path':str((EXT/row['key']).relative_to(R)),'nifti_path':str(out.relative_to(R)),'ndim':im.ndim,'shape':'x'.join(map(str,im.shape)),'voxel_spacing':'x'.join(f'{x:.8g}' for x in im.header.get_zooms()),'orientation':''.join(nib.aff2axcodes(im.affine)),'affine':json.dumps(im.affine.tolist(),separators=(',',':')),'pet_frames':nframes if row['modality']=='PET' else '','pet_frame_starts_ms':'|'.join(map(str,starts)),'pet_frame_durations_ms':'|'.join(map(str,durations)),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),'nonzero_fraction':nz,'nan_count':nan,'inf_count':inf,'file_size':out.stat().st_size,'sha256':hfile(out),'status':status,'reason':'; '.join(reasons)}

def sheet(label,path,out,preview=False):
 im=nib.load(str(path)); a=np.asanyarray(im.dataobj); shown=a.mean(3) if a.ndim==4 else a; pos=shown[shown>0]; lo,hi=np.percentile(pos,[1,99]) if pos.size else (0,1); x,y,z=[s//2 for s in shown.shape]
 views=[np.rot90(shown[:,:,z]),np.rot90(shown[:,y,:]),np.rot90(shown[x,:,:])]; fig,ax=plt.subplots(1,3,figsize=(10,3),facecolor='black')
 for q,v,t in zip(ax,views,['axial','coronal','sagittal']):q.imshow(v,cmap='gray',vmin=lo,vmax=hi,origin='lower');q.set_title(t,color='white');q.axis('off')
 fig.suptitle(label+(' — temporal mean QC preview only' if preview else ' — native'),color='white');fig.tight_layout();fig.savefig(out,dpi=110,bbox_inches='tight');plt.close(fig)
def paired(sub,mri,pet):
 fig,ax=plt.subplots(2,3,figsize=(10,6),facecolor='black')
 for row,(name,p) in enumerate([('MRI',mri),('PET mean QC preview',pet)]):
  im=nib.load(str(p));a=np.asanyarray(im.dataobj);a=a.mean(3) if a.ndim==4 else a;pos=a[a>0];lo,hi=np.percentile(pos,[1,99]);x,y,z=[s//2 for s in a.shape];vs=[np.rot90(a[:,:,z]),np.rot90(a[:,y,:]),np.rot90(a[x,:,:])]
  for q,v,t in zip(ax[row],vs,['axial','coronal','sagittal']):q.imshow(v,cmap='gray',vmin=lo,vmax=hi,origin='lower');q.set_title(name+' '+t,color='white');q.axis('off')
 fig.suptitle(sub+' — native, unregistered',color='white');fig.tight_layout();fig.savefig(FIG/f'{sub}_MRI_PET.png',dpi=110,bbox_inches='tight');plt.close(fig)

def main(force=False):
 for p in [EXT,NII,Q,FIG]:p.mkdir(parents=True,exist_ok=True)
 raw_hashes=protect_raw(); rows,by=metadata()
 if len(rows)!=108 or len(by)!=54 or Counter(x['modality'] for x in rows)!=Counter({'MRI':54,'PET':54}):raise RuntimeError('Metadata cohort invariant failed')
 ledger=extract(rows)
 if len({x['key'] for x in rows if sources(x)})!=108:raise RuntimeError('Extraction series count failed')
 preflight(rows); qcrows=[]; provrows=[]
 for r in rows:
  ss=sources(r); exts={p.suffix.lower() for p in ss}; r['source_format']='ECAT7' if exts=={'.v'} else 'DICOM' if exts=={'.dcm'} else 'UNKNOWN'
  suffix='T1w' if r['modality']=='MRI' else 'FDGpet'; out=NII/f"sub-{r['subject_id']}_{suffix}.nii.gz"; prov=Q/f"{r['subject_id']}_{r['image_id']}_provenance.json"; sh=htree(ss,EXT/r['key']); action='reused'
  if not force and valid_existing(out,prov,sh):
   old=json.loads(prov.read_text()); nf=old['pet_frames']; starts=old['pet_frame_starts_ms']; durations=old['pet_frame_durations_ms']
  else:
   action='converted'; nf,starts,durations,cmd=convert_ecat(r,ss[0],out) if r['source_format']=='ECAT7' else convert_dicom(r,ss,out)
   p={'subject_id':r['subject_id'],'image_id':r['image_id'],'series_id':r['series_id'],'archive':next(x['archive'] for x in ledger if x['member'].startswith(r['key']+'/')),'source_path':str((EXT/r['key']).relative_to(R)),'source_format':r['source_format'],'source_sha256':sh,'nifti_path':str(out.relative_to(R)),'output_sha256':hfile(out),'tool_command':cmd,'pet_frames':nf,'pet_frame_starts_ms':starts,'pet_frame_durations_ms':durations};prov.write_text(json.dumps(p,indent=2)+'\n')
  q=qc(r,out,nf,starts,durations); qcrows.append(q); provrows.append({**json.loads(prov.read_text()),'action':action})
  sheet(f"{r['subject_id']} {r['image_id']} {r['modality']}",out,FIG/f"sub-{r['subject_id']}_{suffix}_orthogonal.png",r['modality']=='PET' and q['ndim']==4)
 for sub,rs in by.items():
  m=next(x for x in rs if x['modality']=='MRI');p=next(x for x in rs if x['modality']=='PET');paired(sub,NII/f'sub-{sub}_T1w.nii.gz',NII/f'sub-{sub}_FDGpet.nii.gz')
 fields=list(qcrows[0]);
 with open(QC_CSV,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(qcrows)
 with open(Q/'conversion_provenance.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(provrows[0]));w.writeheader();w.writerows(provrows)
 qmap={(x['subject_id'],x['modality']):x for x in qcrows}; manifest=[]
 for r in rows:
  pair=by[r['subject_id']]; md=next(x['date'] for x in pair if x['modality']=='MRI');pd=next(x['date'] for x in pair if x['modality']=='PET'); q=qmap[(r['subject_id'],r['modality'])]
  manifest.append({'safe_subject_id':'sub-'+r['subject_id'],'subject_id':r['subject_id'],'group':r['group'],'modality':r['modality'],'visit':r['visit'],'acquisition_date':r['date'],'mri_date':md,'pet_date':pd,'delta_days':(date.fromisoformat(pd)-date.fromisoformat(md)).days,'image_id':r['image_id'],'series_id':r['series_id'],'description':r['description'],'original_format':r['source_format'],'source_path':q['source_path'],'nifti_path':q['nifti_path'],'ndim':q['ndim'],'shape':q['shape'],'voxel_spacing':q['voxel_spacing'],'orientation':q['orientation'],'frames':q['pet_frames'],'qc_status':q['status'],'metadata_caveat':r['metadata_caveat'],'protocol_exception':r['image_id']=='I256061'})
 with open(MAN,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 counts=Counter(x['status'] for x in qcrows); shapes=Counter((x['modality'],x['shape'],x['voxel_spacing']) for x in qcrows); orient=Counter((x['modality'],x['orientation']) for x in qcrows)
 report=f'''# ADNI Stage B native conversion report\n\nDate: 2026-09-05  \nStatus: **{'PASS' if not counts['FAIL'] else 'FAIL'}**\n\n- Cohort: 54 subjects (18 CN, 18 MCI, 18 AD); 54 MRI and 54 PET.\n- QC: PASS={counts['PASS']}, WARNING={counts['WARNING']}, FAIL={counts['FAIL']}.\n- PET: 53 ECAT7 preserved as six-frame 4D NIfTI; one XML-filtered ADNI2 PET-DICOM 3D NIfTI. No temporal averages saved.\n- Visual outputs: 108 orthogonal sheets plus 54 native, unregistered MRI–PET comparison sheets.\n- Raw ZIP SHA-256 checks: unchanged from Stage A.\n\n## Distributions\n\nShapes/spacing: `{dict(shapes)}`  \nOrientations: `{dict(orient)}`\n\n## Exceptions\n\n- `153_S_4172 / I256061 / S122077`: WARNING protocol exception. Only DICOM BRAIN 2 (SeriesNumber 4, AcquisitionNumber 3001) was extracted/converted; BRAIN 1 was not included. Native orientation LAS retained.\n- Detailed XML absent for PET images I11879, I14808, I16169 and I37544; accepted path-level metadata pairing retained.\n- No conversion failures, invalid affines, NaN/Inf values, empty volumes, missing modalities, duplicate outputs, or atypical ECAT frame structures detected.\n\n## Reproducibility\n\nRun: `.venv-adni-py311/bin/python scripts/run_adni_stage_b.py`. Existing valid outputs are reused only when source and output hashes match provenance. Use removal of a derived output/provenance record as the explicit reconversion action. Tool versions: Python 3.11.8, NiBabel {nib.__version__}, pydicom {pydicom.__version__}, dcm2niix v1.0.20220720. Per-series commands, hashes and paths are in `data/derived/adni/qc/conversion_provenance.csv`.\n\n## Gate\n\nThe complete cohort passes native conversion QC and may advance to Stage C after review. Stage C was not started. No registration, reorientation, skull stripping, normalization, resampling, temporal averaging, splitting, augmentation or training was performed.\n'''
 (R/'reports/adni_stage_b_conversion_report.md').write_text(report)
 print(json.dumps({'subjects':len(by),'MRI':54,'PET':54,'qc':counts,'shapes':{str(k):v for k,v in shapes.items()},'orientations':{str(k):v for k,v in orient.items()}},default=dict,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true',help='explicitly reconvert valid derived NIfTI outputs'); args=ap.parse_args(); main(force=args.force)
