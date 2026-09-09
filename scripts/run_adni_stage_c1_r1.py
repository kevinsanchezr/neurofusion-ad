#!/usr/bin/env python3
"""Stage C1-R1: five-subject SynthStrip replacement validation only."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "data/derived/adni/nifti_native"
OLD = ROOT / "data/derived/adni/stage_c1/subjects"
BASE = ROOT / "data/derived/adni/stage_c1_r1"
OUT = BASE / "subjects"
FIG = ROOT / "reports/figures/adni_stage_c1_r1"
QC = ROOT / "reports/adni_stage_c1_r1_qc.csv"
COMPARE = BASE / "previous_mask_vs_synthstrip.csv"
HASHES = BASE / "sha256sums.csv"
CONFIG = ROOT / "configs/adni_stage_c1_r1.yaml"
TPL = ROOT / "data/derived/adni/stage_c1/templateflow/tpl-MNI152NLin2009cAsym"
TPL_HEAD = TPL / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
TPL_BRAIN = TPL / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"
TPL_MASK = TPL / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
IMAGE = "freesurfer/synthstrip:1.8"
IMAGE_DIGEST = "sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06"
MODEL_SHA = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"
SUBJECTS = [
    ("005_S_0223", "CN", "standard ECAT7"),
    ("005_S_0222", "MCI", "standard ECAT7"),
    ("005_S_0221", "AD", "standard ECAT7"),
    ("009_S_1199", "MCI", "47-slice ECAT7 exception"),
    ("153_S_4172", "AD", "ADNI2 PET-DICOM/LAS exception"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_nifti(img: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def nmi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size > 300_000:
        idx = np.linspace(0, x.size - 1, 300_000, dtype=int)
        x, y = x[idx], y[idx]
    h = np.histogram2d(x, y, bins=bins)[0]
    p = h / max(h.sum(), 1)
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return 2 * mi / (hx + hy) if hx + hy else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a) > 0, np.asarray(b) > 0
    den = aa.sum() + bb.sum()
    return float(2 * np.logical_and(aa, bb).sum() / den) if den else 0.0


def affine_stats(path: str) -> tuple[float, list[float], list[float]]:
    p = np.asarray(ants.read_transform(path).parameters, dtype=float)
    mat = p[:9].reshape(3, 3)
    return (float(np.linalg.det(mat)), p[9:12].tolist(),
            Rotation.from_matrix(mat).as_euler("xyz", degrees=True).tolist())


def copy_transforms(paths: list[str], dest: Path, prefix: str) -> list[str]:
    copied = []
    for i, src in enumerate(paths):
        ext = "nii.gz" if str(src).endswith(".nii.gz") else "mat"
        target = dest / f"{prefix}_{i}.{ext}"
        shutil.copy2(src, target)
        copied.append(str(target))
    return copied


def synthstrip(ras: Path, stripped: Path, mask: Path, command_log: Path) -> None:
    if stripped.exists() and mask.exists() and np.count_nonzero(nib.load(mask).dataobj):
        return
    cmd = [
        "docker", "run", "--rm", "--network=none",
        "-v", f"{ras.parent.resolve()}:/data", IMAGE,
        "-i", f"/data/{ras.name}", "-o", f"/data/{stripped.name}",
        "-m", f"/data/{mask.name}", "-b", "1",
    ]
    command_log.write_text(" ".join(cmd) + "\n")
    result = subprocess.run(cmd, text=True, capture_output=True)
    (command_log.parent / "synthstrip.stdout.log").write_text(result.stdout)
    (command_log.parent / "synthstrip.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"SynthStrip failed ({result.returncode}): {result.stderr[-2000:]}")
    if not stripped.exists() or not mask.exists():
        raise RuntimeError("SynthStrip did not create both output and mask")


def mask_qc(mask_path: Path, mri_path: Path) -> dict:
    mi, ii = nib.load(mask_path), nib.load(mri_path)
    m = np.asarray(mi.dataobj) > 0.5
    a = np.asarray(ii.dataobj)
    cc, ncomp = label(m, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(cc.ravel())[1:] if ncomp else np.array([], dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    coords = np.argwhere(m)
    bbox = [coords.min(0).tolist(), coords.max(0).tolist()] if coords.size else [[], []]
    extent_mm = ((coords.max(0) - coords.min(0) + 1) * np.array(mi.header.get_zooms()[:3])).tolist() if coords.size else []
    gaps = {}
    jumps = {}
    for axis, name in enumerate("xyz"):
        counts = np.count_nonzero(m, axis=tuple(i for i in range(3) if i != axis))
        occupied = np.flatnonzero(counts)
        gaps[name] = int((np.diff(occupied) > 1).sum()) if occupied.size else 0
        denom = max(int(counts.max()), 1)
        jumps[name] = float(np.max(np.abs(np.diff(counts))) / denom) if counts.size > 1 else 0.0
    voxvol = float(abs(np.linalg.det(mi.affine[:3, :3])))
    nonzero_mri = np.isfinite(a) & (a != 0)
    return {
        "brain_volume_ml": float(m.sum() * voxvol / 1000.0),
        "mask_fraction_full_fov": float(m.mean()),
        "mask_fraction_of_nonzero_mri": float(m.sum() / max(nonzero_mri.sum(), 1)),
        "connected_components": int(ncomp),
        "largest_component_voxels": largest,
        "largest_component_fraction": float(largest / max(m.sum(), 1)),
        "isolated_component_count": int(max(ncomp - 1, 0)),
        "isolated_voxels": int(m.sum() - largest),
        "bbox_voxels": json.dumps(bbox),
        "bbox_extent_mm": json.dumps(extent_mm),
        "slice_gap_counts_xyz": json.dumps(gaps),
        "max_slice_area_jump_xyz": json.dumps(jumps),
        "mask_finite": bool(np.isfinite(np.asarray(mi.dataobj)).all()),
        "mask_affine_finite": bool(np.isfinite(mi.affine).all()),
        "mask_affine_determinant": float(np.linalg.det(mi.affine[:3, :3])),
    }


def image_qc(path: Path) -> dict:
    im = nib.load(path)
    a = np.asarray(im.dataobj)
    return {
        "final_shape": "x".join(map(str, im.shape)),
        "final_spacing": "x".join(f"{x:.6g}" for x in im.header.get_zooms()),
        "final_orientation": "".join(nib.aff2axcodes(im.affine)),
        "final_affine": json.dumps(im.affine.tolist(), separators=(",", ":")),
        "finite": bool(np.isfinite(a).all()), "minimum": float(np.nanmin(a)),
        "maximum": float(np.nanmax(a)), "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)), "nonzero_fraction": float(np.count_nonzero(a) / a.size),
    }


def robust_limits(a: np.ndarray) -> tuple[float, float]:
    v = a[np.isfinite(a) & (a != 0)]
    return tuple(np.percentile(v, [1, 99])) if v.size else (0.0, 1.0)


def show_slice(ax, a, axis, idx, title, mask=None, cmap="gray", alpha=1.0):
    sl = np.take(a, idx, axis=axis)
    lo, hi = robust_limits(a)
    ax.imshow(np.rot90(sl), cmap=cmap, vmin=lo, vmax=hi, origin="lower", alpha=alpha)
    if mask is not None:
        ax.contour(np.rot90(np.take(mask, idx, axis=axis)), levels=[0.5], colors="lime", linewidths=0.7)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def figures(sub: str, p: dict[str, Path]) -> None:
    ras = nib.load(p["ras"]).get_fdata(); n4 = nib.load(p["n4"]).get_fdata()
    mask = nib.load(p["mask"]).get_fdata(); old = nib.load(p["oldmask"]).get_fdata()
    # Native/N4 contours, orthogonal plus inferior and anterior targeted views.
    fig, ax = plt.subplots(3, 5, figsize=(15, 9))
    centers = [s // 2 for s in ras.shape]
    for col, (axis, name) in enumerate([(2, "axial"), (1, "coronal"), (0, "sagittal")]):
        show_slice(ax[0, col], ras, axis, centers[axis], f"RAS {name}", mask)
        show_slice(ax[1, col], n4, axis, centers[axis], f"N4 {name}", mask)
        show_slice(ax[2, col], n4, axis, centers[axis], f"N4 old mask {name}", old)
    show_slice(ax[0, 3], ras, 2, int(ras.shape[2] * .28), "RAS inferior: cerebellum/pons", mask)
    show_slice(ax[1, 3], n4, 2, int(n4.shape[2] * .28), "N4 inferior: cerebellum/pons", mask)
    show_slice(ax[2, 3], n4, 2, int(n4.shape[2] * .28), "Old inferior", old)
    show_slice(ax[0, 4], ras, 1, int(ras.shape[1] * .76), "RAS anterior: eyes/face", mask)
    show_slice(ax[1, 4], n4, 1, int(n4.shape[1] * .76), "N4 anterior: eyes/face", mask)
    show_slice(ax[2, 4], n4, 1, int(n4.shape[1] * .76), "Old anterior", old)
    fig.suptitle(f"{sub}: SynthStrip contours (green)")
    fig.tight_layout(); fig.savefig(FIG / f"{sub}_mask_native_n4_targeted.png", dpi=150); plt.close(fig)

    mri_mni = nib.load(p["mri_mni"]).get_fdata(); pet_mni = nib.load(p["pet_mni"]).get_fdata()
    pet_native = nib.load(p["pet_before"]).get_fdata(); pet_mri = nib.load(p["pet_mri"]).get_fdata()
    brain = nib.load(p["brain"]).get_fdata()
    fig, ax = plt.subplots(4, 3, figsize=(10, 12))
    for col, axis in enumerate([2, 1, 0]):
        show_slice(ax[0, col], mri_mni, axis, mri_mni.shape[axis] // 2, "MRI→MNI")
        show_slice(ax[1, col], pet_mni, axis, pet_mni.shape[axis] // 2, "PET→MNI")
        show_slice(ax[2, col], brain, axis, brain.shape[axis] // 2, "MRI native space")
        # PET/MRI overlay in MRI grid.
        idx = brain.shape[axis] // 2
        show_slice(ax[3, col], brain, axis, idx, "PET→MRI overlay")
        pp = np.take(pet_mri, idx, axis=axis); lo, hi = robust_limits(pet_mri)
        ax[3, col].imshow(np.rot90(pp), cmap="hot", vmin=lo, vmax=hi, alpha=.35, origin="lower")
    fig.suptitle(f"{sub}: registrations; RAS radiological display, left/right retained")
    fig.tight_layout(); fig.savefig(FIG / f"{sub}_registrations.png", dpi=150); plt.close(fig)

    # PET motion preview (mean in memory is for QC only before output mean comparison).
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    for col, axis in enumerate([2, 1, 0]):
        show_slice(ax[0, col], pet_native, axis, pet_native.shape[axis] // 2, "PET before/motion reference")
        show_slice(ax[1, col], pet_mri, axis, pet_mri.shape[axis] // 2, "PET after motion + rigid to MRI")
    fig.suptitle(f"{sub}: PET motion/registration QC")
    fig.tight_layout(); fig.savefig(FIG / f"{sub}_pet_motion_registration.png", dpi=150); plt.close(fig)


def process_subject(sub: str, group: str, selection: str, template: dict) -> tuple[dict, dict]:
    sd = OUT / f"sub-{sub}"; tr = sd / "transforms"
    sd.mkdir(parents=True, exist_ok=True); tr.mkdir(exist_ok=True)
    native_mri = NATIVE / f"sub-{sub}_T1w.nii.gz"
    native_pet = NATIVE / f"sub-{sub}_FDGpet.nii.gz"
    oldmask = OLD / f"sub-{sub}" / f"sub-{sub}_brain_mask.nii.gz"
    for required in (native_mri, native_pet, oldmask):
        if not required.exists(): raise FileNotFoundError(required)

    ras = sd / f"sub-{sub}_T1w_RAS.nii.gz"
    if not ras.exists(): write_nifti(nib.as_closest_canonical(nib.load(native_mri)), ras)
    n4p = sd / f"sub-{sub}_T1w_RAS_N4.nii.gz"
    if not n4p.exists():
        n4 = ants.n4_bias_field_correction(ants.image_read(str(ras)), shrink_factor=4,
            convergence={"iters": [50, 50, 30, 20], "tol": 1e-7}, rescale_intensities=False)
        ants.image_write(n4, str(n4p))
    synth_brain = sd / f"sub-{sub}_T1w_RAS_SynthStrip.nii.gz"
    maskp = sd / f"sub-{sub}_SynthStrip_mask.nii.gz"
    synthstrip(ras, synth_brain, maskp, sd / "synthstrip_command.txt")
    # Enforce the original binary mask exactly as emitted: only validate, never morphologically edit.
    raw_mask = np.asarray(nib.load(maskp).dataobj)
    if not np.all(np.isin(np.unique(raw_mask), [0, 1])):
        raise RuntimeError(f"SynthStrip mask is not binary for {sub}")
    n4 = ants.image_read(str(n4p)); mask = ants.image_read(str(maskp))
    brain = n4 * mask
    brainp = sd / f"sub-{sub}_T1w_RAS_N4_SynthStrip_brain.nii.gz"
    ants.image_write(brain, str(brainp))

    reg = ants.registration(template["brain"], brain, type_of_transform="SyN",
        outprefix=str(tr / "mri_to_mni_"), reg_iterations=(40, 20, 0),
        random_seed=20260906, singleprecision=True, verbose=False)
    mri_fwd = copy_transforms(reg["fwdtransforms"], tr, "mri_to_mni_fwd")
    mri_inv = copy_transforms(reg["invtransforms"], tr, "mri_to_mni_inv")
    mri_mni = sd / f"sub-{sub}_T1w_MNI2mm.nii.gz"
    ants.image_write(reg["warpedmovout"], str(mri_mni))
    mask_mni_img = ants.apply_transforms(template["mask"], mask, reg["fwdtransforms"], interpolator="nearestNeighbor")
    mask_mni = sd / f"sub-{sub}_SynthStrip_mask_MNI2mm.nii.gz"
    ants.image_write(mask_mni_img, str(mask_mni))
    warp = next((x for x in reg["fwdtransforms"] if str(x).endswith(".nii.gz")), None)
    jac_min = jac_max = jac_bad = np.nan
    if warp:
        jac = ants.create_jacobian_determinant_image(template["head"], warp, do_log=False, geom=False)
        ja = jac.numpy(); jac_min, jac_max, jac_bad = float(ja.min()), float(ja.max()), float(np.mean(ja <= 0))
        ants.image_write(jac, str(sd / f"sub-{sub}_mri_to_mni_jacobian.nii.gz"))

    # Recompute PET motion from native for R1; no old motion output is reused.
    pet = ants.image_read(str(native_pet)); motion_fd=[]; motion_det=[]; motion_trans=[]; motion_rot=[]
    if pet.dimension == 4:
        frame0 = ants.slice_image(pet, axis=3, idx=0)
        pet_before = sd / f"sub-{sub}_PET_frame0_before_motion.nii.gz"; ants.image_write(frame0, str(pet_before))
        mc = ants.motion_correction(pet, type_of_transform="Rigid", fdOffset=50,
            outprefix=str(tr / "pet_motion_"), random_seed=20260906, verbose=False)
        pet4 = mc["motion_corrected"]
        pet4p = sd / f"sub-{sub}_PET_motion_corrected_4D.nii.gz"; ants.image_write(pet4, str(pet4p))
        motion_fd = np.asarray(mc["FD"], float).tolist()
        for item in mc["motion_parameters"]:
            if item == "NA" or item is None:
                motion_det.append(1.0); motion_trans.append([0.,0.,0.]); motion_rot.append([0.,0.,0.])
            else:
                f = item[0] if isinstance(item, list) else item
                det, trans, rot = affine_stats(f)
                motion_det.append(det); motion_trans.append(trans); motion_rot.append(rot)
        petmean = ants.get_average_of_timeseries(pet4)
    else:
        petmean = pet
        pet_before = sd / f"sub-{sub}_PET_3D_BRAIN2_native.nii.gz"; ants.image_write(pet, str(pet_before))
    petmeanp = sd / f"sub-{sub}_PET_mean_after_motion.nii.gz"; ants.image_write(petmean, str(petmeanp))

    before_grid = ants.resample_image_to_target(petmean, brain, interp_type="linear")
    before_nmi = nmi(brain.numpy(), before_grid.numpy())
    before_mask = ants.get_mask(before_grid).numpy()
    before_dice = dice(mask.numpy(), before_mask)
    preg = ants.registration(brain, petmean, type_of_transform="Rigid",
        outprefix=str(tr / "pet_to_mri_"), random_seed=20260906, verbose=False)
    pet_fwd = copy_transforms(preg["fwdtransforms"], tr, "pet_to_mri_fwd")
    pet_inv = copy_transforms(preg["invtransforms"], tr, "pet_to_mri_inv")
    pet_mri = sd / f"sub-{sub}_PET_in_T1w.nii.gz"; ants.image_write(preg["warpedmovout"], str(pet_mri))
    after_nmi = nmi(brain.numpy(), preg["warpedmovout"].numpy())
    after_mask = ants.get_mask(preg["warpedmovout"]).numpy()
    after_dice = dice(mask.numpy(), after_mask)
    pet_det, pet_trans, pet_rot = affine_stats(preg["fwdtransforms"][-1])

    chain = reg["fwdtransforms"] + preg["fwdtransforms"]
    pet_mni_img = ants.apply_transforms(template["head"], petmean, chain,
        interpolator="linear", imagetype=0, singleprecision=True)
    pet_mni = sd / f"sub-{sub}_PET_MNI2mm.nii.gz"; ants.image_write(pet_mni_img, str(pet_mni))
    (tr / "pet_to_mni_chain.json").write_text(json.dumps({
        "ANTs_transformlist_application_order": [str(x) for x in chain],
        "interpolator": "linear", "number_of_resamplings": 1,
    }, indent=2) + "\n")

    mq = mask_qc(maskp, ras)
    mni_overlap = dice(template["mask"].numpy(), mask_mni_img.numpy())
    mni_coverage = float(np.logical_and(template["mask"].numpy()>0, mask_mni_img.numpy()>0).sum() / max((template["mask"].numpy()>0).sum(), 1))
    final = image_qc(pet_mni)
    reasons=[]; warnings=[]
    if not mq["mask_finite"] or not mq["mask_affine_finite"] or abs(mq["mask_affine_determinant"]) < 1e-8: reasons.append("invalid mask/affine")
    if mq["brain_volume_ml"] < 700 or mq["brain_volume_ml"] > 1900: warnings.append("brain mask volume outside broad adult plausibility range")
    if mq["largest_component_fraction"] < .995 or any(json.loads(mq["slice_gap_counts_xyz"]).values()): warnings.append("mask topology/continuity requires visual review")
    if not final["finite"] or final["nonzero_fraction"] == 0: reasons.append("invalid or empty final PET")
    if not np.isfinite(jac_bad) or jac_bad > 0 or jac_min <= 0: reasons.append("non-positive/invalid warp Jacobian")
    if pet_det <= 0 or any(x <= 0 for x in motion_det): reasons.append("invalid rigid transform determinant")
    if after_nmi < before_nmi: reasons.append("PET-to-MRI NMI worsened")
    if after_dice < before_dice: reasons.append("PET-to-MRI Dice worsened")
    if mni_overlap < .75: warnings.append("SynthStrip-to-MNI mask overlap below 0.75")
    if sub == "153_S_4172": warnings.append("protocol exception ADNI2/PET-DICOM/LAS; lateralidad visual obligatoria")
    quantitative = "FAIL" if reasons else ("WARNING" if warnings else "PASS")
    row = {
        "subject_id": sub, "group": group, "selection_reason": selection,
        "status": quantitative, "quantitative_status": quantitative,
        "visual_status": "PENDING", "reasons": "; ".join(reasons + warnings),
        "mri_native_sha256": sha256(native_mri), "pet_native_sha256": sha256(native_pet),
        "synthstrip_mask_sha256": sha256(maskp), "synthstrip_stripped_sha256": sha256(synth_brain),
        **mq, "mni_mask_dice": mni_overlap, "mni_mask_coverage": mni_coverage,
        "motion_framewise_displacement_mm": json.dumps(motion_fd),
        "motion_translation_mm": json.dumps(motion_trans), "motion_rotation_deg": json.dumps(motion_rot),
        "motion_determinants": json.dumps(motion_det), "pet_rigid_translation_mm": json.dumps(pet_trans),
        "pet_rigid_rotation_deg": json.dumps(pet_rot), "pet_rigid_determinant": pet_det,
        "pet_mri_nmi_before": before_nmi, "pet_mri_nmi_after": after_nmi,
        "pet_mri_dice_before": before_dice, "pet_mri_dice_after": after_dice,
        "mri_mni_nmi": nmi(template["brain"].numpy(), reg["warpedmovout"].numpy()),
        "warp_jacobian_min": jac_min, "warp_jacobian_max": jac_max,
        "warp_nonpositive_fraction": jac_bad, **final,
        "mri_fwd_transforms": json.dumps(mri_fwd), "mri_inverse_transforms": json.dumps(mri_inv),
        "pet_fwd_transform": json.dumps(pet_fwd), "pet_inverse_transform": json.dumps(pet_inv),
        "pet_to_mni_single_resampling": True, "old_dependent_outputs_reused": False,
    }
    old_m = nib.load(oldmask).get_fdata() > .5; new_m = nib.load(maskp).get_fdata() > .5
    comp = {"subject_id": sub, "previous_mask_path": str(oldmask), "previous_mask_sha256": sha256(oldmask),
        "synthstrip_mask_path": str(maskp), "synthstrip_mask_sha256": sha256(maskp),
        "previous_volume_ml": float(old_m.sum()*abs(np.linalg.det(nib.load(oldmask).affine[:3,:3]))/1000),
        "synthstrip_volume_ml": mq["brain_volume_ml"], "dice_previous_vs_synthstrip": dice(old_m,new_m),
        "previous_only_voxels": int(np.logical_and(old_m,~new_m).sum()),
        "synthstrip_only_voxels": int(np.logical_and(new_m,~old_m).sum()),
        "previous_validation": "failed_artifact", "synthstrip_visual_status": "PENDING"}
    figures(sub, {"ras":ras,"n4":n4p,"mask":maskp,"oldmask":oldmask,"brain":brainp,
        "mri_mni":mri_mni,"pet_mni":pet_mni,"pet_before":pet_before,"pet_mri":pet_mri})
    return row, comp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    for p in (TPL_HEAD, TPL_BRAIN, TPL_MASK, CONFIG):
        if not p.exists(): raise FileNotFoundError(p)
    inspect = subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True)
    if IMAGE_DIGEST.split(":",1)[1] not in inspect:
        raise RuntimeError("Local SynthStrip image does not match pinned digest")
    # Generate all masks before ANTs registrations to avoid retained-memory pressure.
    for pre_sub, _, _ in SUBJECTS:
        pre_sd = OUT / f"sub-{pre_sub}"
        pre_sd.mkdir(parents=True, exist_ok=True)
        pre_ras = pre_sd / f"sub-{pre_sub}_T1w_RAS.nii.gz"
        if not pre_ras.exists():
            write_nifti(nib.as_closest_canonical(nib.load(NATIVE / f"sub-{pre_sub}_T1w.nii.gz")), pre_ras)
        synthstrip(pre_ras, pre_sd / f"sub-{pre_sub}_T1w_RAS_SynthStrip.nii.gz", pre_sd / f"sub-{pre_sub}_SynthStrip_mask.nii.gz", pre_sd / "synthstrip_command.txt")
    template = {"head":ants.image_read(str(TPL_HEAD)), "brain":ants.image_read(str(TPL_BRAIN)), "mask":ants.image_read(str(TPL_MASK))}
    rows=[]; comparisons=[]
    for sub, group, selection in SUBJECTS:
        print(f"START {sub}", flush=True)
        row, comp = process_subject(sub, group, selection, template)
        rows.append(row); comparisons.append(comp)
        print(f"DONE {sub} {row['quantitative_status']}", flush=True)
    with QC.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with COMPARE.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    provenance = {
        "stage":"C1-R1", "created":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthstrip":{"origin":"official Docker Hub freesurfer organization","image":IMAGE,
            "image_digest":IMAGE_DIGEST,"model":"/freesurfer/models/synthstrip.1.pt","model_sha256":MODEL_SHA,
            "container_python":"3.10.18","torch":"2.1.2+cpu","surfa":"0.6.1","nibabel":"5.3.2"},
        "host":{"python":sys.version,"antspyx":ants.__version__,"nibabel":nib.__version__,"numpy":np.__version__},
        "config":str(CONFIG),"config_sha256":sha256(CONFIG),"native_and_motion_reuse":"native only; motion recomputed",
    }
    (BASE/"tool_versions_and_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    files=[]
    for p in sorted(BASE.rglob("*")):
        if p.is_file() and p != HASHES: files.append({"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"bytes":p.stat().st_size})
    for p in (QC, COMPARE, CONFIG, Path(__file__)):
        if p.is_file(): files.append({"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"bytes":p.stat().st_size})
    with HASHES.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","sha256","bytes"]);w.writeheader();w.writerows(files)
    print(json.dumps({"subjects":len(rows),"quantitative_counts":Counter(r["quantitative_status"] for r in rows),
        "visual_status":"PENDING_MANUAL_REVIEW","stage_c2_started":False},default=dict,indent=2))


if __name__ == "__main__":
    main()
