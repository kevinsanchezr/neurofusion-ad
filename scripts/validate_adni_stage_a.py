#!/usr/bin/env python3
"""Validate representative ADNI MRI DICOM, ECAT7 PET, and ADNI2 PET DICOM.

This script operates only on the representative files already extracted beneath
data/derived/adni/stage_a_validation/source. It does not perform batch conversion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import ecat
import numpy as np
import pydicom


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/derived/adni/stage_a_validation"
SOURCE = BASE / "source"
NIFTI = BASE / "nifti"
QC = BASE / "qc"
FIGURES = ROOT / "reports/figures/adni_stage_a_validation"
DCM2NIIX = ROOT / ".tools/dcm2niix/usr/bin/dcm2niix"
DCM2NIIX_LIB = ROOT / ".tools/dcm2niix/usr/lib/x86_64-linux-gnu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(ds: pydicom.Dataset, name: str) -> str:
    value = getattr(ds, name, "")
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in value)
    return str(value)


def convert_dicom(source: Path, filename: str) -> Path:
    output = NIFTI / filename
    stem = filename.removesuffix(".nii.gz")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(DCM2NIIX_LIB)
    command = [
        str(DCM2NIIX), "-z", "y", "-b", "y", "-w", "1", "-f", stem,
        "-o", str(NIFTI), str(source),
    ]
    subprocess.run(command, check=True, env=env)
    if not output.exists():
        raise RuntimeError(f"Expected dcm2niix output missing: {output}")
    return output


def convert_ecat(source: Path, output: Path) -> dict:
    image = ecat.load(str(source))
    nframes = image._subheader.get_nframes()
    frames = []
    affines = []
    zooms = []
    timing = []
    for frame_index in range(nframes):
        frame = np.asarray(image.get_frame(frame_index), dtype=np.float32)
        frame_affine = np.asarray(image.get_frame_affine(frame_index), dtype=float)
        frame_zooms = tuple(float(x) for x in image._subheader.get_zooms(frame_index)[:3])
        subheader = image._subheader.subheaders[frame_index]
        frames.append(frame)
        affines.append(frame_affine)
        zooms.append(frame_zooms)
        timing.append(
            {
                "frame": frame_index,
                "start_ms": int(subheader["frame_start_time"]),
                "duration_ms": int(subheader["frame_duration"]),
                "shape": list(frame.shape),
                "zooms_mm": list(frame_zooms),
                "affine": frame_affine.tolist(),
                "min": float(np.min(frame)),
                "max": float(np.max(frame)),
                "mean": float(np.mean(frame)),
                "std": float(np.std(frame)),
                "finite": bool(np.isfinite(frame).all()),
                "nonzero_fraction": float(np.count_nonzero(frame) / frame.size),
            }
        )
    if not all(frame.shape == frames[0].shape for frame in frames):
        raise RuntimeError(f"Inconsistent ECAT frame shapes: {source}")
    if not all(np.allclose(affine, affines[0], atol=1e-5) for affine in affines):
        raise RuntimeError(f"Inconsistent ECAT frame affines: {source}")
    if not all(np.allclose(zoom, zooms[0], atol=1e-6) for zoom in zooms):
        raise RuntimeError(f"Inconsistent ECAT frame spacings: {source}")
    if not all(item["finite"] and item["nonzero_fraction"] > 0 for item in timing):
        raise RuntimeError(f"Invalid ECAT frame data: {source}")
    durations = {item["duration_ms"] for item in timing}
    if len(durations) != 1:
        raise RuntimeError(f"Inconsistent ECAT frame durations: {source}")
    data = np.stack(frames, axis=3)
    nifti = nib.Nifti1Image(data, affines[0])
    nifti.header.set_xyzt_units("mm", "sec")
    nifti.header.set_zooms(zooms[0] + (next(iter(durations)) / 1000.0,))
    nifti.set_qform(affines[0], code=1)
    nifti.set_sform(affines[0], code=1)
    nib.save(nifti, output)
    return {"source": str(source.relative_to(ROOT)), "frames": timing}


def nifti_qc(label: str, path: Path, expected_frames: int | None = None) -> dict:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    finite = bool(np.isfinite(data).all())
    nonzero_fraction = float(np.count_nonzero(data) / data.size)
    passed = (
        image.ndim in (3, 4)
        and all(size > 1 for size in image.shape[:3])
        and np.isfinite(image.affine).all()
        and finite
        and nonzero_fraction > 0
        and all(np.isfinite(z) and z > 0 for z in image.header.get_zooms())
        and (expected_frames is None or (image.ndim == 4 and image.shape[3] == expected_frames))
    )
    return {
        "test": label,
        "nifti_path": str(path.relative_to(ROOT)),
        "status": "PASS" if passed else "FAIL",
        "ndim": image.ndim,
        "shape": "x".join(str(x) for x in image.shape),
        "voxel_spacing": "x".join(f"{x:.8g}" for x in image.header.get_zooms()),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "affine": json.dumps(image.affine.tolist(), separators=(",", ":")),
        "finite": finite,
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "nonzero_fraction": nonzero_fraction,
        "sha256": sha256(path),
    }


def contact_sheet(label: str, path: Path) -> None:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4:
        display = np.mean(data, axis=3)  # display only; saved NIfTI remains 4D
        subtitle = f"4D ({data.shape[3]} frames); temporal mean shown for QC only"
    else:
        display = data
        subtitle = "native 3D volume"
    finite = display[np.isfinite(display)]
    positive = finite[finite > 0]
    low, high = (np.percentile(positive, [1, 99]) if positive.size else (0, 1))
    x, y, z = (size // 2 for size in display.shape[:3])
    views = [
        (np.rot90(display[:, :, z]), "axial"),
        (np.rot90(display[:, y, :]), "coronal"),
        (np.rot90(display[x, :, :]), "sagittal"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="black")
    for axis, (view, title) in zip(axes, views):
        axis.imshow(view, cmap="gray", vmin=low, vmax=high, origin="lower")
        axis.set_title(title, color="white")
        axis.axis("off")
    fig.suptitle(f"{label} — {subtitle}", color="white")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{label}_orthogonal.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    NIFTI.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Hash raw archives and every extracted representative source file.
    hash_rows = []
    for path in sorted((ROOT / "data/raw/adni").glob("*.zip")):
        hash_rows.append({"kind": "raw_archive", "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            hash_rows.append({"kind": "representative_source", "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    with (QC / "sha256.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kind", "path", "sha256"])
        writer.writeheader(); writer.writerows(hash_rows)

    # ECAT conversion: retain all native frames as 4D; no temporal averaging.
    ecat_specs = [
        ("011_S_0010_I14808", "sub-011_S_0010_I14808_FDGpet_4D.nii.gz"),
        ("005_S_0221_I103955", "sub-005_S_0221_I103955_FDGpet_4D.nii.gz"),
    ]
    ecat_details = []
    outputs = []
    for label, filename in ecat_specs:
        candidates = list((SOURCE / label).rglob("*.v"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ECAT source for {label}: {candidates}")
        output = NIFTI / filename
        ecat_details.append({"test": label, **convert_ecat(candidates[0], output)})
        outputs.append((label, output, 6))
    (QC / "ecat_frame_metadata.json").write_text(json.dumps(ecat_details, indent=2) + "\n")

    # Group all PET DICOM headers and retain full per-instance evidence.
    pet_root = SOURCE / "153_S_4172_I256061"
    dicom_rows = []
    groups: dict[tuple[str, ...], list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    group_fields = [
        "SeriesInstanceUID", "SeriesDescription", "ProtocolName", "SeriesNumber",
        "AcquisitionNumber", "ImageType", "FrameReferenceTime", "ActualFrameDuration",
    ]
    for path in sorted(pet_root.rglob("*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        key = tuple(scalar(ds, field) for field in group_fields)
        groups[key].append((path, ds))
        dicom_rows.append(
            {
                **{field: scalar(ds, field) for field in group_fields},
                "InstanceNumber": scalar(ds, "InstanceNumber"),
                "Rows": scalar(ds, "Rows"), "Columns": scalar(ds, "Columns"),
                "PixelSpacing": scalar(ds, "PixelSpacing"),
                "SliceThickness": scalar(ds, "SliceThickness"),
                "ImageOrientationPatient": scalar(ds, "ImageOrientationPatient"),
                "ImagePositionPatient": scalar(ds, "ImagePositionPatient"),
                "AcquisitionTime": scalar(ds, "AcquisitionTime"),
                "SeriesTime": scalar(ds, "SeriesTime"),
                "SOPInstanceUID": scalar(ds, "SOPInstanceUID"),
                "source_path": str(path.relative_to(ROOT)),
            }
        )
    instance_fields = list(dicom_rows[0])
    with (QC / "I256061_dicom_instances.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=instance_fields)
        writer.writeheader(); writer.writerows(dicom_rows)

    grouped_rows = []
    selected_dir = QC / "I256061_selected_FDG_PET_BRAIN_2"
    excluded_dir = QC / "I256061_excluded_FDG_PET_BRAIN_1"
    selected_dir.mkdir(exist_ok=True); excluded_dir.mkdir(exist_ok=True)
    for key, items in groups.items():
        first = items[0][1]
        positions = [tuple(float(x) for x in ds.ImagePositionPatient) for _, ds in items]
        orientations = {tuple(float(x) for x in ds.ImageOrientationPatient) for _, ds in items}
        instances = [int(ds.InstanceNumber) for _, ds in items]
        description = scalar(first, "SeriesDescription")
        selected = description == "FDG PET BRAIN 2" and scalar(first, "SeriesNumber") == "4"
        target_dir = selected_dir if selected else excluded_dir
        for path, _ in items:
            link = target_dir / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
        grouped_rows.append(
            {
                **dict(zip(group_fields, key)), "file_count": len(items),
                "instance_min": min(instances), "instance_max": max(instances),
                "unique_instances": len(set(instances)),
                "Rows": scalar(first, "Rows"), "Columns": scalar(first, "Columns"),
                "PixelSpacing": scalar(first, "PixelSpacing"),
                "SliceThickness": scalar(first, "SliceThickness"),
                "orientation_count": len(orientations),
                "ImageOrientationPatient": scalar(first, "ImageOrientationPatient"),
                "unique_positions": len(set(positions)),
                "position_x_range": f"{min(p[0] for p in positions)}..{max(p[0] for p in positions)}",
                "position_y_range": f"{min(p[1] for p in positions)}..{max(p[1] for p in positions)}",
                "position_z_range": f"{min(p[2] for p in positions)}..{max(p[2] for p in positions)}",
                "AcquisitionTime": scalar(first, "AcquisitionTime"),
                "SeriesTime": scalar(first, "SeriesTime"),
                "selection_disposition": "SELECTED_I256061_S122077" if selected else "EXCLUDED_NOT_XML_DESCRIPTION",
            }
        )
    with (QC / "I256061_dicom_groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grouped_rows[0]))
        writer.writeheader(); writer.writerows(grouped_rows)

    # Convert the selected PET group and the already validated MRI reference.
    pet_output = convert_dicom(selected_dir, "sub-153_S_4172_I256061_FDGpet.nii.gz")
    mri_candidates = list((SOURCE / "011_S_0010_I14868").rglob("*.dcm"))
    if len(mri_candidates) != 160:
        raise RuntimeError(f"Expected 160 MRI DICOM files, found {len(mri_candidates)}")
    mri_output = convert_dicom(SOURCE / "011_S_0010_I14868", "sub-011_S_0010_I14868_T1w.nii.gz")
    outputs.extend([
        ("153_S_4172_I256061", pet_output, None),
        ("011_S_0010_I14868", mri_output, None),
    ])

    qc_rows = [nifti_qc(label, path, frames) for label, path, frames in outputs]
    with (ROOT / "reports/adni_stage_a_validation_qc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qc_rows[0]))
        writer.writeheader(); writer.writerows(qc_rows)
    if any(row["status"] != "PASS" for row in qc_rows):
        raise RuntimeError("At least one representative NIfTI failed QC")
    for label, path, _ in outputs:
        contact_sheet(label, path)

    dcm_version = subprocess.run(
        [str(DCM2NIIX), "--version"],
        env={**os.environ, "LD_LIBRARY_PATH": str(DCM2NIIX_LIB)},
        check=False, text=True, capture_output=True,
    )
    versions = {
        "python": os.sys.version,
        "nibabel": nib.__version__, "numpy": np.__version__,
        "pydicom": pydicom.__version__, "matplotlib": matplotlib.__version__,
        "dcm2niix": "\n".join(
            part.strip() for part in (dcm_version.stdout, dcm_version.stderr) if part.strip()
        ),
    }
    (QC / "tool_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "qc": qc_rows, "versions": versions}, indent=2))


if __name__ == "__main__":
    main()
