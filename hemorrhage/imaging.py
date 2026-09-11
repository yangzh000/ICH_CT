import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

from .common import array_digest, digest, file_digest, write_json


@dataclass
class Volume:
    image: np.ndarray
    mask: np.ndarray
    spacing: tuple
    origin: tuple
    direction: tuple


def sitk_image(array, volume, pixel_id=None):
    result = sitk.GetImageFromArray(np.asarray(array))
    result.SetSpacing(volume.spacing)
    result.SetOrigin(volume.origin)
    result.SetDirection(volume.direction)
    return sitk.Cast(result, pixel_id) if pixel_id is not None else result


def bounding_box(mask, margin=0):
    points = np.argwhere(mask)
    if len(points) < 2:
        raise ValueError("ROI must contain at least two voxels")
    low = np.maximum(points.min(axis=0) - margin, 0)
    high = np.minimum(points.max(axis=0) + 1 + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def crop_volume(volume, slices):
    offset = np.array([s.start for s in slices][::-1]) * np.asarray(volume.spacing)
    origin = np.asarray(volume.origin) + np.asarray(volume.direction).reshape(3, 3) @ offset
    return Volume(volume.image[slices], volume.mask[slices], volume.spacing, tuple(origin), volume.direction)


def augment_volume(volume, mask, settings, seed):
    rng = np.random.default_rng(seed)
    angle = np.deg2rad(rng.uniform(-settings["rotation_degrees"], settings["rotation_degrees"], size=3))
    scale = rng.uniform(*settings["scale_range"])
    inverse = Rotation.from_euler("xyz", angle).as_matrix() / scale
    center = np.argwhere(mask).mean(axis=0)
    offset = center - inverse @ center
    image = affine_transform(np.asarray(volume.image), inverse, offset=offset, order=1, mode="nearest", prefilter=False)
    transformed_mask = affine_transform(mask.astype(np.uint8), inverse, offset=offset, order=0, mode="constant", cval=0, prefilter=False) > 0
    augmented = Volume(image, transformed_mask, volume.spacing, volume.origin, volume.direction)
    margin = int(rng.integers(settings["crop_margin_range"][0], settings["crop_margin_range"][1] + 1))
    return crop_volume(augmented, bounding_box(transformed_mask, margin))


class CaseStore:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = config
        self.prepared = {}
        self.hashes = {}
        self.sources = {}

    def prepare(self, patient):
        if patient.patient_id in self.prepared:
            if self.sources[patient.patient_id] != (patient.image, patient.mask):
                raise ValueError(f"Patient identifier reused with different source files: {patient.patient_id}")
            return self.prepared[patient.patient_id]
        hashes = (file_digest(patient.image), file_digest(patient.mask))
        key = digest([hashes, self.config["preprocessing"]])
        directory = self.root / "images" / key
        info_path = directory / "geometry.json"
        if not info_path.exists():
            image = sitk.ReadImage(patient.image, sitk.sitkFloat32)
            mask = sitk.ReadImage(patient.mask)
            if image.GetDimension() != 3 or mask.GetDimension() != 3:
                raise ValueError(f"Expected 3D images for {patient.patient_id}")
            if image.GetSize() != mask.GetSize() or any(not np.allclose(getattr(image, f)(), getattr(mask, f)(), atol=1e-5) for f in ("GetSpacing", "GetOrigin", "GetDirection")):
                raise ValueError(f"Image and ROI geometry differ for {patient.patient_id}")
            values = np.unique(sitk.GetArrayFromImage(mask))
            if not set(values).issubset({0, 1}) or 1 not in values:
                raise ValueError(f"A binary ROI mask containing label 1 is required for {patient.patient_id}")
            settings = self.config["preprocessing"]
            spacing = np.asarray(settings["spacing"], dtype=float)
            size = np.maximum(1, np.rint((np.asarray(image.GetSize()) - 1) * np.asarray(image.GetSpacing()) / spacing).astype(int) + 1)
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize([int(x) for x in size])
            resampler.SetOutputSpacing(tuple(spacing))
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear if settings["interpolator"] == "linear" else sitk.sitkBSpline)
            resampler.SetDefaultPixelValue(float(settings["outside_value"]))
            reconstructed = resampler.Execute(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            binary = sitk.GetArrayFromImage(resampler.Execute(sitk.Cast(mask, sitk.sitkUInt8))) > 0
            array = sitk.GetArrayFromImage(reconstructed).astype(np.float32)
            if not np.isfinite(array).all() or binary.sum() < 2:
                raise ValueError(f"Invalid image values or resampled ROI for {patient.patient_id}")
            if settings["clip_hu"] is not None:
                array = np.clip(array, *settings["clip_hu"])
            offset, scale = 0.0, 1.0
            if settings["normalization"] != "none":
                source = array[binary] if settings["normalization"] == "zscore_roi" else array
                offset, scale = float(source.mean()), float(source.std())
                if scale <= 1e-12:
                    raise ValueError(f"Image normalization has zero variance for {patient.patient_id}")
                array = (array - offset) / scale
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "image.npy", array.astype(np.float32))
            np.save(directory / "mask.npy", binary)
            write_json(info_path, {"spacing": reconstructed.GetSpacing(), "origin": reconstructed.GetOrigin(), "direction": reconstructed.GetDirection(), "source_spacing": image.GetSpacing(), "normalization_offset": offset, "normalization_scale": scale, "image_sha256": hashes[0], "mask_sha256": hashes[1]})
        self.prepared[patient.patient_id] = str(directory)
        self.hashes[patient.patient_id] = hashes
        self.sources[patient.patient_id] = (patient.image, patient.mask)
        return str(directory)

    def get(self, patient):
        directory = Path(self.prepare(patient))
        geometry = json.loads((directory / "geometry.json").read_text())
        return Volume(np.load(directory / "image.npy", mmap_mode="r"), np.load(directory / "mask.npy", mmap_mode="r"), tuple(geometry["spacing"]), tuple(geometry["origin"]), tuple(geometry["direction"]))

    def variant(self, patient, mask=None, index=0):
        volume = self.get(patient)
        active = np.asarray(volume.mask if mask is None else mask, dtype=bool)
        if index == 0:
            return Volume(volume.image, active, volume.spacing, volume.origin, volume.direction)
        seed = int(digest([self.config["seed"], patient.patient_id, index])[:8], 16)
        return augment_volume(volume, active, self.config["augmentation"], seed)

    def save_mask(self, patient, mask, path):
        volume = self.get(patient)
        sitk.WriteImage(sitk_image(mask.astype(np.uint8), volume), str(path), True)

    def save_map(self, patient, probabilities, path):
        volume = self.get(patient)
        sitk.WriteImage(sitk_image(probabilities.astype(np.float32), volume), str(path), True)

    def validate_unique_images(self, patients):
        seen = {}
        for patient in patients:
            self.prepare(patient)
            key = self.hashes[patient.patient_id][0]
            if key in seen:
                raise ValueError(f"Identical image content for patients {seen[key]} and {patient.patient_id}")
            seen[key] = patient.patient_id
