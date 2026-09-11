import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import radiomics
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor

from .common import array_digest, digest
from .imaging import Volume, sitk_image


TEXTURE_CLASSES = {"glcm", "glrlm", "glszm", "gldm", "ngtdm"}


@dataclass
class FeatureBatch:
    values: np.ndarray
    names: list
    labels: np.ndarray
    groups: np.ndarray
    original: np.ndarray

    def subset(self, indices):
        return FeatureBatch(self.values[indices], self.names, self.labels[indices], self.groups[indices], self.original[indices])


class FeatureEngine:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.extractors = {}
        self.root = Path(store.root) / "features"
        self.root.mkdir(parents=True, exist_ok=True)
        radiomics.setVerbosity(logging.ERROR)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["extractors"] = {}
        return state

    def extractor(self, family="all", names=None):
        key = (family, tuple(names) if names else None)
        if key not in self.extractors:
            settings = self.config["radiomics"]
            options = {"normalize": False, "resampledPixelSpacing": None, "label": 1, "minimumROIDimensions": 3, "minimumROISize": 2, "correctMask": False, "additionalInfo": False, "enableCExtensions": True}
            if settings["bin_width"] is not None:
                options["binWidth"] = settings["bin_width"]
            else:
                options["binCount"] = settings["bin_count"]
            extractor = RadiomicsFeatureExtractor(**options)
            extractor.disableAllImageTypes()
            extractor.enableImageTypeByName("Original")
            extractor.disableAllFeatures()
            classes = settings["feature_classes"]
            if names is not None:
                classes = {}
                for name in names:
                    image_type, feature_class, feature = name.split("_", 2)
                    if image_type != "original":
                        raise ValueError(f"Unsupported image type in {name}")
                    classes.setdefault(feature_class, []).append(feature)
            for feature_class, selected in classes.items():
                if family == "texture" and feature_class not in TEXTURE_CLASSES:
                    continue
                extractor.enableFeaturesByName(**{feature_class: selected})
            self.extractors[key] = extractor
        return self.extractors[key]

    def extract(self, volume, family="all", names=None):
        image = np.pad(np.asarray(volume.image), 1, mode="edge")
        mask = np.pad(np.asarray(volume.mask, dtype=np.uint8), 1, mode="constant")
        origin = np.asarray(volume.origin) - np.asarray(volume.direction).reshape(3, 3) @ np.asarray(volume.spacing)
        padded = Volume(image, mask, volume.spacing, tuple(origin), volume.direction)
        result = self.extractor(family, names).execute(sitk_image(image, padded, sitk.sitkFloat32), sitk_image(mask, padded, sitk.sitkUInt8))
        keys = list(names) if names is not None else sorted(k for k in result if k.startswith("original_"))
        if not keys or any(k not in result for k in keys):
            raise ValueError("Feature extraction returned an empty or inconsistent feature set")
        values = np.asarray([float(result[k]) for k in keys])
        values[~np.isfinite(values)] = np.nan
        return keys, values

    def patient_features(self, patient, mask=None, index=0, family="all"):
        base = self.store.get(patient)
        active = np.asarray(base.mask if mask is None else mask)
        key = digest([self.store.prepare(patient), array_digest(active), index, family, self.config["radiomics"], self.config["augmentation"], self.config["seed"]])
        path = self.root / (key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                return saved["names"].tolist(), saved["values"]
        names, values = self.extract(self.store.variant(patient, active, index), family)
        np.savez_compressed(path, names=np.asarray(names), values=values)
        return names, values

    def matrix(self, patients, masks=None, training=False, family="all"):
        rows, labels, groups, original = [], [], [], []
        reference_names = None
        for patient in patients:
            mask = masks[patient.patient_id] if masks is not None else None
            count = self.config["augmentation"]["copies"] if training else 0
            for index in range(count + 1):
                names, values = self.patient_features(patient, mask, index, family)
                if reference_names is not None and names != reference_names:
                    raise ValueError("Radiomic feature names differ between patients")
                reference_names = names
                rows.append(values)
                labels.append(patient.label if patient.label is not None else -1)
                groups.append(patient.patient_id)
                original.append(index == 0)
        return FeatureBatch(np.vstack(rows), reference_names, np.asarray(labels), np.asarray(groups), np.asarray(original))
