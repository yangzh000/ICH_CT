from dataclasses import replace

import numpy as np
import SimpleITK as sitk

from hemorrhage.common import FitAudit, Patient
from hemorrhage.features import FeatureEngine
from hemorrhage.habitat import HabitatBuilder, HabitatClassifier
from hemorrhage.imaging import CaseStore


def test_actual_oof_habitat_fit_and_frozen_prediction(tmp_path, config, matlab):
    config["augmentation"]["copies"] = 0
    config["preprocessing"]["normalization"] = "none"
    config["radiomics"]["feature_classes"] = {"firstorder": ["Mean", "Variance"], "glcm": ["Contrast"]}
    config["svm"]["max_features"] = 1
    config["habitat"]["final_feature_family"] = "texture"
    rng = np.random.default_rng(16)
    z, _, _ = np.indices((32, 32, 32))
    patients = []
    for index in range(18):
        label = index % 2
        array = (z / 32 + label * 0.25 + rng.normal(0, 0.01, z.shape)).astype(np.float32)
        mask = np.zeros(z.shape, dtype=np.uint8)
        mask[2:30, 15:17, 15:17] = 1
        image_path, mask_path = tmp_path / f"image_{index}.nii.gz", tmp_path / f"mask_{index}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(image_path))
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))
        patients.append(Patient(str(index), "A" if index < 16 else "B", str(image_path), str(mask_path), label))
    store = CaseStore(tmp_path / "cache", config)
    engine = FeatureEngine(store, config)
    audit = FitAudit(tmp_path / "fit_audit.jsonl")
    builder = HabitatBuilder(engine, config, audit)
    fitted = HabitatClassifier(engine, config, 31, depth=1).fit(patients[:16], builder)
    assert fitted.depth == 1
    assert fitted.classifier.selected_names[0].startswith("original_glcm_")
    evaluation = patients[16:]
    masks = fitted.transform(evaluation, tmp_path / "maps")
    for patient in evaluation:
        assert 0 < masks[patient.patient_id].sum() < store.get(patient).mask.sum()
    probabilities = fitted.predict(evaluation)
    unlabeled = [replace(patient, label=None) for patient in evaluation]
    altered = [replace(patient, label=1 - patient.label) for patient in evaluation]
    np.testing.assert_array_equal(probabilities, fitted.predict(unlabeled))
    np.testing.assert_array_equal(probabilities, fitted.predict(altered))
    assert np.isfinite(probabilities).all()
    external_ids = {p.patient_id for p in evaluation}
    assert all(not external_ids & set(event["training_patients"]) for event in audit.events)
    assert len(list((tmp_path / "maps").rglob("habitat_01.nii.gz"))) == 2
