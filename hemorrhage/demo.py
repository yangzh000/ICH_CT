import copy
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .common import write_json


def demo_configuration(config):
    result = copy.deepcopy(config)
    result["cohort"]["expected_development"] = 64
    result["cohort"]["expected_external"] = 12
    result["augmentation"]["copies"] = 1
    result["augmentation"]["rotation_degrees"] = 3.0
    result["augmentation"]["scale_range"] = [0.98, 1.02]
    result["radiomics"]["feature_classes"] = {"firstorder": ["Mean", "Variance"], "glcm": ["Contrast"], "shape": ["MeshVolume"]}
    result["svm"]["max_features"] = 2
    for key in ("outer_folds", "selection_folds", "calibration_folds", "crossfit_folds", "depth_folds", "threshold_folds"):
        result["cv"][key] = 2
    result["cv"]["outer_repeats"] = 1
    result["habitat"]["max_depth"] = 1
    result["habitat"]["final_feature_family"] = "all"
    result["deep"]["input_shape"] = [32, 32, 32]
    result["deep"]["epochs"] = 1
    result["deep"]["batch_size"] = 8
    result["deep"]["resnet_width_factor"] = 0.125
    result["deep"]["swin_embed_dim"] = 12
    result["deep"]["swin_window"] = [2, 2, 2]
    result["deep"]["swin_depths"] = [1, 1, 1, 1]
    result["deep"]["use_checkpoint"] = False
    result["deep"]["device"] = "cpu"
    result["evaluation"]["threshold_rule"] = "fixed"
    result["evaluation"]["shap_background"] = 8
    result["evaluation"]["shap_samples"] = 32
    result["execution"]["enforce_confirmed_protocol"] = False
    result["execution"]["torch_threads"] = 2
    return result


def generate_demo(output, config):
    output = Path(output).resolve()
    if (output / "patients.csv").exists():
        raise FileExistsError("Demo dataset already exists")
    directory = output / "images"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    rows = []
    for index in range(76):
        center = "A" if index < 64 else ("B" if index < 70 else "C")
        label = index % 2
        patient_id = f"synthetic_{index:03d}"
        array = rng.normal(35, 5, (32, 32, 32)).astype(np.float32)
        mask = np.zeros(array.shape, dtype=np.uint8)
        origin = rng.integers(12, 16, size=3)
        slices = tuple(slice(int(x), int(x + 4)) for x in origin)
        mask[slices] = 1
        array[slices] += 15 + label * 20 + rng.normal(0, 2, (4, 4, 4))
        image_path, mask_path = directory / f"{patient_id}_image.nii.gz", directory / f"{patient_id}_mask.nii.gz"
        image, segmentation = sitk.GetImageFromArray(array), sitk.GetImageFromArray(mask)
        sitk.WriteImage(image, str(image_path), True)
        sitk.WriteImage(segmentation, str(mask_path), True)
        rows.append({"patient_id": patient_id, "center": center, "image": str(image_path.relative_to(output)), "mask": str(mask_path.relative_to(output)), "label": label})
    pd.DataFrame(rows).to_csv(output / "patients.csv", index=False)
    write_json(output / "config.demo.json", demo_configuration(config))
    write_json(output / "dataset_status.json", {"synthetic": True, "purpose": "software integration testing", "clinical_results": False})
    return output
