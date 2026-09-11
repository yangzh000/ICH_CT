import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    required = {"seed", "cohort", "preprocessing", "augmentation", "radiomics", "svm", "cv", "habitat", "deep", "evaluation", "execution", "matlab"}
    if not required.issubset(config):
        raise ValueError(f"Missing configuration sections: {sorted(required - set(config))}")
    if set(config["execution"]["models"]) - {"radiomics", "habitat", "resnet34", "swin"}:
        raise ValueError("Unknown model name")
    if len(config["execution"]["models"]) != len(set(config["execution"]["models"])):
        raise ValueError("Model names must not be repeated")
    cv = config["cv"]
    for name in ("outer_folds", "selection_folds", "calibration_folds", "crossfit_folds", "depth_folds", "threshold_folds"):
        if not isinstance(cv[name], int) or cv[name] < 2:
            raise ValueError(f"{name} must be at least 2")
    if cv["outer_repeats"] < 1 or cv["selection_repeats"] < 1:
        raise ValueError("Cross-validation repetition counts must be positive")
    if config["preprocessing"]["normalization"] not in {"zscore_image", "zscore_roi", "none"}:
        raise ValueError("Unknown normalization")
    if config["preprocessing"]["interpolator"] not in {"linear", "bspline"}:
        raise ValueError("Unknown image interpolator")
    if config["radiomics"]["bin_width"] is None and config["radiomics"]["bin_count"] is None:
        raise ValueError("Set either bin_width or bin_count")
    if config["radiomics"]["bin_width"] is not None and config["radiomics"]["bin_count"] is not None:
        raise ValueError("Set only one discretization parameter")
    if config["svm"]["C"] <= 0:
        raise ValueError("SVM C must be positive")
    if config["svm"]["max_features"] is not None and config["svm"]["max_features"] < 1:
        raise ValueError("max_features must be positive or null")
    h = config["habitat"]
    if h["patch_size"] < 2 or h["stride"] < 1 or h["clusters"] < 2 or h["max_depth"] < 1:
        raise ValueError("Invalid habitat parameters")
    if h["patch_batch_size"] < 1 or h["kmeans_n_init"] < 1:
        raise ValueError("Invalid habitat batch or K-means configuration")
    if config["deep"]["spatial_dims"] != 3:
        raise ValueError("This implementation uses the confirmed 3D network inputs")
    if len(config["deep"]["input_shape"]) != 3 or min(config["deep"]["input_shape"]) < 32:
        raise ValueError("3D network input dimensions must each be at least 32")
    if config["deep"]["epochs"] < 1 or config["deep"]["batch_size"] < 2:
        raise ValueError("Positive epochs and batch_size >= 2 are required")
    if config["deep"]["optimizer"] not in {"adamw", "sgd"}:
        raise ValueError("Unknown optimizer")
    if config["deep"]["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Unknown device")
    if config["evaluation"]["threshold_rule"] not in {"youden", "fixed"}:
        raise ValueError("Unknown threshold rule")
    if not 0 <= config["evaluation"]["fixed_threshold"] <= 1:
        raise ValueError("fixed_threshold must lie in [0, 1]")
    if config["cohort"]["development_center"] in config["cohort"]["external_centers"]:
        raise ValueError("Development and external centers must be distinct")
    if config["execution"].get("enforce_confirmed_protocol", True):
        checks = [(cv["outer_folds"], 5), (cv["outer_repeats"], 5), (h["patch_size"], 20), (h["stride"], 1), (h["clusters"], 10), (config["preprocessing"]["spacing"], [1.0, 1.0, 1.0])]
        if any(actual != expected for actual, expected in checks):
            raise ValueError("The confirmed protocol requires 5 folds repeated 5 times, 20-voxel patches, stride 1, 10 clusters, and 1-mm isotropic spacing")
    return config
