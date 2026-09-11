import gc
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import FitAudit, digest, file_digest, partition_patients, read_manifest, write_json
from .evaluation import binary_metrics, evaluate_external, select_threshold
from .features import FeatureEngine
from .habitat import HabitatBuilder, HabitatClassifier, HabitatDegeneracyError, RadiomicsClassifier
from .imaging import CaseStore


class ModelFactory:
    def __init__(self, store, config, audit):
        self.store, self.config, self.audit = store, config, audit
        self.engine = FeatureEngine(store, config)
        self.builder = HabitatBuilder(self.engine, config, audit)

    def fit(self, name, patients, seed):
        self.audit.record("model_fit", patients, model=name, seed=seed)
        if name == "radiomics":
            return RadiomicsClassifier(self.engine, self.config, seed).fit(patients)
        if name == "habitat":
            splits = partition_patients(patients, self.config["cv"]["depth_folds"], seed + 503)
            history, selected, previous = [], 0, -np.inf
            for depth in range(self.config["habitat"]["max_depth"] + 1):
                scores = []
                try:
                    for fold, (train, validation) in enumerate(splits):
                        self.audit.record("depth_selection", train, validation, depth=depth, fold=fold)
                        model = HabitatClassifier(self.engine, self.config, seed + fold, depth).fit(train, self.builder)
                        predictions = model.predict(validation)
                        scores.append(roc_auc_score([p.label for p in validation], predictions))
                except HabitatDegeneracyError as exception:
                    history.append({"depth": depth, "status": "invalid_cluster_map", "message": str(exception)})
                    break
                mean = float(np.mean(scores))
                history.append({"depth": depth, "mean_auc": mean, "fold_auc": scores, "status": "evaluated"})
                if depth > 0 and mean < previous:
                    break
                selected, previous = depth, mean
            model = HabitatClassifier(self.engine, self.config, seed, selected).fit(patients, self.builder)
            model.depth_history = history
            model.depth_cap_reached = selected == self.config["habitat"]["max_depth"]
            return model
        if name in {"resnet34", "swin"}:
            from .networks import DeepClassifier
            return DeepClassifier(name, self.store, self.config, seed).fit(patients)
        raise ValueError(name)

    def fit_with_threshold(self, name, patients, seed):
        calibration_rows = []
        if self.config["evaluation"]["threshold_rule"] == "fixed":
            threshold = self.config["evaluation"]["fixed_threshold"]
        else:
            for fold, (train, validation) in enumerate(partition_patients(patients, self.config["cv"]["threshold_folds"], seed + 601)):
                self.audit.record("threshold_selection", train, validation, model=name, fold=fold)
                candidate = self.fit(name, train, seed + 701 + fold)
                probabilities = candidate.predict(validation)
                calibration_rows.extend({"patient_id": patient.patient_id, "label": patient.label, "probability": probability} for patient, probability in zip(validation, probabilities))
                del candidate
                gc.collect()
            calibration = pd.DataFrame(calibration_rows)
            threshold = select_threshold(calibration.label, calibration.probability, self.config["evaluation"])
        result = self.fit(name, patients, seed)
        result.threshold = float(threshold)
        return result, calibration_rows


def prediction_rows(model_name, model, patients, **extra):
    probabilities = model.predict(patients)
    if len(probabilities) != len(patients):
        raise RuntimeError("Prediction count does not match patient count")
    return [{"patient_id": patient.patient_id, "center": patient.center, "label": patient.label, "model": model_name, "probability": float(probability), "threshold": model.threshold, **extra} for patient, probability in zip(patients, probabilities)]


def validate_cohorts(patients, config):
    settings = config["cohort"]
    allowed = {settings["development_center"], *settings["external_centers"]}
    if {p.center for p in patients} - allowed:
        raise ValueError("Manifest contains an unconfigured center")
    development = [p for p in patients if p.center == settings["development_center"]]
    external = [p for p in patients if p.center in settings["external_centers"]]
    for cohort, expected, name in [(development, settings["expected_development"], "development"), (external, settings["expected_external"], "external")]:
        if not cohort or set(p.label for p in cohort) != {0, 1}:
            raise ValueError(f"The {name} cohort must contain both classes")
        if expected is not None and len(cohort) != expected:
            raise ValueError(f"Expected {expected} {name} patients, found {len(cohort)}")
    return development, external


def environment_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["numpy", "scipy", "pandas", "scikit-learn", "SimpleITK", "pyradiomics", "torch", "monai", "einops", "shap"]:
        result[package] = importlib.metadata.version(package)
    return result


def summarize_internal(table, output):
    rows = []
    for (model, repetition), group in table.groupby(["model", "repetition"]):
        if group.patient_id.duplicated().any():
            raise ValueError("A patient has multiple held-out predictions within one repetition")
        metrics = binary_metrics(group.label, group.probability, group.threshold)
        rows.append({"model": model, "repetition": repetition, **{k: v for k, v in metrics.items() if "_ci_" not in k}})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(Path(output) / "internal_metrics_by_repetition.csv", index=False)
    summary = metrics.groupby("model")[["auc", "accuracy", "sensitivity", "specificity", "ppv", "npv"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(Path(output) / "internal_descriptive_summary.csv")
    table.groupby(["model", "patient_id", "center", "label"]).probability.agg(["mean", "std", "count"]).reset_index().to_csv(Path(output) / "internal_patient_mean_predictions.csv", index=False)


def run_study(manifest, config, output, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    patients = read_manifest(manifest)
    development, external = validate_cohorts(patients, config)
    store = CaseStore(output / "cache", config)
    store.validate_unique_images(patients)
    code_root = Path(__file__).resolve().parent
    code_hash = digest({str(path.relative_to(code_root)): file_digest(path) for path in sorted(code_root.glob("*.py"))})
    matlab_files = code_root.parent / "matlab"
    matlab_hash = digest({path.name: file_digest(path) for path in sorted(matlab_files.glob("*.m"))})
    versions = environment_versions()
    fingerprint = digest([config, [(p.patient_id, p.center, p.label, store.hashes[p.patient_id]) for p in patients], code_hash, matlab_hash, versions])
    state_path = output / "run_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("Run output exists; choose a new directory or use --resume")
        if json.loads(state_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Configuration, code, or data changed; use a new output directory")
    write_json(state_path, {"fingerprint": fingerprint, "status": "running", "development_n": len(development), "external_n": len(external)})
    write_json(output / "config.json", config)
    write_json(output / "environment.json", versions)
    audit = FitAudit(output / "fit_audit.jsonl")
    factory = ModelFactory(store, config, audit)
    assignments, internal = [], []
    for repetition in range(config["cv"]["outer_repeats"]):
        splits = partition_patients(development, config["cv"]["outer_folds"], config["seed"] + repetition)
        for fold, (train, validation) in enumerate(splits):
            assignments.extend({"patient_id": p.patient_id, "repetition": repetition + 1, "fold": fold + 1} for p in validation)
            for model_index, name in enumerate(config["execution"]["models"]):
                directory = output / "internal" / f"repeat_{repetition + 1:02d}" / f"fold_{fold + 1:02d}" / name
                path = directory / "predictions.csv"
                if resume and path.exists():
                    internal.extend(pd.read_csv(path, dtype={"patient_id": str, "center": str}).to_dict("records"))
                    continue
                print(f"Internal repetition {repetition + 1}, fold {fold + 1}: {name}", flush=True)
                audit.record("outer_evaluation", train, validation, model=name, repetition=repetition + 1, fold=fold + 1)
                seed = config["seed"] + 10000 * repetition + 100 * fold + model_index
                model, threshold_rows = factory.fit_with_threshold(name, train, seed)
                rows = prediction_rows(name, model, validation, repetition=repetition + 1, fold=fold + 1)
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "model_metadata.json", {"threshold": model.threshold, **model.metadata()})
                if threshold_rows:
                    pd.DataFrame(threshold_rows).to_csv(directory / "threshold_training_oof.csv", index=False)
                if config["execution"]["save_outer_models"]:
                    joblib.dump(model, directory / "model.joblib", compress=3)
                pd.DataFrame(rows).to_csv(path, index=False)
                internal.extend(rows)
                del model
                gc.collect()
    pd.DataFrame(assignments).to_csv(output / "patient_folds.csv", index=False)
    internal_table = pd.DataFrame(internal)
    internal_table.to_csv(output / "internal_predictions.csv", index=False)
    summarize_internal(internal_table, output)
    external_rows = []
    for model_index, name in enumerate(config["execution"]["models"]):
        directory = output / "final_models" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "external_predictions.csv"
        if resume and path.exists():
            external_rows.extend(pd.read_csv(path, dtype={"patient_id": str, "center": str}).to_dict("records"))
            continue
        print(f"Refitting all {len(development)} development patients: {name}", flush=True)
        audit.record("external_evaluation", development, external, model=name)
        model, threshold_rows = factory.fit_with_threshold(name, development, config["seed"] + 90001 + model_index)
        joblib.dump(model, directory / "model.joblib", compress=3)
        write_json(directory / "model_metadata.json", {"threshold": model.threshold, **model.metadata()})
        if threshold_rows:
            pd.DataFrame(threshold_rows).to_csv(directory / "threshold_training_oof.csv", index=False)
        rows = prediction_rows(name, model, external)
        if name == "habitat":
            if config["execution"]["save_external_maps"]:
                model.transform(external, output / "external_habitats")
            if config["evaluation"]["shap_enabled"]:
                from .visualization import explain_habitat
                explain_habitat(model, external, output / "shap", config)
        pd.DataFrame(rows).to_csv(path, index=False)
        external_rows.extend(rows)
        del model
        gc.collect()
    external_table = pd.DataFrame(external_rows)
    external_table.to_csv(output / "external_predictions.csv", index=False)
    evaluate_external(external_table, output / "evaluation", config["evaluation"])
    if config["execution"]["matlab_statistics"]:
        from .matlab_bridge import evaluate_matlab, get_service
        evaluate_matlab(output / "external_predictions.csv", output / "matlab_evaluation", output / "config.json", config)
        versions["matlab"] = (get_service(config).directory / "READY").read_text()
        write_json(output / "environment.json", versions)
    write_json(state_path, {"fingerprint": fingerprint, "status": "complete", "development_n": len(development), "external_n": len(external)})
    return output


def predict_saved(model_path, manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    patients = read_manifest(manifest, require_labels=False)
    store = CaseStore(output / "cache", model.config)
    if hasattr(model, "engine"):
        model.engine = FeatureEngine(store, model.config)
    else:
        model.store = store
    name = model.metadata()["model"]
    table = pd.DataFrame(prediction_rows(name, model, patients))
    table.to_csv(output / "predictions.csv", index=False)
    return table
