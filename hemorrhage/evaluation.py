from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .common import write_json


def validate_predictions(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both outcome classes")
    labels = labels.astype(int)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return labels, probabilities


def delong_covariance(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.atleast_2d(predictions).astype(float)
    if predictions.shape[1] != len(labels):
        raise ValueError("Predictions must have shape models by patients")
    structural_positive, structural_negative, aucs = [], [], []
    for scores in predictions:
        validate_predictions(labels, scores)
        positive, negative = scores[labels == 1], scores[labels == 0]
        kernel = (positive[:, None] > negative[None, :]).astype(float) + 0.5 * (positive[:, None] == negative[None, :])
        structural_positive.append(kernel.mean(axis=1))
        structural_negative.append(kernel.mean(axis=0))
        aucs.append(float(kernel.mean()))
    n_positive, n_negative = np.sum(labels == 1), np.sum(labels == 0)
    if min(n_positive, n_negative) < 2:
        covariance = np.full((len(aucs), len(aucs)), np.nan)
    else:
        covariance = np.atleast_2d(np.cov(structural_positive, ddof=1)) / n_positive + np.atleast_2d(np.cov(structural_negative, ddof=1)) / n_negative
    return np.asarray(aucs), covariance


def delong_test(labels, first, second):
    aucs, covariance = delong_covariance(labels, np.vstack([first, second]))
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if not np.isfinite(variance):
        p_value, reason = np.nan, "insufficient_class_count"
    elif variance <= 1e-15:
        p_value = 1.0 if abs(difference) <= 1e-15 else np.nan
        reason = "identical_auc_zero_variance" if np.isfinite(p_value) else "zero_variance"
    else:
        p_value, reason = float(2 * norm.sf(abs(difference) / np.sqrt(variance))), "ok"
    return {"auc_first": aucs[0], "auc_second": aucs[1], "auc_difference": difference, "variance": variance, "p_value": p_value, "status": reason}


def ratio_interval(numerator, denominator, alpha=0.05):
    if denominator == 0:
        return np.nan, np.nan, np.nan
    estimate = numerator / denominator
    z = norm.ppf(1 - alpha / 2)
    denominator_adjusted = 1 + z * z / denominator
    center = (estimate + z * z / (2 * denominator)) / denominator_adjusted
    distance = z * np.sqrt(estimate * (1 - estimate) / denominator + z * z / (4 * denominator * denominator)) / denominator_adjusted
    return estimate, max(0.0, center - distance), min(1.0, center + distance)


def binary_metrics(labels, probabilities, threshold, alpha=0.05):
    labels, probabilities = validate_predictions(labels, probabilities)
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), probabilities.shape)
    if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
        raise ValueError("Invalid classification threshold")
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1]).ravel()
    aucs, covariance = delong_covariance(labels, probabilities)
    standard_error = np.sqrt(max(0.0, covariance[0, 0])) if np.isfinite(covariance[0, 0]) else np.nan
    z = norm.ppf(1 - alpha / 2)
    result = {"n": len(labels), "ht": int(labels.sum()), "cs": int(np.sum(labels == 0)), "tp": tp, "fn": fn, "tn": tn, "fp": fp, "auc": aucs[0], "auc_ci_low": max(0.0, aucs[0] - z * standard_error) if np.isfinite(standard_error) else np.nan, "auc_ci_high": min(1.0, aucs[0] + z * standard_error) if np.isfinite(standard_error) else np.nan}
    for name, numerator, denominator in [("accuracy", tp + tn, len(labels)), ("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp), ("ppv", tp, tp + fp), ("npv", tn, tn + fn)]:
        estimate, lower, upper = ratio_interval(numerator, denominator, alpha)
        result.update({name: estimate, name + "_ci_low": lower, name + "_ci_high": upper})
    return result


def select_threshold(labels, probabilities, settings):
    if settings["threshold_rule"] == "fixed":
        return float(settings["fixed_threshold"])
    labels, probabilities = validate_predictions(labels, probabilities)
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates = np.flatnonzero(valid)
    return float(thresholds[candidates[np.argmax((tpr - fpr)[valid])]])


def decision_curve(labels, probabilities, thresholds):
    labels, probabilities = validate_predictions(labels, probabilities)
    rows = []
    prevalence = labels.mean()
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("Decision thresholds must be strictly between 0 and 1")
        positive = probabilities >= threshold
        tp = np.sum(positive & (labels == 1))
        fp = np.sum(positive & (labels == 0))
        weight = threshold / (1 - threshold)
        rows.append({"threshold": threshold, "net_benefit": tp / len(labels) - fp / len(labels) * weight, "treat_all": prevalence - (1 - prevalence) * weight, "treat_none": 0.0})
    return pd.DataFrame(rows)


def aligned_external(table):
    required = {"patient_id", "model", "label", "probability", "threshold"}
    if not required.issubset(table.columns) or table.duplicated(["model", "patient_id"]).any():
        raise ValueError("External predictions require one row per patient and model")
    reference_ids, reference_labels, groups = None, None, {}
    for name, group in table.groupby("model", sort=True):
        group = group.sort_values("patient_id").reset_index(drop=True)
        validate_predictions(group.label.to_numpy(), group.probability.to_numpy())
        ids, labels = group.patient_id.tolist(), group.label.tolist()
        if reference_ids is not None and (ids != reference_ids or labels != reference_labels):
            raise ValueError("Models must be evaluated on the same patients and labels")
        reference_ids, reference_labels = ids, labels
        groups[name] = group
    return groups


def evaluate_external(table, output, settings, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = aligned_external(table)
    metrics, comparisons, curves = [], [], []
    for name, group in groups.items():
        metrics.append({"model": name, **binary_metrics(group.label, group.probability, group.threshold, settings["alpha"])})
        curve = decision_curve(group.label, group.probability, settings["dca_thresholds"])
        curve.insert(0, "model", name)
        curves.append(curve)
        if "habitat" in groups and name != "habitat":
            comparisons.append({"first_model": "habitat", "second_model": name, **delong_test(group.label, groups["habitat"].probability, group.probability)})
    pd.DataFrame(metrics).to_csv(output / "external_metrics.csv", index=False)
    pd.DataFrame(comparisons, columns=["first_model", "second_model", "auc_first", "auc_second", "auc_difference", "variance", "p_value", "status"]).to_csv(output / "delong_comparisons.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output / "decision_curves.csv", index=False)
    write_json(output / "evaluation_methods.json", {"auc_ci": "DeLong normal approximation, clipped to [0,1]", "proportion_ci": "Wilson score", "alpha": settings["alpha"], "delong": "paired, two-sided, unadjusted comparisons with habitat", "decision_curve": "exploratory"})
    if plots:
        from .visualization import plot_evaluation
        plot_evaluation(groups, curves, output)
    return pd.DataFrame(metrics)
