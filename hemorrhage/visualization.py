from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .common import write_json


def plot_evaluation(groups, curves, output):
    output = Path(output)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), layout="constrained")
    for name, group in groups.items():
        fpr, tpr, _ = roc_curve(group.label, group.probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(group.label, group.probability):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(output / "external_roc.pdf")
    fig.savefig(output / "external_roc.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.5), layout="constrained")
    for curve in curves:
        ax.plot(curve.threshold, curve.net_benefit, label=curve.model.iloc[0])
    ax.plot(curves[0].threshold, curves[0].treat_all, linestyle="--", color="0.5", label="Treat all")
    ax.axhline(0, color="black", linewidth=1, label="Treat none")
    minimum = min(float(curve.net_benefit.min()) for curve in curves)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(min(-0.1, minimum - 0.02), 1.0))
    ax.legend(frameon=False)
    fig.savefig(output / "external_decision_curve.pdf")
    fig.savefig(output / "external_decision_curve.png", dpi=300)
    plt.close(fig)


def explain_habitat(model, patients, output, config):
    import shap
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    classifier = model.classifier
    imputer = classifier.estimator.named_steps["imputer"]
    background = imputer.transform(classifier.background)
    inputs = imputer.transform(model.selected_matrix(patients))
    rng = np.random.default_rng(config["seed"])
    maximum = config["evaluation"]["shap_background"]
    if len(background) > maximum:
        background = background[rng.choice(len(background), maximum, replace=False)]
    np.random.seed(config["seed"])
    explainer = shap.KernelExplainer(classifier.predict_selected, background, link="identity")
    values = np.asarray(explainer.shap_values(inputs, nsamples=config["evaluation"]["shap_samples"], silent=True))
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.shape != inputs.shape:
        raise ValueError("SHAP output dimensions differ from the selected feature matrix")
    table = pd.DataFrame(values, columns=classifier.selected_names)
    table.insert(0, "patient_id", [p.patient_id for p in patients])
    table.to_csv(output / "external_shap_values.csv", index=False)
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    residual = classifier.predict_selected(inputs) - expected - values.sum(axis=1)
    write_json(output / "shap_metadata.json", {"expected_probability": expected, "background_source": "development patients, original samples only", "background_count": len(background), "output_class": "HT", "max_additivity_residual": float(np.max(np.abs(residual))), "selected_features": classifier.selected_names})
    importance = pd.DataFrame({"feature": classifier.selected_names, "mean_absolute_shap": np.abs(values).mean(axis=0)}).sort_values("mean_absolute_shap")
    importance.to_csv(output / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, max(3, len(importance) * 0.3)), layout="constrained")
    ax.barh(importance.feature, importance.mean_absolute_shap)
    ax.set_xlabel("Mean absolute SHAP value")
    fig.savefig(output / "feature_importance.pdf")
    fig.savefig(output / "feature_importance.png", dpi=300)
    plt.close(fig)
