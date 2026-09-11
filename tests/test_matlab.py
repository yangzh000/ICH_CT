import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hemorrhage.common import write_json
from hemorrhage.evaluation import evaluate_external
from hemorrhage.selection import GaussianSFS, grouped_feature_splits, svm_pipeline
from test_core import sample_batch


def test_matlab_sfs_mean_cv_and_python_kernel_consistency(matlab, config):
    batch = sample_batch()
    folds = grouped_feature_splits(batch, 2, 21)
    result = matlab.select(batch.values, batch.labels, folds, 2, 1.0, "scale")
    assert result["order"][0] == 0
    assert result["selected"].tolist() == result["order"][:np.argmax(result["scores"]) + 1].tolist()
    for index, actual in enumerate(result["scores"]):
        columns = result["order"][:index + 1]
        expected = []
        for train, validation in folds:
            model = svm_pipeline(config, 21).fit(batch.values[train][:, columns], batch.labels[train])
            expected.append(roc_auc_score(batch.labels[validation], model.decision_function(batch.values[validation][:, columns])))
        np.testing.assert_allclose(actual, np.mean(expected), atol=0.01)
    repeated = matlab.select(batch.values, batch.labels, folds, 2, 1.0, "scale")
    np.testing.assert_array_equal(result["order"], repeated["order"])
    np.testing.assert_allclose(result["scores"], repeated["scores"])


def test_calibrated_svm_matlab_selection(config, matlab):
    batch = sample_batch()
    fitted = GaussianSFS(config, 17).fit(batch)
    probabilities = fitted.predict(batch.subset(np.flatnonzero(batch.original)))
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert fitted.metadata()["selection_backend"] == "MATLAB fitcsvm"
    for split in fitted.calibration_groups:
        assert not set(split["training"]) & set(split["validation"])


def test_matlab_python_statistics_agree(tmp_path, config, matlab):
    rng = np.random.default_rng(51)
    labels = np.arange(60) % 2
    table = pd.concat([pd.DataFrame({"patient_id": [f"p{i:03d}" for i in range(60)], "model": name, "label": labels, "probability": np.round(rng.random(60) * 0.7 + labels * 0.3, 1), "threshold": 0.5}) for name in ["habitat", "radiomics", "resnet34", "swin"]], ignore_index=True)
    table.to_csv(tmp_path / "predictions.csv", index=False)
    write_json(tmp_path / "config.json", config)
    evaluate_external(table, tmp_path / "python", config["evaluation"], plots=False)
    response = matlab.request({"operation": "evaluate", "predictionFile": str(tmp_path / "predictions.csv"), "outputDirectory": str(tmp_path / "matlab"), "configFile": str(tmp_path / "config.json")})
    assert response["status"] == "complete"
    for name, keys in [("external_metrics.csv", ["model"]), ("decision_curves.csv", ["model", "threshold"]), ("delong_comparisons.csv", ["first_model", "second_model"])]:
        first = pd.read_csv(tmp_path / "python" / name).sort_values(keys).reset_index(drop=True)
        second = pd.read_csv(tmp_path / "matlab" / name).sort_values(keys).reset_index(drop=True)
        assert set(first.columns) == set(second.columns)
        for column in first.select_dtypes(include="number"):
            np.testing.assert_allclose(first[column], second[column], rtol=1e-11, atol=1e-12, equal_nan=True)
