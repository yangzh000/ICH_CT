import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .matlab_bridge import matlab_select


def grouped_feature_splits(batch, folds, seed, repeats=1):
    ids = sorted(set(batch.groups.tolist()))
    labels = []
    for patient_id in ids:
        group = batch.groups == patient_id
        unique = np.unique(batch.labels[group])
        if len(unique) != 1 or np.sum(group & batch.original) != 1:
            raise ValueError("Each patient needs one original row and consistent labels")
        labels.append(int(unique[0]))
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < folds:
        raise ValueError(f"SFS or calibration requests {folds} folds with class counts {counts.tolist()}")
    result = []
    for repetition in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repetition)
        for train, validation in splitter.split(ids, labels):
            train_ids = np.asarray(ids)[train]
            validation_ids = np.asarray(ids)[validation]
            a = np.flatnonzero(np.isin(batch.groups, train_ids))
            b = np.flatnonzero(np.isin(batch.groups, validation_ids) & batch.original)
            result.append((a, b))
    return result


def svm_pipeline(config, seed):
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler()), ("svm", SVC(C=config["svm"]["C"], gamma=config["svm"]["gamma"], kernel="rbf", probability=False, class_weight=None, random_state=seed))])


def forward_selection(batch, config, seed):
    splits = grouped_feature_splits(batch, config["cv"]["selection_folds"], seed, config["cv"]["selection_repeats"])
    return matlab_select(batch, splits, config)


def fit_sigmoid(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positive, negative = np.sum(labels == 1), np.sum(labels == 0)
    targets = np.where(labels == 1, (positive + 1) / (positive + 2), 1 / (negative + 2))

    def objective(parameters):
        logits = parameters[0] * scores + parameters[1]
        loss = np.mean(np.logaddexp(0, logits) - targets * logits) + 1e-10 * np.sum(parameters ** 2)
        residuals = expit(logits) - targets
        gradient = np.array([np.mean(residuals * scores), np.mean(residuals)]) + 2e-10 * parameters
        return float(loss), gradient

    result = minimize(objective, np.array([0.0, np.log((positive + 1) / (negative + 1))]), jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Sigmoid calibration failed: {result.message}")
    return result.x


class GaussianSFS:
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def fit(self, batch):
        self.names = list(batch.names)
        self.training_ids = sorted(set(batch.groups.tolist()))
        self.columns, self.selection_history = forward_selection(batch, self.config, self.seed)
        calibration = grouped_feature_splits(batch, self.config["cv"]["calibration_folds"], self.seed + 103)
        margins = np.full(len(batch.labels), np.nan)
        self.calibration_groups = []
        for fold, (train, validation) in enumerate(calibration):
            training_batch = batch.subset(train)
            columns, _ = forward_selection(training_batch, self.config, self.seed + 211 + fold)
            model = svm_pipeline(self.config, self.seed + fold)
            model.fit(training_batch.values[:, columns], training_batch.labels)
            margins[validation] = model.decision_function(batch.values[validation][:, columns])
            self.calibration_groups.append({"training": sorted(set(training_batch.groups.tolist())), "validation": sorted(set(batch.groups[validation].tolist()))})
        if not np.isfinite(margins[batch.original]).all():
            raise ValueError("Calibration did not cover all original patients")
        self.sigmoid = fit_sigmoid(margins[batch.original], batch.labels[batch.original])
        self.estimator = svm_pipeline(self.config, self.seed)
        self.estimator.fit(batch.values[:, self.columns], batch.labels)
        self.selected_names = [self.names[i] for i in self.columns]
        self.background = batch.values[batch.original][:, self.columns].copy()
        return self

    def predict_selected(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("Selected feature matrix has an unexpected shape")
        margins = self.estimator.decision_function(values)
        return expit(self.sigmoid[0] * margins + self.sigmoid[1])

    def predict(self, batch):
        if batch.names != self.names:
            raise ValueError("Feature names or order differ from fitted model")
        return self.predict_selected(batch.values[:, self.columns])

    def metadata(self):
        return {"selection_backend": "MATLAB fitcsvm", "selected_features": self.selected_names, "sfs_history": self.selection_history, "sigmoid_parameters": self.sigmoid, "training_patients": self.training_ids}
