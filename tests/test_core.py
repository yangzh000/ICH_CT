import ast
import io
import json
import tokenize
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score

from hemorrhage.common import FitAudit, Patient, partition_patients
from hemorrhage.evaluation import aligned_external, binary_metrics, decision_curve, delong_test, select_threshold
from hemorrhage.features import FeatureBatch, FeatureEngine
from hemorrhage.habitat import HabitatBuilder, HabitatDegeneracyError, PatchMapper, add_box, cumulative_boxes, patch_bounds, prune_clusters
from hemorrhage.imaging import CaseStore, Volume, crop_volume
from hemorrhage.selection import GaussianSFS, grouped_feature_splits, svm_pipeline


def sample_batch(n=40):
    rng = np.random.default_rng(73)
    labels = np.arange(n) % 2
    data = rng.normal(size=(n, 3))
    data[:, 0] += labels * 4
    values = np.repeat(data, 2, axis=0)
    values[1::2] += rng.normal(0, 0.02, (n, 3))
    values[4, 2] = np.nan
    return FeatureBatch(values, ["signal", "noise_a", "noise_b"], np.repeat(labels, 2), np.repeat([f"p{i:03d}" for i in range(n)], 2), np.tile([True, False], n))


def test_group_splits_exclude_augmented_validation():
    batch = sample_batch()
    splits = grouped_feature_splits(batch, 5, 91, 5)
    coverage = np.zeros(len(batch.labels), dtype=int)
    for train, validation in splits:
        assert not set(batch.groups[train]) & set(batch.groups[validation])
        assert batch.original[validation].all()
        coverage[validation] += 1
    assert np.all(coverage[batch.original] == 5)
    assert np.all(coverage[~batch.original] == 0)


def test_box_average_matches_naive():
    rng = np.random.default_rng(9)
    shape = (27, 23, 24)
    summed = np.zeros(tuple(np.array(shape) + 1))
    counted = np.zeros_like(summed)
    naive_sum = np.zeros(shape)
    naive_count = np.zeros(shape)
    for center in rng.integers(0, 22, (30, 3)):
        bounds = patch_bounds(center, shape, 20)
        probability = rng.random()
        add_box(summed, bounds, probability)
        add_box(counted, bounds, 1)
        slices = tuple(slice(a, b) for a, b in bounds)
        naive_sum[slices] += probability
        naive_count[slices] += 1
    np.testing.assert_allclose(cumulative_boxes(summed), naive_sum, atol=1e-12)
    np.testing.assert_array_equal(cumulative_boxes(counted), naive_count)
    assert patch_bounds((0, 0, 0), shape, 20) == ((0, 20), (0, 20), (0, 20))


def test_cluster_removal_and_reclustering():
    scores = np.linspace(0.01, 0.99, 1000).reshape(10, 10, 10)
    mask = np.ones(scores.shape, dtype=bool)
    first, detail = prune_clusters(scores, mask, 10, 37, 10)
    second, again = prune_clusters(scores, first, 10, 38, 10)
    assert len(detail["cluster_means"]) == len(again["cluster_means"]) == 10
    assert scores[mask & ~first].max() < scores[first].min()
    assert np.all(second <= first)
    assert 0 < second.sum() < first.sum() < mask.sum()
    with pytest.raises(HabitatDegeneracyError):
        prune_clusters(np.ones_like(scores), mask, 10, 37, 10)


def test_metrics_ties_and_degenerate_cases():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.4, 0.9, 0.6, 0.2, 0.7])
    result = binary_metrics(labels, probabilities, 0.5)
    assert result["auc"] == roc_auc_score(labels, probabilities)
    assert [result[k] for k in ["tp", "fn", "tn", "fp"]] == [3, 1, 3, 1]
    assert delong_test(labels, probabilities, probabilities)["p_value"] == 1
    assert np.isnan(binary_metrics(labels, np.zeros(8), 0.5)["ppv"])
    curve = decision_curve(labels, probabilities, [0.5])
    assert curve.net_benefit.iloc[0] == 0.25
    with pytest.raises(ValueError):
        binary_metrics([0, 0.5, 1], [0.2, 0.4, 0.8], 0.5)


def test_alignment_and_threshold():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.5, 0.9]
    assert select_threshold(labels, probabilities, {"threshold_rule": "youden"}) == 0.5
    table = pd.DataFrame({"patient_id": ["a", "b", "a", "b"], "model": ["a", "a", "b", "b"], "label": [0, 1, 1, 0], "probability": [0.2, 0.7, 0.3, 0.8], "threshold": 0.5})
    with pytest.raises(ValueError):
        aligned_external(table)


def test_image_geometry_features_and_context(tmp_path, config):
    image = sitk.GetImageFromArray(np.arange(12 ** 3, dtype=np.float32).reshape(12, 12, 12))
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    mask_array = np.zeros((12, 12, 12), dtype=np.uint8)
    mask_array[3:9, 3:9, 3:9] = 1
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    patient = Patient("p", "A", str(image_path), str(mask_path), 1)
    store = CaseStore(tmp_path / "cache", config)
    volume = store.get(patient)
    assert volume.image.shape == (23, 12, 12)
    assert volume.spacing == (1.0, 1.0, 1.0)
    assert volume.origin == (10.0, 20.0, 30.0)
    crop = crop_volume(volume, (slice(2, 8), slice(3, 9), slice(4, 10)))
    assert crop.origin == (14.0, 23.0, 32.0)
    engine = FeatureEngine(store, config)
    original = engine.matrix([patient])
    augmented = engine.matrix([patient], training=True)
    assert original.values.shape == (1, 4)
    assert augmented.values.shape == (2, 4)
    assert np.isfinite(augmented.values).all()
    full = Volume(np.ones((20, 20, 20)), np.ones((20, 20, 20), dtype=bool), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
    names, values = engine.extract(full, names=["original_firstorder_Mean"])
    assert values[0] == 1


def test_patch_uses_external_roi_context_and_real_radiomics(tmp_path, config):
    z, y, x = np.indices((32, 32, 32))
    image = (z ** 2 + y + 0.1 * x).astype(np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[2:30, 15:17, 15:17] = True
    volume = Volume(image, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
    store = SimpleNamespace(root=tmp_path, get=lambda patient: volume, prepare=lambda patient: "synthetic")
    engine = FeatureEngine(store, config)
    model = SimpleNamespace(selected_names=["original_firstorder_Mean"], predict_selected=lambda values: 1 / (1 + np.exp(-values[:, 0] / 1000)))
    probabilities = PatchMapper(engine, config).probability_map(SimpleNamespace(patient_id="p"), mask, model)
    assert np.isfinite(probabilities[mask]).all()
    assert np.isnan(probabilities[~mask]).all()
    retained, details = prune_clusters(probabilities, mask, 10, 51, 10)
    assert 0 < retained.sum() < mask.sum()
    assert len(details["cluster_means"]) == 10
    image[~mask] += 100
    store.prepare = lambda patient: "changed_context"
    changed = PatchMapper(engine, config).probability_map(SimpleNamespace(patient_id="p"), mask, model)
    assert np.all(changed[mask] > probabilities[mask])


def test_recursive_crossfit_excludes_all_prior_sources(tmp_path, config, monkeypatch):
    patients = [Patient(str(i), "A", "", "", i % 2) for i in range(32)]
    store = SimpleNamespace(root=tmp_path, get=lambda p: SimpleNamespace(mask=np.zeros((2, 2, 2), dtype=np.int64)), prepare=lambda p: p.patient_id)

    class Engine:
        def __init__(self):
            self.store = store

        def matrix(self, scope, masks, training, family):
            own = {p.patient_id for p in scope}
            dependency_bits = 0
            for mask in masks.values():
                dependency_bits |= int(mask.flat[0])
            dependencies = {str(i) for i in range(32) if dependency_bits & (1 << i)}
            return SimpleNamespace(sources=own | dependencies)

    class Model:
        def __init__(self, *args):
            pass

        def fit(self, batch):
            self.training_ids = sorted(batch.sources)
            return self

    monkeypatch.setattr("hemorrhage.habitat.GaussianSFS", Model)
    audit_path = tmp_path / "audit.jsonl"
    builder = HabitatBuilder(Engine(), config, FitAudit(audit_path))
    applications = []

    def apply(patient, chain, capture=False):
        sources = set().union(*(set(stage.training_ids) for stage in chain))
        assert patient.patient_id not in sources
        applications.append((patient.patient_id, sources, len(chain)))
        encoded = sum(1 << int(source) for source in sources)
        return np.full((2, 2, 2), encoded, dtype=np.int64), [], []

    builder.apply_chain = apply
    masks = builder.crossfit_masks(patients, 3)
    assert len(masks) == 32
    assert any(depth == 3 for _, _, depth in applications)
    assert all(int(mask.flat[0]) & (1 << int(patient_id)) == 0 for patient_id, mask in masks.items())
    for event in map(json.loads, audit_path.read_text().splitlines()):
        assert not set(event["training_patients"]) & set(event["evaluation_patients"])


def test_source_files_have_no_comments_or_docstrings():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(text).readline)), path
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node) is None, path
    for path in root.rglob("*.m"):
        assert not any(line.lstrip().startswith("%") for line in path.read_text().splitlines()), path


def test_threshold_is_selected_from_training_oof(config, tmp_path):
    from hemorrhage.workflow import ModelFactory
    config["evaluation"]["threshold_rule"] = "youden"
    patients = [Patient(f"p{i:03d}", "A", "", "", i % 2) for i in range(20)]
    factory = object.__new__(ModelFactory)
    factory.config = config
    factory.audit = FitAudit(tmp_path / "threshold_audit.jsonl")
    fitted_sets = []

    def fit(name, training, seed):
        ids = {p.patient_id for p in training}
        fitted_sets.append(ids)

        def predict(evaluation):
            assert not ids & {p.patient_id for p in evaluation}
            return np.asarray([0.8 if p.label else 0.2 for p in evaluation])

        return SimpleNamespace(predict=predict, training_ids=ids)

    factory.fit = fit
    model, rows = factory.fit_with_threshold("radiomics", patients, 7)
    assert model.training_ids == {p.patient_id for p in patients}
    assert len(rows) == len(patients)
    assert len({row["patient_id"] for row in rows}) == len(patients)
    assert model.threshold == 0.8
    assert len(fitted_sets) == config["cv"]["threshold_folds"] + 1
