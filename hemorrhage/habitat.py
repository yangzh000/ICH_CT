import itertools
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .common import FitAudit, digest, partition_patients, patients_key
from .imaging import Volume
from .selection import GaussianSFS


class HabitatDegeneracyError(ValueError):
    pass


def patch_bounds(center, shape, size):
    bounds = []
    for coordinate, length in zip(center, shape):
        start = min(max(int(coordinate) - size // 2, 0), max(0, length - size))
        bounds.append((start, min(start + size, length)))
    return tuple(bounds)


def add_box(difference, bounds, value):
    for corners in itertools.product((0, 1), repeat=3):
        index = tuple(bounds[axis][corner] for axis, corner in enumerate(corners))
        difference[index] += value * (-1 if sum(corners) % 2 else 1)


def cumulative_boxes(difference):
    return difference.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)[:-1, :-1, :-1]


def prune_clusters(probabilities, mask, clusters, seed, n_init):
    values = np.asarray(probabilities[mask], dtype=float)
    if len(values) < clusters or not np.isfinite(values).all() or len(np.unique(np.round(values, 12))) < clusters:
        raise HabitatDegeneracyError(f"Cannot form {clusters} distinct clusters from the current probability map")
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=clusters, random_state=seed, n_init=n_init).fit_predict(values[:, None])
    present = np.unique(labels)
    if len(present) != clusters:
        raise HabitatDegeneracyError(f"K-means produced fewer than {clusters} nonempty clusters")
    means = np.asarray([values[labels == k].mean() for k in range(clusters)])
    removed = int(np.argmin(means))
    retained = np.zeros(mask.shape, dtype=bool)
    retained[mask] = labels != removed
    if retained.sum() < 2 or retained.sum() >= np.sum(mask):
        raise HabitatDegeneracyError("Cluster removal did not leave a valid smaller ROI")
    return retained, {"cluster_means": means.tolist(), "removed_cluster": removed, "removed_voxels": int(np.sum(labels == removed)), "retained_voxels": int(retained.sum())}


class PatchMapper:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def probability_map(self, patient, mask, model):
        volume = self.engine.store.get(patient)
        settings = self.config["habitat"]
        size = settings["patch_size"]
        centers = np.argwhere(mask)
        if settings["stride"] > 1:
            anchor = centers.min(axis=0)
            centers = centers[np.all((centers - anchor) % settings["stride"] == 0, axis=1)]
        shape = tuple(np.asarray(mask.shape) + 1)
        sums = np.zeros(shape, dtype=np.float64)
        counts = np.zeros(shape, dtype=np.int64)
        names = sorted(model.selected_names)
        reorder = [names.index(name) for name in model.selected_names]
        root = Path(self.engine.store.root) / "patches"
        root.mkdir(parents=True, exist_ok=True)
        source = self.engine.store.prepare(patient)
        database = root / (digest(source) + ".sqlite")
        signature = digest([names, size, self.config["radiomics"]])
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            for offset in range(0, len(centers), settings["patch_batch_size"]):
                block = centers[offset:offset + settings["patch_batch_size"]]
                bounds_list, rows = [], []
                for center in block:
                    bounds = patch_bounds(center, mask.shape, size)
                    key = digest([signature, bounds])
                    cached = connection.execute("SELECT value FROM features WHERE key = ?", (key,)).fetchone()
                    if cached is None:
                        slices = tuple(slice(a, b) for a, b in bounds)
                        image = np.asarray(volume.image[slices])
                        image = np.pad(image, tuple((0, size - length) for length in image.shape), mode="edge")
                        patch = Volume(image, np.ones(image.shape, dtype=bool), volume.spacing, volume.origin, volume.direction)
                        _, values = self.engine.extract(patch, names=names)
                        connection.execute("INSERT OR REPLACE INTO features VALUES (?, ?)", (key, values.astype(np.float64).tobytes()))
                    else:
                        values = np.frombuffer(cached[0], dtype=np.float64)
                    rows.append(values[reorder])
                    bounds_list.append(bounds)
                probabilities = model.predict_selected(np.vstack(rows))
                for bounds, probability in zip(bounds_list, probabilities):
                    add_box(sums, bounds, float(probability))
                    add_box(counts, bounds, 1)
                connection.commit()
        sum_map, coverage = cumulative_boxes(sums), cumulative_boxes(counts)
        if not np.all(coverage[mask] > 0):
            raise HabitatDegeneracyError("Patch grid does not cover every ROI voxel")
        result = np.full(mask.shape, np.nan, dtype=np.float64)
        result[mask] = np.clip(sum_map[mask] / coverage[mask], 0, 1)
        return result


class HabitatBuilder:
    def __init__(self, engine, config, audit=None):
        self.engine = engine
        self.config = config
        self.audit = audit or FitAudit()
        self.mapper = PatchMapper(engine, config)
        self.chains = OrderedDict()
        self.mask_root = Path(engine.store.root) / "crossfit_masks"
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.namespace = digest(config)

    def apply_chain(self, patient, chain, capture=False):
        mask = np.asarray(self.engine.store.get(patient).mask).copy()
        maps, details = [], []
        for index, stage in enumerate(chain):
            probabilities = self.mapper.probability_map(patient, mask, stage)
            seed = int(digest([self.config["seed"], patient.patient_id, index])[:8], 16)
            mask, detail = prune_clusters(probabilities, mask, self.config["habitat"]["clusters"], seed, self.config["habitat"]["kmeans_n_init"])
            if capture:
                maps.append((probabilities, mask.copy()))
                details.append(detail)
        return mask, maps, details

    def cohort_key(self, patients):
        return digest([patients_key(patients), [self.engine.store.prepare(p) for p in patients]])

    def crossfit_masks(self, patients, depth):
        if depth == 0:
            return {p.patient_id: np.asarray(self.engine.store.get(p).mask) for p in patients}
        key = digest([self.namespace, self.cohort_key(patients), depth])
        paths = {p.patient_id: self.mask_root / (digest([key, p.patient_id]) + ".npz") for p in patients}
        if all(path.exists() for path in paths.values()):
            result = {}
            for patient_id, path in paths.items():
                with np.load(path, allow_pickle=False) as saved:
                    result[patient_id] = saved["mask"]
            return result
        result = {}
        splits = partition_patients(patients, self.config["cv"]["crossfit_folds"], self.config["seed"] + 307 + depth)
        for fold, (train, validation) in enumerate(splits):
            self.audit.record("habitat_crossfit", train, validation, depth=depth, fold=fold)
            chain = self.fit_chain(train, depth)
            for patient in validation:
                if any(patient.patient_id in stage.training_ids for stage in chain):
                    raise RuntimeError("Out-of-fold habitat chain contains the evaluated patient")
                mask, _, _ = self.apply_chain(patient, chain)
                result[patient.patient_id] = mask
                np.savez_compressed(paths[patient.patient_id], mask=mask)
        return result

    def fit_chain(self, patients, depth):
        if depth == 0:
            return []
        key = (self.cohort_key(patients), depth)
        if key in self.chains:
            self.chains.move_to_end(key)
            return self.chains[key]
        previous = self.fit_chain(patients, depth - 1)
        masks = self.crossfit_masks(patients, depth - 1)
        family = "all" if depth == 1 else self.config["habitat"]["later_feature_family"]
        batch = self.engine.matrix(patients, masks, training=True, family=family)
        self.audit.record("local_svm_fit", patients, depth=depth)
        stage = GaussianSFS(self.config, self.config["seed"] + 401 + depth).fit(batch)
        result = previous + [stage]
        self.chains[key] = result
        while len(self.chains) > self.config["execution"]["model_cache_size"]:
            self.chains.popitem(last=False)
        return result


class RadiomicsClassifier:
    def __init__(self, engine, config, seed):
        self.engine, self.config, self.seed = engine, config, seed

    def fit(self, patients):
        self.training_ids = sorted(p.patient_id for p in patients)
        self.classifier = GaussianSFS(self.config, self.seed).fit(self.engine.matrix(patients, training=True))
        return self

    def predict(self, patients):
        return self.classifier.predict(self.engine.matrix(patients))

    def selected_matrix(self, patients):
        return self.engine.matrix(patients).values[:, self.classifier.columns]

    def metadata(self):
        return {"model": "radiomics", **self.classifier.metadata()}


class HabitatClassifier:
    def __init__(self, engine, config, seed, depth):
        self.engine, self.config, self.seed, self.depth = engine, config, seed, depth

    def fit(self, patients, builder):
        self.training_ids = sorted(p.patient_id for p in patients)
        self.chain = builder.fit_chain(patients, self.depth)
        masks = builder.crossfit_masks(patients, self.depth)
        family = "all" if self.depth == 0 else self.config["habitat"]["final_feature_family"]
        self.classifier = GaussianSFS(self.config, self.seed).fit(self.engine.matrix(patients, masks, training=True, family=family))
        return self

    def transform(self, patients, output=None):
        builder = HabitatBuilder(self.engine, self.config)
        masks = {}
        for patient in patients:
            mask, maps, details = builder.apply_chain(patient, self.chain, capture=output is not None)
            masks[patient.patient_id] = mask
            if output is not None:
                from .common import write_json
                directory = Path(output) / digest(patient.patient_id)[:16]
                directory.mkdir(parents=True, exist_ok=True)
                for index, (probability, retained) in enumerate(maps, 1):
                    self.engine.store.save_map(patient, probability, directory / f"probability_{index:02d}.nii.gz")
                    self.engine.store.save_mask(patient, retained, directory / f"habitat_{index:02d}.nii.gz")
                self.engine.store.save_mask(patient, mask, directory / "final_habitat.nii.gz")
                write_json(directory / "iterations.json", {"patient_id": patient.patient_id, "depth": self.depth, "iterations": details})
        return masks

    def feature_batch(self, patients, output=None):
        masks = self.transform(patients, output)
        family = "all" if self.depth == 0 else self.config["habitat"]["final_feature_family"]
        return self.engine.matrix(patients, masks, family=family)

    def predict(self, patients):
        return self.classifier.predict(self.feature_batch(patients))

    def selected_matrix(self, patients):
        return self.feature_batch(patients).values[:, self.classifier.columns]

    def metadata(self):
        return {"model": "habitat", "depth": self.depth, "depth_cap_reached": getattr(self, "depth_cap_reached", False), "depth_selection": getattr(self, "depth_history", []), "stages": [stage.metadata() for stage in self.chain], **self.classifier.metadata()}
