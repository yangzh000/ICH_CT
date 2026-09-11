import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


@dataclass(frozen=True)
class Patient:
    patient_id: str
    center: str
    image: str
    mask: str
    label: int | None


def read_manifest(path, require_labels=True):
    path = Path(path).resolve()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"patient_id", "center", "image", "mask"}
    if not required.issubset(table.columns):
        raise ValueError(f"Manifest requires columns {sorted(required)}")
    if require_labels and "label" not in table.columns:
        raise ValueError("Manifest requires label: 1 for HT and 0 for CS")
    patients = []
    for row in table.to_dict("records"):
        if any(not row[k].strip() for k in required):
            raise ValueError("Patient identifiers, centers, and paths must be nonempty")
        label = row.get("label", "").strip()
        if label not in {"0", "1", ""} or require_labels and label == "":
            raise ValueError(f"Invalid label for {row['patient_id']}")
        paths = []
        for key in ("image", "mask"):
            resolved = Path(row[key]).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            resolved = resolved.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            paths.append(str(resolved))
        patients.append(Patient(row["patient_id"], row["center"], *paths, int(label) if label else None))
    if not patients:
        raise ValueError("Manifest is empty")
    for values in ([p.patient_id for p in patients], [p.image for p in patients]):
        if len(values) != len(set(values)):
            raise ValueError("Each patient and image must occur once; merge lesions into one patient mask")
    return sorted(patients, key=lambda p: p.patient_id)


def partition_patients(patients, folds, seed):
    labels = np.asarray([p.label for p in patients])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training requires binary labels")
    counts = np.bincount(labels.astype(int), minlength=2)
    if folds < 2 or counts.min() < folds:
        raise ValueError(f"Requested {folds} folds but class counts are {counts.tolist()}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(list(np.asarray(patients, dtype=object)[a]), list(np.asarray(patients, dtype=object)[b])) for a, b in splitter.split(labels, labels)]


def ensure_disjoint(train, validation):
    overlap = {p.patient_id for p in train} & {p.patient_id for p in validation}
    if overlap:
        raise ValueError(f"Patient overlap: {sorted(overlap)}")


def patients_key(patients):
    return digest([asdict(p) for p in sorted(patients, key=lambda p: p.patient_id)])


class FitAudit:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.events = []

    def record(self, kind, train, validation=(), **values):
        ensure_disjoint(train, validation)
        event = {"kind": kind, "training_patients": sorted(p.patient_id for p in train), "evaluation_patients": sorted(p.patient_id for p in validation), **values}
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(clean_json(event), ensure_ascii=False) + "\n")
