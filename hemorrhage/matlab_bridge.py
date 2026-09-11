import atexit
import os
import shutil
import subprocess
import tempfile
import time
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


_SERVICES = {}


def matlab_executable(requested):
    if requested != "auto":
        resolved = shutil.which(requested) or (requested if Path(requested).is_file() else None)
        if not resolved:
            raise FileNotFoundError(f"MATLAB executable not found: {requested}")
        return str(resolved)
    resolved = shutil.which("matlab")
    if resolved:
        return resolved
    installations = sorted(Path("/Applications").glob("MATLAB_R*.app/bin/matlab"), reverse=True)
    if installations:
        return str(installations[0])
    raise FileNotFoundError("MATLAB is required for SFS; configure matlab.executable")


class MatlabService:
    def __init__(self, settings):
        self.settings = settings
        supplied = os.environ.get("HEMORRHAGE_MATLAB_EXCHANGE")
        if supplied:
            self.directory = Path(supplied).resolve()
            self.log_path = self.directory / "matlab.log"
            self.log = None
            self.process = None
            self.wait_for(self.directory / "READY", settings["startup_timeout_seconds"])
            return
        self.directory = Path(tempfile.mkdtemp(prefix="hemorrhage_matlab_"))
        self.log_path = self.directory / "matlab.log"
        self.log = None
        self.process = None
        scripts = Path(__file__).resolve().parents[1] / "matlab"
        if not scripts.is_dir():
            scripts = Path(sys.prefix) / "share" / "hemorrhage" / "matlab"
        if not (scripts / "sfs_worker.m").is_file():
            raise FileNotFoundError("MATLAB SFS scripts are missing from the installation")
        quoted_scripts = str(scripts).replace("'", "''")
        quoted_directory = str(self.directory).replace("'", "''")
        command = f"addpath('{quoted_scripts}'); sfs_worker('{quoted_directory}');"
        try:
            self.log = self.log_path.open("w")
            self.process = subprocess.Popen([matlab_executable(settings["executable"]), "-nodesktop", "-nosplash", "-batch", command], stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.directory)
            self.wait_for(self.directory / "READY", settings["startup_timeout_seconds"])
        except BaseException:
            self.close()
            raise

    def wait_for(self, path, timeout):
        deadline = time.monotonic() + timeout
        while not path.exists():
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"MATLAB worker exited: {self.log_path.read_text()[-6000:]}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"MATLAB request timed out; log: {self.log_path}")
            time.sleep(0.05)

    def select(self, values, labels, folds, maximum, C, gamma):
        train = np.empty((len(folds), 1), dtype=object)
        validation = np.empty((len(folds), 1), dtype=object)
        for index, (a, b) in enumerate(folds):
            train[index, 0] = (np.asarray(a, dtype=np.float64) + 1).reshape(-1, 1)
            validation[index, 0] = (np.asarray(b, dtype=np.float64) + 1).reshape(-1, 1)
        result = self.request({"X": np.asarray(values, dtype=np.float64), "y": np.asarray(labels, dtype=np.float64).reshape(-1, 1), "trainFolds": train, "validationFolds": validation, "maxFeatures": float(maximum), "C": float(C), "gammaSpec": gamma, "randomSeed": float(self.settings["seed"])})
        return {"selected": np.atleast_1d(result["selected_indices"]).astype(int) - 1, "order": np.atleast_1d(result["feature_order"]).astype(int) - 1, "scores": np.atleast_1d(result["mean_auc_history"]).astype(float)}

    def request(self, payload):
        identifier = uuid.uuid4().hex
        temporary = self.directory / (identifier + ".pending")
        request = self.directory / ("request_" + identifier + ".mat")
        response = self.directory / ("response_" + identifier + ".mat")
        savemat(temporary, payload, appendmat=False)
        temporary.replace(request)
        self.wait_for(response, self.settings["request_timeout_seconds"])
        result = loadmat(response, squeeze_me=True, simplify_cells=True)
        response.unlink()
        if "error_message" in result:
            raise RuntimeError(f"MATLAB request failed: {result['error_message']}")
        return result

    def close(self):
        if self.process is None:
            (self.directory / "STOP").write_text("stop")
            if self.log is not None:
                self.log.close()
            return
        if self.process.poll() is None:
            (self.directory / "STOP").write_text("stop")
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()


def matlab_select(batch, folds, config):
    service = get_service(config)
    maximum = config["svm"]["max_features"]
    maximum = batch.values.shape[1] if maximum is None else min(maximum, batch.values.shape[1])
    result = service.select(batch.values, batch.labels, folds, maximum, config["svm"]["C"], config["svm"]["gamma"])
    history = [{"size": index + 1, "mean_auc": score, "features": [batch.names[column] for column in result["order"][:index + 1]]} for index, score in enumerate(result["scores"])]
    return result["selected"].tolist(), history


def get_service(config):
    settings = config["matlab"]
    settings = {**settings, "seed": config["seed"]}
    key = tuple(sorted(settings.items()))
    if key not in _SERVICES:
        _SERVICES[key] = MatlabService(settings)
    return _SERVICES[key]


def evaluate_matlab(predictions, output, configuration, config):
    service = get_service(config)
    result = service.request({"operation": "evaluate", "predictionFile": str(Path(predictions).resolve()), "outputDirectory": str(Path(output).resolve()), "configFile": str(Path(configuration).resolve())})
    return str(result["status"])


def close_matlab_services():
    for service in list(_SERVICES.values()):
        service.close()
    _SERVICES.clear()


atexit.register(close_matlab_services)
