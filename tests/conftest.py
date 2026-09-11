from pathlib import Path

import pytest

from hemorrhage.config import load_config
from hemorrhage.demo import demo_configuration
from hemorrhage.matlab_bridge import close_matlab_services, get_service


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return demo_configuration(load_config(root / "config.study.json"))


@pytest.fixture(scope="session")
def matlab():
    root = Path(__file__).resolve().parents[1]
    configuration = load_config(root / "config.study.json")
    service = get_service(configuration)
    yield service
    close_matlab_services()
