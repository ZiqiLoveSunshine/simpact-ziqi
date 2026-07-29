import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    return Path(os.environ.get("SIMPACT_DATA_DIR", get_project_root() / "data"))

def get_outputs_dir() -> Path:
    return Path(os.environ.get("SIMPACT_OUTPUTS_DIR", get_project_root() / "outputs"))

def get_rollouts_dir() -> Path:
    return Path(os.environ.get("SIMPACT_ROLLOUTS_DIR", get_project_root() / "outputs" / "rollouts"))

def get_assets_dir() -> Path:
    """Top-level static rig data (robot models, camera calibration profiles) — a repo
    sibling of the code package, not shipped inside it. Env-overridable."""
    return Path(os.environ.get("SIMPACT_ASSETS_DIR", get_project_root() / "assets"))

def get_calibration_dir() -> Path:
    """Camera-calibration registry: ``assets/calibration/<profile>/``."""
    return get_assets_dir() / "calibration"

def get_materials_dir() -> Path:
    """MPM material registry: ``assets/materials/<profile>.yaml`` (+ ``bands.yaml``)."""
    return get_assets_dir() / "materials"