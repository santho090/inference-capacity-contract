"""Minimal dependency-free PEP 517 build backend."""

from __future__ import annotations

import base64
import hashlib
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).parent


def _metadata() -> dict[str, object]:
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return {
        "Metadata-Version": "2.3",
        "Name": "inference-capacity-contract",
        "Version": "0.1.0",
        "Summary": "Deterministic, evidence-aware capacity contracts for LLM model and hardware fit",
        "Requires-Python": ">=3.12",
        "License": "Apache-2.0",
        "Author": "Santhosh Vaithiyanathan",
        "Description-Content-Type": "text/markdown",
        "Description": (ROOT / "README.md").read_text(encoding="utf-8"),
        "_pyproject_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def _metadata_text() -> str:
    fields = _metadata()
    lines: list[str] = []
    for key, value in fields.items():
        if key.startswith("_"):
            continue
        text = str(value).replace("\n", "\n ")
        lines.append(f"{key}: {text}")
    return "\n".join(lines) + "\n"


def _files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts]


def _package_files() -> list[tuple[Path, str]]:
    package_root = ROOT / "src"
    return [(path, str(path.relative_to(package_root))) for path in package_root.rglob("*.py")]


def _wheel_name() -> str:
    return "inference_capacity_contract-0.1.0-py3-none-any.whl"


def _dist_info() -> str:
    return "inference_capacity_contract-0.1.0.dist-info"


def build_wheel(wheel_directory: str, config_settings: object = None, metadata_directory: str | None = None) -> str:
    del config_settings, metadata_directory
    destination = Path(wheel_directory) / _wheel_name()
    record_rows: list[str] = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in _package_files():
            archive.write(path, name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
            record_rows.append(f"{name},sha256={digest},{path.stat().st_size}")
        metadata_name = f"{_dist_info()}/METADATA"
        metadata_bytes = _metadata_text().encode()
        archive.writestr(metadata_name, metadata_bytes)
        record_rows.append(f"{metadata_name},,")
        wheel_name = f"{_dist_info()}/WHEEL"
        wheel_bytes = b"Wheel-Version: 1.0\nGenerator: capacity-contract-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        archive.writestr(wheel_name, wheel_bytes)
        record_rows.append(f"{wheel_name},,")
        record_name = f"{_dist_info()}/RECORD"
        record_rows.append(f"{record_name},,")
        archive.writestr(record_name, "\n".join(record_rows) + "\n")
    return destination.name


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: object = None) -> str:
    del config_settings
    dist = Path(metadata_directory) / _dist_info()
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: capacity-contract-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    return _dist_info()


def build_sdist(sdist_directory: str, config_settings: object = None) -> str:
    del config_settings
    name = "inference-capacity-contract-0.1.0.tar.gz"
    destination = Path(sdist_directory) / name
    with tarfile.open(destination, "w:gz") as archive:
        for path in _files():
            arcname = Path("inference-capacity-contract-0.1.0") / path.relative_to(ROOT)
            archive.add(path, arcname=arcname)
    return name


def get_requires_for_build_wheel(config_settings: object = None) -> list[str]:
    del config_settings
    return []


def get_requires_for_build_sdist(config_settings: object = None) -> list[str]:
    del config_settings
    return []
