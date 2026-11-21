"""Locate the external csxsl checkout and keep report references portable."""

import os
from pathlib import Path


ENGINE = Path(__file__).resolve().parent


def operator_root():
    configured = os.environ.get("PREROUND_OPERATOR_ROOT")
    root = Path(configured).expanduser() if configured else ENGINE.parent.parent / "Preround"
    root = root.resolve()
    required = ("common/validation.py", "SwiGLU_backward/solution.py")
    if root == ENGINE.parent or not all((root / name).is_file() for name in required):
        raise ValueError(f"Set PREROUND_OPERATOR_ROOT to a separate checkout of csxsl. Invalid operator root: {root}")
    return root


def source_key(path):
    path = Path(path).resolve()
    if path.is_relative_to(ENGINE):
        return path.relative_to(ENGINE.parent).as_posix()
    return path.relative_to(operator_root()).as_posix()


def report_directory(experiment):
    return operator_root() / "h20_sim" if experiment == "sim006" else ENGINE


def evidence_path(value):
    """Resolve archived h20_sim/results-* keys independently of the working directory."""
    path = Path(value)
    if path.parent == Path("h20_sim") and path.name.startswith("results-sim") and path.suffix == ".json":
        experiment = path.name.split("-", 2)[1]
        return report_directory(experiment) / path.name
    return path


def evidence_key(path):
    path = Path(path).resolve()
    key = Path("h20_sim") / path.name
    return key.as_posix() if evidence_path(key) == path else str(path)
