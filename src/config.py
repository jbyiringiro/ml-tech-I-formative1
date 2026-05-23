"""Configuration loader.

Reads ``config.yaml`` from the project root, resolves all relative paths to
absolute paths, and exposes a single ``CONFIG`` object used across the whole
project. Centralising configuration keeps experiments reproducible: every
hyper-parameter and path lives in one auditable file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = parent directory of this ``src`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


class Config:
    """Lightweight wrapper around the parsed ``config.yaml`` dictionary.

    Supports both attribute access (``cfg.dates``) and dictionary access
    (``cfg["dates"]``). Path entries are returned as absolute :class:`Path`
    objects and the corresponding directories are created on demand.
    """

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {path}. Run scripts from the project root."
            )
        with open(path, "r", encoding="utf-8") as fh:
            self._raw: dict[str, Any] = yaml.safe_load(fh)
        self.path = path

    # -- generic access --------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self._raw[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    # -- path helpers ----------------------------------------------------
    def path_of(self, key: str, create: bool = True) -> Path:
        """Return an absolute :class:`Path` for a ``paths:`` entry.

        Parameters
        ----------
        key : one of the keys under ``paths:`` in config.yaml.
        create : when True (default) the directory is created if missing.
        """
        rel = self._raw["paths"][key]
        abs_path = (PROJECT_ROOT / rel).resolve()
        if create:
            abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path

    # convenient named properties -------------------------------------
    @property
    def raw_telecom(self) -> Path:
        return self.path_of("raw_telecom")

    @property
    def raw_grid(self) -> Path:
        return self.path_of("raw_grid")

    @property
    def processed(self) -> Path:
        return self.path_of("processed")

    @property
    def figures(self) -> Path:
        return self.path_of("results_figures")

    @property
    def tables(self) -> Path:
        return self.path_of("results_tables")

    @property
    def experiments_dir(self) -> Path:
        return self.path_of("experiments")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(path={self.path}, sections={list(self._raw)})"


# Singleton used everywhere else in the project.
CONFIG = Config()
