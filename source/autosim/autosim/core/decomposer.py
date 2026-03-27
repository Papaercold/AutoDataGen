import json
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict
from pathlib import Path

from dacite import from_dict
from isaaclab.utils import configclass

from autosim.core.logger import AutoSimLogger

from .types import DecomposeResult, EnvExtraInfo


@configclass
class DecomposerCfg:
    """Configuration for the decomposer."""

    class_type: type = MISSING
    """The class type of the decomposer."""

    cache_dir: str = "~/.cache/autosim/decomposer_cache"
    """The cache directory for the decomposer."""

    debug_output_dir: str | None = None
    """If set, decomposition results are also written to this directory as {task_name}.json. Useful for debugging."""


class Decomposer(ABC):
    def __init__(self, cfg: DecomposerCfg) -> None:
        self.cfg = cfg
        self._cache_dir = Path(self.cfg.cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._debug_output_dir = Path(self.cfg.debug_output_dir).expanduser() if self.cfg.debug_output_dir else None
        if self._debug_output_dir:
            self._debug_output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = AutoSimLogger("Decomposer")

    @abstractmethod
    def decompose(self, extra_info: EnvExtraInfo) -> DecomposeResult:
        """Decompose the task with the given extra information."""

        raise NotImplementedError(f"{self.__class__.__name__}.decompose() must be implemented.")

    def is_cache_hit(self, task_name: str) -> bool:
        """Check if the cache hit for the given task name."""
        return (self._cache_dir / f"{task_name}.json").exists()

    def write_cache(self, task_name: str, decompose_result: DecomposeResult) -> None:
        """Write the cache for the given task name."""

        data = asdict(decompose_result)
        with open(self._cache_dir / f"{task_name}.json", "w") as f:
            json.dump(data, f, indent=4)
        if self._debug_output_dir:
            debug_path = self._debug_output_dir / f"{task_name}.json"
            with open(debug_path, "w") as f:
                json.dump(data, f, indent=4)
            self._logger.info(f"Decomposition result written to {debug_path}")

    def read_cache(self, task_name: str) -> DecomposeResult:
        """Read the cache for the given task name."""

        if not (self._cache_dir / f"{task_name}.json").exists():
            raise FileNotFoundError(f"Cache file not found for task name: {task_name} in storage: {self.cfg.cache_dir}")
        with open(self._cache_dir / f"{task_name}.json") as f:
            return from_dict(DecomposeResult, json.load(f))
