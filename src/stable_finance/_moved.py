"""Old module paths, kept importable after the package was reorganized.

Every module that moved is listed in :data:`MOVED` with where it went. An
import of an old path returns the moved module itself, so existing code,
pickles and ``mock.patch`` targets keep working, and ``python -m`` on an old
path runs the moved module. A path whose module was split in two returns a
merged view of both parts.

New code should import from the new paths; this table is the whole cost of
keeping the old ones, and nothing else in the package refers to them.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types

_EVAL = "stable_finance.evaluation"
_DATA = "stable_finance.dataset"

#: Every module path published before the reorganization, and where it went.
MOVED: dict[str, tuple[str, ...]] = {
    # The evaluation pipeline, now one package per stage.
    "stable_finance.data": (f"{_EVAL}._grid", f"{_EVAL}.embeddings.contract",
                            f"{_EVAL}.returns.contract", f"{_EVAL}.portfolio.contract",
                            f"{_EVAL}.trades.contract", f"{_EVAL}.market_data"),
    "stable_finance.metrics": (f"{_EVAL}.returns.scoring", f"{_EVAL}.portfolio.performance",
                               f"{_EVAL}.market_data"),
    "stable_finance.market": (f"{_EVAL}.market_data", f"{_EVAL}.from_dataset"),
    "stable_finance.probes": (f"{_EVAL}.returns.probes",),
    "stable_finance.streaming": (f"{_EVAL}.returns.streaming",),
    "stable_finance.oracle": (f"{_EVAL}.returns.oracle",),
    "stable_finance.portfolio": (f"{_EVAL}.portfolio.performance",),
    "stable_finance.universe": (f"{_EVAL}.portfolio.universe",),
    "stable_finance.selection": (f"{_EVAL}.portfolio.selection",),
    "stable_finance.risk": (f"{_EVAL}.portfolio.risk",),
    "stable_finance.weighting": (f"{_EVAL}.portfolio.weighting",),
    "stable_finance.allocation": (f"{_EVAL}.portfolio.allocation",),
    "stable_finance.config": (f"{_EVAL}.portfolio.config",),
    "stable_finance.backtest": (f"{_EVAL}.portfolio.backtest",),
    "stable_finance.costs": (f"{_EVAL}.execution_model.costs",),
    "stable_finance.execution": (f"{_EVAL}.trades.execution", f"{_EVAL}.execution_model.fills"),
    # The dataset, now grouped by job.
    f"{_DATA}.schema": (f"{_DATA}.spec.schema",),
    f"{_DATA}.anchors": (f"{_DATA}.spec.timeline",),
    f"{_DATA}.grid": (f"{_DATA}.loader.grid",),
    f"{_DATA}.transforms": (f"{_DATA}.loader.views",),
    f"{_DATA}.views": (f"{_DATA}.loader.views",),
    f"{_DATA}.panels": (f"{_DATA}.loader.panels",),
    f"{_DATA}.panel_cache": (f"{_DATA}.loader.panels",),
    f"{_DATA}.cells": (f"{_DATA}.loader.cells",),
    f"{_DATA}.outcomes": (f"{_DATA}.labels.families",),
    f"{_DATA}.targets": (f"{_DATA}.labels.tables",),
    f"{_DATA}.build_targets": (f"{_DATA}.labels.tables",),
    f"{_DATA}.months": (f"{_DATA}.storage.locations",),
    f"{_DATA}.periods": (f"{_DATA}.storage.locations",),
    f"{_DATA}.source": (f"{_DATA}.storage.locations",),
    f"{_DATA}.mds": (f"{_DATA}.storage.mosaic",),
    f"{_DATA}.densify": (f"{_DATA}.storage.mosaic",),
    f"{_DATA}.reshard": (f"{_DATA}.storage.mosaic",),
    f"{_DATA}.daystore": (f"{_DATA}.storage.daystore",),
    f"{_DATA}.calendar": (f"{_DATA}.datasets.market1t.spec",),
    f"{_DATA}.write_mds": (f"{_DATA}.datasets.market1t.ingest", f"{_DATA}.storage.meta"),
    f"{_DATA}.shard_months": (f"{_DATA}.datasets.market1t.ingest",),
    f"{_DATA}.industry": (f"{_DATA}.datasets.market1t.industry",),
}

#: What ``python -m <old path>`` runs, for the old command modules. Several
#: now share a module, so each names its command function.
MAIN: dict[str, str] = {
    f"{_DATA}.build_targets": f"{_DATA}.labels.tables:main",
    f"{_DATA}.densify": f"{_DATA}.storage.mosaic:densify_main",
    f"{_DATA}.reshard": f"{_DATA}.storage.mosaic:reshard_main",
    f"{_DATA}.daystore": f"{_DATA}.storage.daystore:main",
    f"{_DATA}.write_mds": f"{_DATA}.datasets.market1t.ingest:main",
    f"{_DATA}.shard_months": f"{_DATA}.datasets.market1t.ingest:shard_main",
    f"{_DATA}.industry": f"{_DATA}.datasets.market1t.industry:main",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, targets: tuple[str, ...]) -> None:
        self.targets = targets

    def create_module(self, spec):
        if len(self.targets) == 1:
            return importlib.import_module(self.targets[0])
        merged = types.ModuleType(spec.name, f"Split into {', '.join(self.targets)}.")
        for target in self.targets:
            for name, value in vars(importlib.import_module(target)).items():
                if not (name.startswith("__") and name.endswith("__")):
                    merged.__dict__.setdefault(name, value)
        return merged

    def exec_module(self, module) -> None:
        """The moved module has already run; there is nothing more to execute."""

    def get_code(self, fullname):
        """``python -m <old path>`` runs that old path's command."""
        entry = MAIN.get(fullname)
        if entry is None:
            raise ImportError(f"{fullname} has no command to run; it moved to {self.targets}")
        module, function = entry.split(":")
        return compile(f"from {module} import {function} as _main\nraise SystemExit(_main())",
                       f"<{fullname}>", "exec")


class _MovedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        targets = MOVED.get(fullname)
        if targets is None:
            return None
        # The moved file's path as origin: ``python -m`` names the program from it.
        origin = importlib.util.find_spec(targets[0]).origin
        return importlib.util.spec_from_loader(fullname, _AliasLoader(targets), origin=origin)


def install() -> None:
    if not any(isinstance(finder, _MovedFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _MovedFinder())
