"""Compatibility package for public-release experiment imports.

The release repository keeps source files under ``code/experiments`` while
historical tests import modules as ``experiments.*``. Extending ``__path__``
keeps those imports working without duplicating the source tree.
"""

from pathlib import Path

_release_root = Path(__file__).resolve().parents[1]
_experiment_source = _release_root / "code" / "experiments"

if _experiment_source.is_dir():
    __path__.append(str(_experiment_source))

