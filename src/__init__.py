"""GPU / CUDA environment detection.

This package exposes the detection helpers through the submodule
``src.profiling.gpu_info``.  It deliberately does **not** re-export the
symbols at package level.

Reason
------
Re-exporting with ``from .gpu_info import ...`` causes the module to be
loaded when the *package* is imported, and then again when it is executed
as ``python -m src.profiling.gpu_info``.  Python detects the double import
and emits::

    RuntimeWarning: 'src.profiling.gpu_info' found in sys.modules ...

Keeping ``__init__.py`` empty (apart from this docstring) removes the
warning and does not change the public API, because the canonical import
path has always been::

    from src.profiling.gpu_info import collect_environment

Nothing else in the codebase imports the re-exported names, so this change
is backwards compatible.
"""