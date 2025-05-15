import os

from vila_u.utils import io

__all__ = ["EVAL_ROOT", "TASKS"]


EVAL_ROOT = "scripts/eval"
TASKS = io.load(os.path.join(os.path.dirname(__file__), "registry.yaml"))
