"""Load the two hyphenated directories as isolated packages for comparisons."""

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
for name, directory in (("cpu_reference", "og-cpu-implementation"),
                        ("torch_impl", "torch-implementation")):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / directory / "__init__.py",
        submodule_search_locations=[str(ROOT / directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

# Tiny correctness cases should not oversubscribe the CPU's thread pool.
torch.set_num_threads(1)
