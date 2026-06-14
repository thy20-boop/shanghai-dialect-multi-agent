from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    passed = 0
    failed = 0
    for path in test_files:
        module = load_module(path)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                func()
            except Exception as exc:
                failed += 1
                print(f"FAIL {path.name}::{name}: {exc}")
            else:
                passed += 1
                print(f"PASS {path.name}::{name}")
    print(f"passed={passed} failed={failed}")
    return 1 if failed else 0


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
