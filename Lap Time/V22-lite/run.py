#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local runner for the bundled v22-lite training package.

Usage:
  python run.py
"""

import os
import subprocess
import sys


def ensure_packages() -> None:
    required = ["numpy", "pandas", "scikit-learn", "catboost"]
    import_names = ["numpy", "pandas", "sklearn", "catboost"]

    for pkg, mod in zip(required, import_names):
        try:
            __import__(mod)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[INSTALL] {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    print("=" * 50)
    print("v22-lite local training runner")
    print("=" * 50)

    ensure_packages()

    print()
    print("Starting v22-lite training (2-seed, CPU)")
    print("Expected runtime: several hours depending on machine")
    print()

    rc = subprocess.call([sys.executable, "-u", "v22_lite_notebook.py"])
    if rc == 0:
        print("\nDone. Check result_v22_lite for outputs.")
    else:
        print(f"\nFailed (exit code: {rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
