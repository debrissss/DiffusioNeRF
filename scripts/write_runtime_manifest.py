#!/usr/bin/env python
"""Write a compact manifest beside the data-disk runtime."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT.parent / ".runtime"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    python = RUNTIME_ROOT / "miniconda3" / "bin" / "python"
    conda = RUNTIME_ROOT / "miniconda3" / "bin" / "conda"
    package_list = subprocess.check_output([str(conda), "list", "--explicit"], text=True)
    (RUNTIME_ROOT / "conda-explicit.txt").write_text(package_list)
    files = {}
    for relative in ["scripts/runtime_env.sh", "scripts/runtime_check.py", "run_DiffusioNeRF_LLFF_3v_NeurTV_Ray.sh", "environment.yml", "requirements.txt"]:
        path = ROOT / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema": 1,
        "project_root_relative_runtime": "../.runtime",
        "python": str(python),
        "python_version": subprocess.check_output([str(python), "-c", "import sys; print(sys.version)"], text=True).strip(),
        "conda_prefix": str(RUNTIME_ROOT / "miniconda3"),
        "files": files,
    }
    (RUNTIME_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
