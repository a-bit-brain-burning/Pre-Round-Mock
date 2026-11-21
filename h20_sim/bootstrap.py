"""Local-only FlashGPU-Sim bootstrap; never installs system packages or cleans."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request


HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "backend.lock.json").read_text())
DEFAULT_ROOT = Path("/home/csxsi/out/preround-h20-sim-20261110")
SHARED_RULES = """# Local simulator build workspace

- upstream/: pinned upstream Git checkout; preserve licenses and local patches.
- downloads/: immutable dependency archives and metadata; verify checksums.
- local/: extracted build dependencies and CUDA toolkit; no system installation.
- logs/: uniquely named bootstrap/build logs; retain failed attempts.
- runs/: unique smoke/export/simulation directories with inputs, outputs and logs.
- No credentials, automatic deletion, cleanup, or overwriting previous runs.
"""
RULES = SHARED_RULES + "- Engine adapters and regressions belong in H20-sim; operator tuning inputs and summaries belong in csxsl.\n"
LEGACY_RULES = SHARED_RULES + "- Source adapters and small result summaries belong in Preround/h20_sim/.\n"


def prepare(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rules = root / "AGENTS.md"
    if not rules.exists():
        if any(root.iterdir()):
            raise RuntimeError(f"Refusing ungoverned nonempty workspace: {root}")
        rules.write_text(RULES)
    elif rules.read_text() not in (RULES, LEGACY_RULES):
        raise RuntimeError(f"Review existing workspace rules first: {rules}")
    for name in ("upstream", "downloads", "local", "logs", "runs"):
        (root / name).mkdir(exist_ok=True)
    return root


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def run_logged(args, root, label, *, cwd=None, env=None):
    log = root / "logs" / f"{label}-{timestamp()}.log"
    print(json.dumps({"command": [str(x) for x in args], "log": str(log)}), flush=True)
    with log.open("x") as output:
        process = subprocess.run(args, cwd=cwd, env=env, stdout=output,
                                 stderr=subprocess.STDOUT)
    if process.returncode:
        print(log.read_text(errors="replace")[-6000:], file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, args)
    return log


def fetch(root):
    source = root / "upstream" / "FlashGPU-Sim"
    if not source.exists():
        run_logged(["git", "init", str(source)], root, "git-init")
        run_logged(["git", "remote", "add", "origin", LOCK["repository"]], root,
                   "git-remote", cwd=source)
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                            capture_output=True, text=True)
    if actual.returncode:
        run_logged(["git", "fetch", "--depth=1", "origin", LOCK["revision"]],
                   root, "git-fetch", cwd=source)
        run_logged(["git", "checkout", "--detach", "FETCH_HEAD"], root,
                   "git-checkout", cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source,
                                     text=True).strip()
    if actual != LOCK["revision"]:
        raise RuntimeError(f"Unexpected upstream revision: {actual}")
    print(json.dumps({"source": str(source), "revision": actual}), flush=True)
    return source


def probe(root):
    report = {"python": sys.executable, "python_version": sys.version,
              "root": str(root), "tools": {name: shutil.which(name) for name in
              ("git", "gcc", "g++", "make", "cmake", "nvcc", "bison", "flex", "m4", "ar")},
              "gpu_execution": False, "h20_tested": False}
    try:
        import importlib.metadata as metadata
        report["packages"] = {name: metadata.version(name)
                              for name in ("torch", "triton", "numpy")}
    except metadata.PackageNotFoundError as exc:
        report["package_probe_error"] = str(exc)
    output = root / "logs" / f"environment-{timestamp()}.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backend_state(source, library):
    from patch_backend import verify
    patch = verify(source)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=source)
    return {"library_sha256": digest(library), "patch": patch,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "config_sha256": {p.name: digest(p) for p in sorted((source / LOCK["baseline_config"]).iterdir())
                              if p.is_file()}}


def validated_build(root, source, library):
    state = backend_state(source, library)
    for path in sorted((root / "logs").glob("backend-build-*.json"), reverse=True):
        record = json.loads(path.read_text())
        if record.get("state") == state and record.get("backend") == LOCK:
            return {"build_receipt": str(path), **record}
    raise RuntimeError("No build receipt matches the current library, source patch and config; run bootstrap.py build")


def download(url, path, expected=None):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as out:
            shutil.copyfileobj(response, out)
    if expected and digest(path) != expected:
        raise RuntimeError(f"Checksum mismatch, retained for inspection: {path}")


def link_tree(source, target):
    """Combine extracted official packages without replacing any existing file."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            link_tree(item, dest)
        elif not dest.exists() and not dest.is_symlink():
            dest.symlink_to(item.resolve())
        elif dest.resolve() != item.resolve() and digest(dest) != digest(item):
            raise RuntimeError(f"Conflicting component file: {dest}")


def dependencies(root):
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    manifest = root / "downloads" / "redistrib_13.1.0.json"
    download(base + manifest.name, manifest)
    data = json.loads(manifest.read_text())
    cuda = root / "local" / "cuda-13.1"
    components = ("cuda_nvcc", "cuda_crt", "cuda_cudart", "cuda_cccl",
                  "cuda_cuobjdump", "cuda_nvdisasm", "libnvvm")
    records = []
    for component in components:
        if component not in data:
            raise RuntimeError(f"Required CUDA component missing: {component}")
        item = data[component]["linux-x86_64"]
        archive = root / "downloads" / Path(item["relative_path"]).name
        print(json.dumps({"download": component, "bytes": item["size"]}), flush=True)
        download(base + item["relative_path"], archive, item["sha256"])
        extracted = root / "local" / "components" / archive.name.removesuffix(".tar.xz")
        stamp = extracted / ".extracted"
        if not stamp.exists():
            extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter="data")
            stamp.write_text(item["sha256"] + "\n")
        package = next(p for p in extracted.iterdir() if p.is_dir())
        for name in ("bin", "include", "lib", "lib64", "nvvm"):
            if (package / name).is_dir():
                link_tree(package / name, cuda / name)
        records.append({"component": component, **item})
    if not (cuda / "lib64").exists():
        (cuda / "lib64").symlink_to("lib")
    deb_dir = root / "downloads" / "ubuntu-focal"
    deb_dir.mkdir(exist_ok=True)
    packages = ("bison", "flex", "libfl2", "libfl-dev", "zlib1g-dev", "xutils-dev")
    missing = [name for name in packages if not list(deb_dir.glob(f"{name}_*.deb"))]
    if missing:
        run_logged(["apt-get", "download", *missing], root, "download-build-tools", cwd=deb_dir)
    prefix = root / "local" / "build-tools"
    prefix.mkdir(exist_ok=True)
    for package in sorted(deb_dir.glob("*.deb")):
        stamp = prefix / (package.name + ".extracted")
        if not stamp.exists():
            run_logged(["dpkg-deb", "--extract", str(package), str(prefix)], root, "extract-build-tool")
            stamp.write_text(digest(package) + "\n")
    record = root / "logs" / f"dependencies-{timestamp()}.json"
    with record.open("x") as stream:
        json.dump({"cuda": records, "debs": {p.name: digest(p) for p in deb_dir.glob("*.deb")}}, stream, indent=2)
    print(json.dumps({"cuda_root": str(cuda), "build_tools": str(prefix), "manifest": str(record)}))


def build_environment(root):
    env = os.environ.copy()
    cuda = root / "local" / "cuda-13.1"
    prefix = root / "local" / "build-tools" / "usr"
    env.update({"CUDA_INSTALL_PATH": str(cuda), "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void", "PYTHONDONTWRITEBYTECODE": "1",
                "BISON_PKGDATADIR": str(prefix / "share/bison"),
                "PATH": f"{cuda / 'bin'}:{prefix / 'bin'}:/usr/bin:/bin",
                "CPATH": str(prefix / "include"),
                "LIBRARY_PATH": str(prefix / "lib/x86_64-linux-gnu"),
                "LD_LIBRARY_PATH": str(prefix / "lib/x86_64-linux-gnu"),
                "OMP_NUM_THREADS": "4",
                # Intersim inherits the compiler default otherwise (GCC 9 uses C++14).
                "EXTRA_FLAGS": "-std=c++17"})
    return env


def build(root, jobs):
    source = fetch(root)
    from patch_backend import apply
    patch = apply(source, root)
    env = build_environment(root)
    run_logged([str(root / "local/cuda-13.1/bin/nvcc"), "--version"], root,
               "nvcc-version", env=env)
    log = run_logged(["bash", "-c", 'source ./setup_environment && make -j "$1"',
                      "build", str(jobs)], root, "backend-build", cwd=source, env=env)
    libraries = list((source / "lib").glob("gcc-*/cuda-*/release/libcudart.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one simulator runtime, found {libraries}")
    record = {"experiment": "sim001", "kind": "backend_build", "status": "PASS", "backend": LOCK,
              "state": backend_state(source, libraries[0]), "applied_patch": patch,
              "build_log": str(log), "compiler_flags": env["EXTRA_FLAGS"],
              "dependencies": {p.name: digest(p) for p in sorted((root / "downloads").rglob("*"))
                               if p.is_file()},
              "tools": {name: subprocess.check_output([name, "--version"], env=env, text=True).splitlines()
                        for name in ("g++", "nvcc", "bison", "flex")}}
    receipt = root / "logs" / f"backend-build-{timestamp()}.json"
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"library": str(libraries[0]), "log": str(log), "receipt": str(receipt)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "fetch", "deps", "build"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run inside Linux/WSL")
    root = prepare(args.root)
    if args.action == "fetch":
        fetch(root)
    elif args.action == "deps":
        dependencies(root)
    elif args.action == "build":
        build(root, args.jobs)
    else:
        probe(root)


if __name__ == "__main__":
    main()
