"""Bundle source, settings, and real experiment evidence in one submission notebook.

The large official COMET checkpoint is downloaded at runtime, never embedded.
The output keeps readable source text and base64-encoded compressed evidence in
collapsed setup cells. It can be rebuilt from itself after updating repository
code or refreshing the evidence snapshot.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pprint
import re
import textwrap
import tomllib

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TAG = "lipoq-single-file-setup"


def setup_cell(source):
    return nbformat.v4.new_code_cell(source, metadata={
        "tags": [TAG], "jupyter": {"source_hidden": True},
    })


def build(path):
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get("tags", [])]
    source_files = sorted((ROOT / "src").rglob("*.py"))
    source_files += [ROOT / "configs/vqa.yaml", ROOT / "scripts/setup_comet.py",
                     ROOT / "notebooks/data/manifest.json"]
    sources = {p.relative_to(ROOT).as_posix(): p.read_text() for p in source_files}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    evidence = (ROOT / "notebooks/data/experiment_snapshot.json.gz").read_bytes()
    evidence_hash = hashlib.sha256(evidence).hexdigest()
    manifest = json.loads(sources["notebooks/data/manifest.json"])
    assert manifest["sha256"] == evidence_hash

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    package_names = [re.split(r"[<>=!~\[]", spec)[0] for spec in project["dependencies"]]
    package_names += ["gdown", "nbformat", "nbclient", "pylatexenc"]
    pins = {name: importlib.metadata.version(name) for name in dict.fromkeys(package_names)}
    dependencies = setup_cell(textwrap.dedent('''\
        # Install only missing or different tested versions in this Python kernel.
        import importlib.metadata
        import importlib.util
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys

        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("Use a Python 3.12 kernel (Python 3.11 is also supported).")
        REQUIRED_VERSIONS = __PINS__
        needed = []
        for package, version in REQUIRED_VERSIONS.items():
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                needed.append(f"{package}=={version}")
        if needed:
            # Avoid silently retaining already-imported versions after an upgrade.
            loaded = set(sys.modules) & {"numpy", "torch", "scipy", "pandas", "qiskit", "rdkit"}
            if importlib.util.find_spec("pip"):
                command = [sys.executable, "-m", "pip", "install", "--quiet", *needed]
            elif shutil.which("uv"):
                command = ["uv", "pip", "install", "--python", sys.executable, *needed]
            else:
                raise RuntimeError("Install pip or uv in this kernel, then run this cell again.")
            subprocess.run(command, check=True)
            if loaded:
                raise RuntimeError("Dependencies installed. Restart the kernel and Run All.")
        print("Tested package versions are installed. No LipoQ repository checkout is required.")
        if not shutil.which("git"):
            raise RuntimeError("Install Git before downloading the pinned upstream COMET source.")
        ''').replace("__PINS__", pprint.pformat(pins, sort_dicts=False, width=100)))

    implementation = setup_cell(textwrap.dedent('''\
        # Embedded readable project source. Expand this cell to inspect all modules.
        # Materialize the exact source bytes so model provenance hashes stay valid.
        import hashlib
        import json

        EMBEDDED_SOURCES = __SOURCES__
        SOURCE_SHA256 = "__SOURCE_HASH__"
        assert hashlib.sha256(json.dumps(EMBEDDED_SOURCES, sort_keys=True).encode()).hexdigest() == SOURCE_SHA256
        ROOT = Path(os.environ.get("LIPOQ_WORKSPACE", str(Path.cwd() / "lipoq_workspace"))).expanduser().resolve()
        for relative, content in EMBEDDED_SOURCES.items():
            destination = ROOT / relative
            if not destination.resolve().is_relative_to(ROOT):
                raise ValueError("Embedded path escapes the notebook workspace")
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = content.encode()
            if destination.exists() and destination.read_bytes() != encoded:
                raise RuntimeError(f"Existing generated file differs: {relative}. Choose an empty LIPOQ_WORKSPACE.")
            if not destination.exists():
                destination.write_bytes(encoded)
        for name in ("common", "vqa", "random_search"):
            loaded_module = sys.modules.get(name)
            if loaded_module is not None and not Path(loaded_module.__file__).resolve().is_relative_to(ROOT):
                raise RuntimeError("Restart the kernel to use the embedded implementation.")
        sys.path.insert(0, str(ROOT / "src"))
        print(f"Restored {len(EMBEDDED_SOURCES)} embedded source/configuration files; SHA256 verified.")
        print("Generated workspace:", ROOT.name)
        ''').replace("__SOURCES__", pprint.pformat(sources, sort_dicts=True, width=110))
             .replace("__SOURCE_HASH__", source_hash))

    evidence_text = base64.b64encode(evidence).decode()
    encoded_lines = "\n".join(f'    "{line}"' for line in textwrap.wrap(evidence_text, 100))
    snapshot = setup_cell(textwrap.dedent('''\
        # Compact real experiment records, not a replacement prediction model.
        import base64
        EVIDENCE_BASE64 = (
        __EVIDENCE__
        )
        evidence_bytes = base64.b64decode(EVIDENCE_BASE64, validate=True)
        assert hashlib.sha256(evidence_bytes).hexdigest() == "__EVIDENCE_HASH__"
        evidence_path = ROOT / "notebooks/data/experiment_snapshot.json.gz"
        evidence_path.write_bytes(evidence_bytes)
        print(f"Restored {len(evidence_bytes):,} bytes of verified real experiment evidence.")
        ''').replace("__EVIDENCE__", encoded_lines).replace("__EVIDENCE_HASH__", evidence_hash))

    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("# LipoQ"):
            cell.source = """# LipoQ
### Black-box variational quantum search for lipid nanoparticle compositions
**IBM Quantum Qiskit Fall Fest — single-file submission notebook**

This `.ipynb` contains the project source, configuration, compact real experiment evidence, and executed tables and figures. **This is the only file to submit.** No LipoQ repository, ZIP, YAML, or result file needs to accompany it.

**To read:** open the saved notebook; no downloads or execution are needed. **To execute:** select a Python 3.12 kernel and Run All. Setup cells restore the embedded implementation and evidence, install tested dependencies when necessary, and download the pinned official COMET assets. The checkpoint is approximately **586 MiB**, downloaded separately at runtime; it is not included in the submission file. Initial execution therefore requires internet access and Git. The first download can take several minutes.

**Reading guide:** Sections 1–6 explain the formulation space and algorithm; Sections 7–8 execute a three-update demo and implementation checks; Sections 9–11 analyze completed experiments. The full multi-seed study is never launched automatically. Large setup cells are collapsed for readability and can be expanded to inspect the source.

**Evidence boundary:** planned seeds are 21, 22, and 23. Only completed archives are counted; unavailable seeds are listed explicitly. Scores are frozen COMET predictions, not measured biological efficacy."""
        elif cell.cell_type == "markdown" and cell.source.startswith("## 3."):
            cell.source = """## 3. Environment, configuration, and embedded evidence

The next three setup cells make this notebook self-contained with respect to **our implementation and experiment records**. They install tested packages if needed, write the embedded Python modules and configuration into `lipoq_workspace`, and verify the embedded evidence by SHA256. This generated directory is created automatically; it is not a submission dependency.

The pretrained COMET checkpoint and upstream source are external scientific assets. Section 4 obtains them from their official distribution and verifies the pinned revision and checksum. No score substitution is used if setup or inference fails.

The defaults work from any writable directory. Optional environment variables `LIPOQ_WORKSPACE`, `LIPOQ_COMET_REPO`, and `LIPOQ_COMET_CHECKPOINT` allow a different generated directory or reuse of already-downloaded official assets. A compatible environment and verified cached assets allow offline reruns. The live demo and archived results remain separate."""
        elif cell.cell_type == "code" and cell.source.startswith("from pathlib import Path\nimport contextlib"):
            cell.source = cell.source.replace(
                "from common.results import create_run_dir, write_json",
                "from common.results import artifact_path, create_run_dir, write_json")
            start = cell.source.find("ROOT = next(")
            if start >= 0:
                end = cell.source.index('sys.path.insert(0, str(ROOT / "src"))', start)
                end += len('sys.path.insert(0, str(ROOT / "src"))')
                cell.source = cell.source[:start] + '# ROOT and imports are prepared by the embedded setup cells above.' + cell.source[end:]
            marker = 'cfg = load_config(ROOT / "configs/vqa.yaml")'
            if 'LIPOQ_COMET_REPO' not in cell.source:
                cell.source = cell.source.replace(marker, marker + '''
cfg["comet"]["repo"] = str(Path(os.environ.get("LIPOQ_COMET_REPO", cfg["comet"]["repo"])).expanduser().resolve())
cfg["comet"]["checkpoint"] = str(Path(os.environ.get("LIPOQ_COMET_CHECKPOINT", cfg["comet"]["checkpoint"])).expanduser().resolve())
assert Path(sys.modules["common.comet"].__file__).resolve().is_relative_to(ROOT / "src")
assert Path(sys.modules["vqa.optimizer"].__file__).resolve().is_relative_to(ROOT / "src")''')
        elif cell.cell_type == "code" and cell.source.startswith('runtime_dir = create_run_dir('):
            if 'scripts/setup_comet.py' not in cell.source:
                marker = 'setup_log = io.StringIO()'
                cell.source = cell.source.replace(marker, '''print("Preparing official COMET assets (first download: about 586 MiB for the checkpoint).")
asset_setup = subprocess.run([
    sys.executable, str(ROOT / "scripts/setup_comet.py"),
    "--repo", cfg["comet"]["repo"], "--checkpoint", cfg["comet"]["checkpoint"],
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(runtime_dir / "asset_setup.log").write_text(asset_setup.stdout)
if asset_setup.returncode:
    raise RuntimeError("Official COMET asset setup failed:\\n" + asset_setup.stdout[-2000:])
print("Pinned COMET source and checkpoint verified.")
''' + marker)
        elif cell.cell_type == "code" and cell.source.startswith("demo_cfg = deepcopy(cfg)"):
            cell.source = cell.source.replace('pd.read_csv(demo_dir / "history.csv")',
                                              'pd.read_csv(artifact_path(demo_dir, "history.csv"))')
        elif cell.cell_type == "markdown" and "**Reproduction**" in cell.source:
            start, end = cell.source.index("**Reproduction**"), cell.source.index("**References and attribution**")
            cell.source = cell.source[:start] + """**Reproduction**

Submit this `.ipynb` alone. To inspect the results, open the saved notebook without running it. To reproduce the executable example, use a Python 3.12 kernel and **Run All** from any writable directory. Initial setup needs Git, internet access, and space for the external COMET checkpoint and dependencies. The generated `lipoq_workspace` directory contains the embedded source, configuration, evidence, downloaded assets, and new demo outputs. Existing archived main results are analyzed from the embedded evidence.

The source payload retains the common, VQA, and random-search modules in readable text. The evidence payload is a compressed copy of actual run records with source-file hashes. Both are verified before use. The original COMET adapter bytes are preserved, keeping the model fingerprint comparable to the archived experiment.

To run additional main experiments, the generated `configs/vqa.yaml` and `vqa`/`random_search` modules expose the same configuration and command interfaces as the repository. These longer experiments are deliberately separate from Run All. Changing the search space or experimental protocol requires fresh compatible evidence; it does not change the saved results retroactively.

""" + cell.source[end:]

    insertion = next(i for i, c in enumerate(notebook.cells)
                     if c.cell_type == "markdown" and c.source.startswith("## 3.")) + 1
    notebook.cells[insertion:insertion] = [dependencies, implementation, snapshot]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook cell", "exec")
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop("lipoq_execution", None)
    notebook.metadata["kernelspec"] = {
        "name": "python3", "language": "python", "display_name": "Python 3 (ipykernel)",
    }
    notebook.metadata["lipoq_single_file"] = {
        "source_sha256": source_hash, "evidence_sha256": evidence_hash,
        "embedded_files": len(sources), "external_checkpoint_bytes": 614071415,
        "submission_limit_bytes": 20_000_000,
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    if path.stat().st_size >= 20_000_000:
        raise ValueError("Submission notebook exceeds 20 MB")
    print(json.dumps({"notebook": path.name, "bytes_before_execution": path.stat().st_size,
                      "embedded_files": len(sources)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/LipoQ_Submission.ipynb")
    build(parser.parse_args().notebook.resolve())
