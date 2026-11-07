"""Download pinned original COMET assets; preserve an existing checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.comet import CHECKPOINT_ID, CHECKPOINT_SHA256, COMET_COMMIT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "vendor/COMET")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/comet_lipid_fold0.pt")
    args = parser.parse_args()
    repo, checkpoint = args.repo.resolve(), args.checkpoint.resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/alvinchangw/COMET.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", COMET_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMET_COMMIT:
        raise RuntimeError(f"Existing checkout is {actual}; choose another --repo path for pinned {COMET_COMMIT}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        import gdown
        partial = checkpoint.with_suffix(".download")
        gdown.download(id=CHECKPOINT_ID, output=str(partial), quiet=False, use_cookies=False)
        if sha256(partial) != CHECKPOINT_SHA256:
            raise RuntimeError(f"Downloaded checkpoint checksum mismatch: {partial}")
        partial.replace(checkpoint)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Existing checkpoint checksum mismatch; no files were overwritten")
    report = {"comet_commit": actual, "checkpoint_sha256": CHECKPOINT_SHA256,
              "checkpoint_id": CHECKPOINT_ID, "checkpoint_bytes": checkpoint.stat().st_size,
              "repo": str(repo), "checkpoint": str(checkpoint)}
    (checkpoint.parent / "assets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

