"""Frozen original COMET model, original conformers and original preprocessing.

No surrogate, retraining, score rescaling or synthetic fallback is provided.
The sole supported checkpoint is the upstream lipid-only fold-0 deployment model.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

from .encoding import COMET_TYPES, GROUPS, Composition

COMET_COMMIT = "eee99d2055bad3b7d253c436b3eb7603550b37c5"
CHECKPOINT_ID = "1C1j2od4wwUISn9GJegHATe4cLtY2NwEU"
CHECKPOINT_SHA256 = "8de532194feead3662a33d62855801ebe55b02014eb05e3eedbeb27ee60f7335"
SCHEMA = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
CELLS = {"DC2.4": "in_house_lnp_DC24_luc", "B16-F10": "in_house_lnp_B16F10_luc"}
NAMES = {0: "ALC-0315", 1: "DOPE", 2: "Cholesterol", 3: "C14-PEG (COMET representation)",
         6: "SM-102", 10: "beta-Sitosterol", 11: "C18-PEG (COMET representation)", 13: "DSPC"}
SCORE_DESCRIPTION = "Frozen COMET contrastive delivery ranking score; higher is better; not measured efficacy or a probability"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_catalog(library):
    import lmdb
    records = json.loads((library / "in_house_lnp/infer.json").read_text())
    types = {}
    for row in records:
        for c in row["components"]:
            types.setdefault(c["smi"], set()).add(c["component_type"])
    env = lmdb.open(str(library / "mol.lmdb"), subdir=False, readonly=True, lock=False)
    catalog = {}
    try:
        with env.begin() as txn:
            for key, value in txn.cursor():
                # Trusted, pinned upstream dataset, containing RDKit conformers.
                mol = pickle.loads(value)
                mid = int(key)
                if len(types[mol["smi"]]) != 1:
                    raise ValueError(f"Ambiguous component type for mol_id {mid}")
                ctype = next(iter(types[mol["smi"]]))
                catalog[mid] = {"name": NAMES.get(mid, f"COMET-{ctype}-{mid}"), "mol_id": mid,
                                "component_type": ctype, "smiles": mol["smi"],
                                "conformer_count": len(mol["coordinates"])}
    finally:
        env.close()
    return catalog, records


class CometOracle:
    def __init__(self, config, condition, log_path):
        start = time.perf_counter()
        self.config, self.condition = config, condition
        if condition["cell"] not in CELLS:
            raise ValueError(f"Unsupported cell {condition['cell']!r}. This checkpoint supports {list(CELLS)}")
        self.target = CELLS[condition["cell"]]
        repo, checkpoint = Path(config["repo"]), Path(config["checkpoint"])
        if not (repo / "unimol/models/unimol.py").exists() or not checkpoint.is_file():
            raise FileNotFoundError("COMET source/checkpoint missing. Run: python scripts/setup_comet.py")
        self.checkpoint_hash = sha256(checkpoint)
        if self.checkpoint_hash != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint checksum mismatch; refusing unknown/corrupt pretrained weights")
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if revision != COMET_COMMIT:
            raise ValueError(f"COMET revision must be {COMET_COMMIT}; run setup_comet.py")
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise ValueError("Pinned COMET checkout has modified files; use an unmodified checkout")
        self.library = next((repo / "experiments/in_house_lnp_library").glob("*lmdb"))
        self.catalog, self.records = read_catalog(self.library)
        sys.path.insert(0, str(repo))
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                import torch
                from unimol.core import tasks
                from unimol.core.data import Dictionary
                self.torch = torch
                torch.set_num_threads(config["cpu_threads"])
                # weights_only=False is needed for the legacy Namespace checkpoint.
                # Its publisher-sourced SHA256 is verified BEFORE deserialization.
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                args = deepcopy(state["args"])
                args.data = str(repo / "experiments")
                args.task_name = str(self.library)
                args.full_dataset_task_schema_path = SCHEMA
                # Field added upstream after this lipid-only checkpoint was trained.
                args.include_excipients = False
                args.load_full_np_model = True
                args.cpu, args.fp16, args.bf16 = True, False, False
                args.mode, args.seed, args.num_workers = "infer", config["preprocessing_seed"], 0
                args.noise_augment_percent = False
                self.task = tasks.setup_task(args)
                self.model = self.task.build_model(args)
                self.model.load_state_dict(state["model"], strict=True)
                del state
                self.model.float().eval().requires_grad_(False)
                self.task.load_concat_dataset("infer")
                self.dataset = self.task.dataset("infer")
                self.dataset.set_epoch(1)
                # Upstream NP dataset does not propagate epoch to its molecule dataset.
                # Set it explicitly: otherwise hash((seed, None, id)) varies by process.
                self.dataset.mol_dataset.set_epoch(1)
                self.template = self.dataset[0]
                self.component_dictionary = self.task.component_type_dictionary["component_type"]["dictionary"]
                self.volume_dictionary = Dictionary.load_from_list(self.task.np_prop_schema["volumetric_ratio"]["dictionary"])
        finally:
            Path(log_path).write_text(log.getvalue())
        if condition["volumetric_ratio"] not in ("1:1", "3:1"):
            raise ValueError("volumetric_ratio must be '1:1' or '3:1' for this experiment")
        np_values = [r["NP_ratio"] for r in self.records]
        if not min(np_values) <= condition["np_ratio"] <= max(np_values):
            raise ValueError("N/P ratio lies outside the observed lipid-only data envelope")
        self.provenance = {
            "source": "real_pretrained_COMET", "score_description": SCORE_DESCRIPTION,
            "comet_commit": revision, "checkpoint_sha256": self.checkpoint_hash,
            "checkpoint_url": f"https://drive.google.com/file/d/{CHECKPOINT_ID}/view",
            "strict_state_dict": True, "frozen": True, "device": "cpu", "dtype": "float32",
            "condition": condition, "preprocessing_seed": args.seed, "preprocessing_epoch": 1,
            "conformers": "one deterministic upstream conformer per molecule; no TTA averaging",
            "mol_lmdb_sha256": sha256(self.library / "mol.lmdb"),
            "schema_sha256": sha256(repo / "experiments" / SCHEMA),
            "atom_dictionary_sha256": sha256(repo / "experiments/dict.txt"),
            "adapter_sha256": sha256(__file__),
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "rdkit")},
        }
        self.fingerprint = digest_json(self.provenance)
        self.load_seconds = time.perf_counter() - start
        self.evaluation_seconds = 0.0
        self.forward_seconds = 0.0
        self.compositions_evaluated = 0
        self.forward_batches = 0

    def validate_search(self, encoder):
        bounds = {}
        four = [r for r in self.records if len(r["components"]) == 4 and
                {c["component_type"] for c in r["components"]} == set(COMET_TYPES)]
        for i, (group, ctype, ids) in enumerate(zip(GROUPS, COMET_TYPES, encoder.candidates)):
            for mid in ids:
                if mid not in self.catalog or self.catalog[mid]["component_type"] != ctype:
                    raise ValueError(f"mol_id {mid} does not belong to {group} in COMET")
            values = [c["percent"] for r in four for c in r["components"] if c["component_type"] == ctype]
            low, high = min(values), max(values)
            chosen = [r[i] / encoder.denominator for r in encoder.ratios]
            if min(chosen) < low - 1e-9 or max(chosen) > high + 1e-9:
                raise ValueError(f"{group}: search range outside observed data envelope [{low}, {high}]")
            bounds[group] = {"observed_min": low, "observed_max": high,
                             "search_min": min(chosen), "search_max": max(chosen)}
        return {"source_records": len(self.records), "four_component_records": len(four),
                "marginal_fraction_bounds": bounds,
                "note": "Marginal bounds and known molecules do not establish joint in-domain validity or experimental efficacy."}

    def _sample(self, composition):
        t = self.torch
        sample = deepcopy(self.template)
        components = [{"mol_id": mid, "smi": self.catalog[mid]["smiles"], "component_type": ctype,
                       "mol": fraction, "percent": fraction}
                      for mid, fraction, ctype in zip(composition.lipid_ids, composition.fractions, COMET_TYPES)]
        sample["net_input.mol_ids"] = t.tensor(composition.lipid_ids, dtype=t.long)
        # Match upstream float64 fraction collation; GaussianLayer casts to float32.
        sample["net_input.percents"] = t.tensor(composition.fractions, dtype=t.float64)
        sample["net_input.component_types"] = t.tensor([self.component_dictionary.index(c) for c in COMET_TYPES])
        sample["net_input.components"] = components
        sample["net_input.lnp_ids"] = composition.key
        sample["net_input.NP_ratio"] = self.condition["np_ratio"]
        sample["net_input.volumetric_ratio"] = self.volume_dictionary.index(self.condition["volumetric_ratio"])
        sample["net_input.NP_ratio_mask"] = True
        sample["net_input.volumetric_ratio_mask"] = True
        # Labels are NOT passed into model.forward or used by the search objective.
        return sample

    def _forward(self, batch):
        batch.pop("target", None)
        start = time.perf_counter()
        with self.torch.inference_mode():
            scores = self.model(batch, np_classification_head_name=self.task.args.classification_head_name)[0][self.target]
        self.forward_seconds += time.perf_counter() - start
        self.forward_batches += 1
        if not self.torch.isfinite(scores).all():
            raise RuntimeError("COMET returned nonfinite scores; stopping without substituting scores")
        return scores.flatten().tolist()

    def predict(self, compositions):
        scores = []
        start = time.perf_counter()
        try:
            for offset in range(0, len(compositions), self.config["batch_size"]):
                chunk = compositions[offset:offset+self.config["batch_size"]]
                batch = self.dataset.collater([self._sample(c) for c in chunk])
                scores.extend(self._forward(batch))
                self.compositions_evaluated += len(chunk)
        finally:
            self.evaluation_seconds += time.perf_counter() - start
        return scores

    def preflight(self):
        """Compare the adapter with upstream collation on a real known formulation."""
        start = time.perf_counter()
        previous_evaluation_seconds = self.evaluation_seconds
        original = self.dataset.collater([self.dataset[0]])
        net = original["net_input"]
        ids = tuple(c["mol_id"] for c in net["components"][0])
        # Upstream record 0 is exactly 0.355 / 0.163 / 0.472 / 0.01.
        fractions = tuple(c["percent"] for c in net["components"][0])
        comp = Composition(ids, tuple(round(x * 1000) for x in fractions), 1000)
        # Compare at the configured process conditions for both paths.
        original["net_input"]["NP_ratio"][:] = self.condition["np_ratio"]
        original["net_input"]["volumetric_ratio"][:] = self.volume_dictionary.index(self.condition["volumetric_ratio"])
        upstream_score = self._forward(original)[0]
        adapter_score, repeat_score = self.predict([comp, comp])
        self.evaluation_seconds = previous_evaluation_seconds + time.perf_counter()-start
        self.compositions_evaluated += 1
        error = abs(adapter_score-upstream_score)
        if error > 1e-5 or abs(repeat_score-adapter_score) > 1e-6:
            raise RuntimeError(f"COMET preflight parity/repeatability failed (error={error})")
        return {"status": "ok", "source": "real_pretrained_COMET", "cell": self.condition["cell"],
                "upstream_score": upstream_score, "adapter_score": adapter_score,
                "absolute_error": error, "repeat_score": repeat_score,
                "seconds": time.perf_counter()-start, "composition": comp.as_dict(self.catalog)}

