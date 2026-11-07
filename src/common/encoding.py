"""Total bitstring decoder over four lipid identities and a bounded simplex grid."""
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from math import prod

GROUPS = ("ionizable", "phospholipid", "sterol", "peg")
# Preserve the original COMET type identifiers; HL maps to phospholipid here.
COMET_TYPES = ("IL", "HL", "CH", "PEG")


@dataclass(frozen=True)
class Composition:
    lipid_ids: tuple[int, int, int, int]
    ticks: tuple[int, int, int, int]
    denominator: int

    def __post_init__(self):
        if (len(self.lipid_ids) != 4 or len(self.ticks) != 4 or
            type(self.denominator) is not int or self.denominator <= 0 or
            any(type(v) is not int or v < 0 for v in self.lipid_ids) or
            any(type(v) is not int or v <= 0 for v in self.ticks) or
            sum(self.ticks) != self.denominator):
            raise ValueError("Composition must contain four lipid IDs and positive fractions summing exactly to 1")

    @property
    def fractions(self):
        return tuple(v / self.denominator for v in self.ticks)

    @property
    def key(self):
        # Fractions, rather than bitstrings, eliminate alias evaluations.
        fractions = [str(Decimal(v) / self.denominator) for v in self.ticks]
        return ",".join(map(str, self.lipid_ids)) + "|" + ",".join(fractions)

    def as_dict(self, catalog):
        return {
            "components": [
                {"group": group, "mol_id": mid, "name": catalog[mid]["name"],
                 "smiles": catalog[mid]["smiles"], "molar_fraction": fraction}
                for group, mid, fraction in zip(GROUPS, self.lipid_ids, self.fractions)
            ],
            "fraction_ticks": list(self.ticks), "denominator": self.denominator,
            "fraction_sum": sum(self.ticks) / self.denominator,
        }


class CompositionEncoder:
    def __init__(self, config):
        self.candidates = tuple(tuple(config["candidates"][g]) for g in GROUPS)
        for group, ids in zip(GROUPS, self.candidates):
            if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError(f"{group}: require nonempty, unique nonnegative integer mol_ids")
        self.denominator = config["ratios"]["denominator"]
        if type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("ratios.denominator must be a positive integer")
        axes = []
        for group in GROUPS:
            spec = config["ratios"]["bounds"][group]
            vals = []
            for key in ("min", "max", "step"):
                scaled = Decimal(str(spec[key])) * self.denominator
                if not scaled.is_finite() or scaled != scaled.to_integral_value():
                    raise ValueError(f"{group}.{key} must lie on the denominator grid")
                vals.append(int(scaled))
            low, high, step = vals
            if not 0 < low <= high <= self.denominator or step <= 0 or (high-low) % step:
                raise ValueError(f"{group}: invalid bounds or step (all components must be present)")
            axes.append(range(low, high + 1, step))
        # Enumerate three axes; determine the fourth exactly in integer arithmetic.
        if prod(len(axis) for axis in axes[:3]) > 2_000_000:
            raise ValueError("Ratio grid too large; increase steps (max 2 million grid checks)")
        self.ratios = tuple((a, b, c, self.denominator-a-b-c)
                            for a, b, c in product(*axes[:3])
                            if self.denominator-a-b-c in axes[3])
        if not self.ratios:
            raise ValueError("No feasible molar fractions sum to 1 within the configured grid")
        sizes = tuple(len(ids) for ids in self.candidates) + (len(self.ratios),)
        self.widths = tuple((size - 1).bit_length() for size in sizes)
        self.num_qubits = sum(self.widths)
        self.cardinality = prod(sizes)

    def decode(self, bitstring):
        # Qiskit prints q[n-1] ... q[0]; low bits select the ionizable lipid.
        if len(bitstring) != self.num_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError(f"Expected exactly {self.num_qubits} binary digits")
        remaining = int(bitstring or "0", 2)
        indices = []
        for width in self.widths:
            indices.append(remaining & ((1 << width) - 1))
            remaining >>= width
        ids = tuple(options[i % len(options)] for options, i in zip(self.candidates, indices))
        return Composition(ids, self.ratios[indices[-1] % len(self.ratios)], self.denominator)

    def composition_at(self, index):
        """Index unique formulations uniformly, independently of bitstring aliases."""
        if not 0 <= index < self.cardinality:
            raise IndexError(index)
        index, ratio_index = divmod(int(index), len(self.ratios))
        ids = []
        for options in reversed(self.candidates):
            index, lipid_index = divmod(index, len(options))
            ids.append(options[lipid_index])
        return Composition(tuple(reversed(ids)), self.ratios[ratio_index], self.denominator)

    def compositions(self):
        return (self.composition_at(i) for i in range(self.cardinality))

    def metadata(self):
        return {"num_qubits": self.num_qubits, "field_widths_lsb_first": dict(zip((*GROUPS, "ratio"), self.widths)),
                "feasible_ratio_count": len(self.ratios), "unique_search_space_size": self.cardinality,
                "ratio_ticks": self.ratios, "denominator": self.denominator,
                "unused_codes": "modulo each field; non-power-of-two fields have alias bias"}

