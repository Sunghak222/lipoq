from collections import Counter
import pytest
from lnp_vqa.encoding import Composition, CompositionEncoder, GROUPS


def test_every_bitstring_valid_and_bijective_in_default_space(cfg):
    encoder = CompositionEncoder(cfg["search"])
    assert (encoder.num_qubits, len(encoder.ratios), encoder.cardinality) == (7, 8, 128)
    compositions = [encoder.decode(f"{i:07b}") for i in range(128)]
    assert len(set(c.key for c in compositions)) == 128
    for c in compositions:
        assert sum(c.ticks) == c.denominator
        for i, group in enumerate(GROUPS):
            assert c.lipid_ids[i] in cfg["search"]["candidates"][group]
            b = cfg["search"]["ratios"]["bounds"][group]
            assert b["min"] <= c.fractions[i] <= b["max"]


def test_qiskit_endianness_and_zero_width_singletons(cfg):
    e = CompositionEncoder(cfg["search"])
    assert e.decode("0000001").lipid_ids == (6, 1, 2, 3)
    assert e.decode("0000010").lipid_ids == (0, 13, 2, 3)
    assert e.decode("0010000").ticks != e.decode("0000000").ticks
    cfg["search"]["candidates"]["ionizable"] = [0]
    e = CompositionEncoder(cfg["search"])
    assert e.decode("000001").lipid_ids == (0, 13, 2, 3)


def test_non_power_of_two_aliases_are_valid_and_documented(cfg):
    cfg["search"]["candidates"]["ionizable"] = [0, 4, 6]
    e = CompositionEncoder(cfg["search"])
    counts = Counter(e.decode(f"{i:0{e.num_qubits}b}").key for i in range(2**e.num_qubits))
    assert len(counts) == e.cardinality == 192
    assert set(counts.values()) == {1, 2}


@pytest.mark.parametrize("problem", ["infeasible", "duplicate", "off_grid", "zero"])
def test_invalid_search_rejected(cfg, problem):
    s = cfg["search"]
    if problem == "infeasible":
        s["ratios"]["bounds"]["ionizable"] = dict(min=0.9, max=0.9, step=0.1)
    elif problem == "duplicate":
        s["candidates"]["helper"] = [1, 1]
    elif problem == "off_grid":
        s["ratios"]["bounds"]["peg"]["min"] = 0.0101
    else:
        s["ratios"]["bounds"]["peg"]["min"] = 0
    with pytest.raises(ValueError):
        CompositionEncoder(s)


def test_invalid_composition_and_bitstrings(cfg):
    with pytest.raises(ValueError):
        Composition((0, 1, 2, 3), (1, 2, 3, 4), 100)
    e = CompositionEncoder(cfg["search"])
    for bad in ("010", "000000x", "00000000"):
        with pytest.raises(ValueError):
            e.decode(bad)

