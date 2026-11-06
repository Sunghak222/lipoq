"""Bounded continuous ratio refinement with fixed molecular identities."""
from dataclasses import dataclass
from decimal import Decimal
import math
import time

import numpy as np
from scipy.optimize import minimize

from .encoding import GROUPS
from .results import write_json


@dataclass(frozen=True)
class ContinuousComposition:
    lipid_ids: tuple
    fractions: tuple

    def __post_init__(self):
        if (len(self.lipid_ids) != 4 or len(self.fractions) != 4 or
            any(type(i) is not int or i < 0 for i in self.lipid_ids) or
            any(not math.isfinite(x) or x <= 0 for x in self.fractions) or
            not math.isclose(math.fsum(self.fractions), 1.0, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("Require four lipids and positive finite fractions summing to one")

    @property
    def key(self):
        # 12 decimal places only identify numerically equivalent cache entries;
        # optimizer inputs are not snapped to the discrete search grid.
        values = [str(Decimal(str(x)).quantize(Decimal('0.000000000001')).normalize())
                  for x in self.fractions]
        return ','.join(map(str, self.lipid_ids)) + '|' + ','.join(values)

    def as_dict(self, catalog):
        return {"components": [{"group": group, "mol_id": mid, "name": catalog[mid]["name"],
                                "smiles": catalog[mid]["smiles"], "molar_fraction": float(x)}
                               for group, mid, x in zip(GROUPS, self.lipid_ids, self.fractions)],
                "fraction_sum": math.fsum(self.fractions), "ratio_mode": "continuous"}


class BoundedSimplex:
    """Map a unit cube onto four fractions with bounds and an exact residual.

    At each coordinate, reserve enough mass for every remaining component.
    This permits boundary solutions without sending infeasible points to COMET.
    """
    def __init__(self, bounds):
        self.low = np.array([bounds[g]['min'] for g in GROUPS], dtype=float)
        self.high = np.array([bounds[g]['max'] for g in GROUPS], dtype=float)
        if (not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)) or
            np.any(self.low <= 0) or np.any(self.high < self.low) or
            self.low.sum() > 1 + 1e-12 or self.high.sum() < 1 - 1e-12):
            raise ValueError('Infeasible continuous fraction bounds')

    def interval(self, i, remaining):
        low = max(self.low[i], remaining - self.high[i+1:].sum())
        high = min(self.high[i], remaining - self.low[i+1:].sum())
        return low, max(low, high)

    def decode(self, coordinates):
        if len(coordinates) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in coordinates):
            raise ValueError('Expected three finite coordinates in [0, 1]')
        fractions = []
        for i, coordinate in enumerate(coordinates):
            low, high = self.interval(i, 1 - math.fsum(fractions))
            fractions.append(float(low + coordinate * (high-low)))
        fractions.append(1 - math.fsum(fractions))
        self.validate(fractions)
        return tuple(fractions)

    def encode(self, fractions):
        self.validate(fractions)
        coordinates = []
        for i in range(3):
            low, high = self.interval(i, 1 - math.fsum(fractions[:i]))
            coordinates.append(float(np.clip((fractions[i]-low)/(high-low), 0, 1)) if high-low > 1e-14 else 0.5)
        return np.array(coordinates)

    def validate(self, fractions):
        f = np.asarray(fractions)
        if (f.shape != (4,) or not np.all(np.isfinite(f)) or
            np.any(f < self.low-1e-12) or np.any(f > self.high+1e-12) or
            not math.isclose(math.fsum(f), 1, abs_tol=1e-12, rel_tol=0)):
            raise ValueError('Composition outside continuous bounds/simplex')


def refine(evaluator, bounds, settings, directory):
    start = time.perf_counter()
    previous_seconds = evaluator.comet_seconds
    previous_count = len(evaluator.seen)
    previous_requests = evaluator.query_requests
    starting_best = evaluator.best
    space = BoundedSimplex(bounds)
    ranked = sorted(evaluator.compositions, key=lambda k: (-evaluator.scores[k], k))
    selected, identities = [], set()
    for key in ranked:
        composition = evaluator.compositions[key]
        if composition.lipid_ids not in identities:
            selected.append(composition)
            identities.add(composition.lipid_ids)
        if len(selected) == settings['top_k']:
            break
    solvers = []
    for index, composition in enumerate(selected):
        def objective(coordinates):
            candidate = ContinuousComposition(composition.lipid_ids, space.decode(coordinates))
            return -evaluator.score([candidate], stage='continuous')[0]

        result = minimize(objective, space.encode(composition.fractions), method='Powell',
                          bounds=[(0, 1)]*3,
                          options={'maxfev': settings['maxfev_per_start'],
                                   'xtol': settings['xtol'], 'ftol': settings['ftol']})
        solvers.append({'start_rank': index+1, 'lipid_ids': composition.lipid_ids,
                        'starting_fractions': composition.fractions,
                        'starting_score': evaluator.scores[composition.key],
                        'nfev': int(result.nfev), 'nit': int(result.nit),
                        'success': bool(result.success), 'message': str(result.message),
                        'returned_fractions': space.decode(result.x),
                        'returned_score': -float(result.fun)})
    summary = {'method': 'scipy.optimize.minimize/Powell', 'settings': settings,
               'bounds': {g: {k: bounds[g][k] for k in ('min', 'max')} for g in GROUPS},
               'discrete_best': starting_best, 'refined_best': evaluator.best,
               'improvement': evaluator.best['comet_score']-starting_best['comet_score'],
               'additional_unique_comet_evaluations': len(evaluator.seen)-previous_count,
               'objective_requests': evaluator.query_requests-previous_requests,
               'comet_seconds': evaluator.comet_seconds-previous_seconds,
               'seconds': time.perf_counter()-start, 'solvers': solvers}
    write_json(directory/'continuous.json', summary)
    return summary
