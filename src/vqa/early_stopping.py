"""Noise-aware plateau monitoring; this is not a convergence certificate."""
from collections import deque
import math


DEFAULTS = {"enabled": False, "min_iterations": 30, "patience": 4,
            "window": 5, "min_delta": 0.02, "confidence_z": 1.0}


class PlateauStopper:
    def __init__(self, settings):
        self.settings = {**DEFAULTS, **settings}
        self.observations = deque(maxlen=self.settings["window"])
        self.best_mean = None
        self.best_error = None
        self.stale_windows = 0
        self.should_stop = False
        self.last_check = None

    def observe(self, iteration, mean, standard_error):
        if not self.settings["enabled"]:
            return False
        self.observations.append((mean, standard_error))
        window = self.settings["window"]
        if len(self.observations) < window or iteration % window:
            return False
        smoothed = math.fsum(m for m, _ in self.observations)/window
        # Monitoring shots use fresh seeds. The means are at nearby, changing
        # parameter vectors; this is a trend heuristic, not a fixed-state test.
        uncertainty = (math.sqrt(math.fsum(e*e for _, e in self.observations))/window
                       if all(e is not None for _, e in self.observations) else None)
        required = self.settings["min_delta"]
        if self.best_mean is not None and uncertainty is not None and self.best_error is not None:
            required += self.settings["confidence_z"]*math.hypot(uncertainty, self.best_error)
        improved = self.best_mean is None or smoothed > self.best_mean+required
        if improved:
            self.best_mean, self.best_error = smoothed, uncertainty
            self.stale_windows = 0
        elif iteration >= self.settings["min_iterations"]:
            self.stale_windows += 1
        # With a single shot, no standard error can be estimated: do not claim
        # a noise-aware plateau and fall back to the maximum iteration budget.
        self.should_stop = (uncertainty is not None and iteration >= self.settings["min_iterations"]
                            and self.stale_windows >= self.settings["patience"])
        self.last_check = {"iteration": iteration, "smoothed_mean": smoothed,
                           "standard_error": uncertainty, "required_improvement": required,
                           "improved": improved, "stale_windows": self.stale_windows,
                           "best_smoothed_mean": self.best_mean}
        return self.should_stop
