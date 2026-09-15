"""Domain models for the interactive decision-energy explorer."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np


class Algorithm(str, Enum):
    FTGDBF = "ftgdbf"
    PMGDBF = "pmgdbf"

    @property
    def title(self):
        if self is Algorithm.FTGDBF:
            return "FTGDBF"
        return "PMGDBF"


@dataclass(frozen=True)
class FtgdbfParameters:
    alpha: float

    def validate(self):
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")

    def to_dict(self):
        return {"alpha": float(self.alpha)}


@dataclass(frozen=True)
class PmgdbfParameters:
    delta: float
    alpha: float
    p: float
    rho: Tuple[float, ...]
    L: int

    def validate(self):
        if not np.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(self.p) or not 0 <= self.p <= 1:
            raise ValueError("p must be finite and in [0, 1]")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if self.L >= np.iinfo(np.int16).max:
            raise ValueError("L is too large for the stored momentum age")
        if len(self.rho) != self.L:
            raise ValueError("the number of rho values must equal L")
        if not all(np.isfinite(value) for value in self.rho):
            raise ValueError("all rho values must be finite")

    def to_dict(self):
        return {
            "delta": float(self.delta),
            "alpha": float(self.alpha),
            "p": float(self.p),
            "rho": [float(value) for value in self.rho],
            "L": int(self.L),
        }


DecoderParameters = Union[FtgdbfParameters, PmgdbfParameters]


@dataclass(frozen=True)
class FrameBatch:
    received: np.ndarray
    transmitted_symbols: np.ndarray

    def validate(self):
        if self.received.ndim != 2:
            raise ValueError("received must have shape (frames, block_length)")
        if self.received.shape[0] == 0 or self.received.shape[1] == 0:
            raise ValueError("frame batch must not be empty")
        if not np.issubdtype(self.received.dtype, np.floating):
            raise ValueError("received symbols must use a floating-point dtype")
        if not np.all(np.isfinite(self.received)):
            raise ValueError("received symbols must be finite")
        if self.transmitted_symbols.shape != self.received.shape:
            raise ValueError("received and transmitted_symbols shapes must match")
        if not np.all(np.isin(self.transmitted_symbols, (-1, 1))):
            raise ValueError("transmitted symbols must be -1 or +1")

    @property
    def frames(self):
        return self.received.shape[0]

    @property
    def block_length(self):
        return self.received.shape[1]


@dataclass(frozen=True)
class DecoderState:
    x: np.ndarray
    active: np.ndarray
    ages: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Metrics:
    ber: float
    fer: float
    bit_errors: int
    frame_errors: int

    def to_dict(self):
        return {
            "ber": float(self.ber),
            "fer": float(self.fer),
            "bit_errors": int(self.bit_errors),
            "frame_errors": int(self.frame_errors),
        }


@dataclass(frozen=True)
class StepObservation:
    energy: np.ndarray
    threshold: np.ndarray
    margin: np.ndarray
    should_flip: np.ndarray
    flip_mask: np.ndarray
    correct_action: np.ndarray
    decision_frames: np.ndarray
    before: Metrics
    after: Metrics


@dataclass(frozen=True)
class Snapshot:
    iteration: int
    state: DecoderState
    metrics: Metrics


@dataclass(frozen=True)
class TransitionRecord:
    from_iteration: int
    to_iteration: int
    parameters: DecoderParameters
    before: Metrics
    after: Metrics

    def to_dict(self):
        return {
            "from_iteration": self.from_iteration,
            "to_iteration": self.to_iteration,
            "parameters": self.parameters.to_dict(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class HistogramResult:
    edges: np.ndarray
    correct_counts: np.ndarray
    incorrect_counts: np.ndarray
    correct_values: np.ndarray
    incorrect_values: np.ndarray
    method: str


@dataclass(frozen=True)
class ComparisonHistory:
    name: str
    algorithm: str
    iterations: np.ndarray
    ber: np.ndarray
    fer: np.ndarray
    metadata: dict


def parameters_from_dict(algorithm, values):
    algorithm = Algorithm(algorithm)
    if algorithm is Algorithm.FTGDBF:
        result = FtgdbfParameters(alpha=float(values["alpha"]))
    else:
        result = PmgdbfParameters(
            delta=float(values["delta"]),
            alpha=float(values["alpha"]),
            p=float(values["p"]),
            rho=tuple(float(value) for value in values["rho"]),
            L=int(values["L"]),
        )
    result.validate()
    return result


def parse_rho(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("rho must not be empty")
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError("rho must be a comma-separated list of numbers") from exc
    if not result:
        raise ValueError("rho must not be empty")
    return result


def parameter_history_rows(transitions):
    rows = []
    for transition in transitions:
        params = transition.parameters.to_dict()
        rows.append({
            "step": f"{transition.from_iteration} → {transition.to_iteration}",
            "parameters": ", ".join(
                f"{key}={value}" for key, value in params.items()
            ),
            "ber": transition.after.ber,
            "fer": transition.after.fer,
        })
    return rows
