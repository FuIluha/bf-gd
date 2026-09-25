"""Framework-neutral contracts for step-wise LDPC visualisation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    kind: str
    default: object
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None

    def parse(self, raw):
        if raw is None:
            raise ValueError(f"Параметр {self.label} не задан")
        if self.kind == "float":
            value = float(raw)
        elif self.kind == "int":
            numeric = float(raw)
            if not numeric.is_integer():
                raise ValueError(f"{self.label} должен быть целым")
            value = int(numeric)
        elif self.kind == "float_list":
            if isinstance(raw, str):
                value = [float(part.strip()) for part in raw.split(",")]
            else:
                value = [float(part) for part in raw]
            if not value:
                raise ValueError(f"{self.label} не должен быть пустым")
        else:
            raise ValueError(f"Неизвестный тип параметра: {self.kind}")
        values = value if isinstance(value, list) else [value]
        if not all(np.isfinite(item) for item in values):
            raise ValueError(f"{self.label} должен содержать конечные числа")
        if self.minimum is not None and any(item < self.minimum for item in values):
            raise ValueError(f"{self.label} меньше {self.minimum:g}")
        if self.maximum is not None and any(item > self.maximum for item in values):
            raise ValueError(f"{self.label} больше {self.maximum:g}")
        return value


@dataclass(frozen=True)
class ObservableSpec:
    key: str
    label: str
    zero_line: bool = False


@dataclass(frozen=True)
class CategorySpec:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class CategoryGroupSpec:
    """A non-overlapping category partition used for shared normalisation."""

    key: str
    label: str
    categories: Tuple[str, ...]


@dataclass(frozen=True)
class DecoderSpec:
    key: str
    title: str
    parameters: Tuple[ParameterSpec, ...]
    observables: Tuple[ObservableSpec, ...]
    categories: Tuple[CategorySpec, ...]
    category_groups: Tuple[CategoryGroupSpec, ...]

    def validate(self):
        if not self.key or not self.title or not self.observables or not self.categories:
            raise ValueError("Decoder description is incomplete")
        for group in (self.parameters, self.observables, self.categories):
            keys = [item.key for item in group]
            if len(keys) != len(set(keys)) or any(not key for key in keys):
                raise ValueError("Decoder description contains duplicate/empty keys")
        for parameter in self.parameters:
            parameter.parse(parameter.default)
        category_keys = {item.key for item in self.categories}
        group_keys = [item.key for item in self.category_groups]
        if not group_keys:
            raise ValueError("Decoder description requires at least one category group")
        if len(group_keys) != len(set(group_keys)) or any(not key for key in group_keys):
            raise ValueError("Decoder description contains duplicate/empty category group keys")
        grouped = []
        for group in self.category_groups:
            if not group.label or not group.categories:
                raise ValueError("Category group must have a label and categories")
            if len(group.categories) != len(set(group.categories)):
                raise ValueError(f"Category group {group.key} contains duplicates")
            if not set(group.categories).issubset(category_keys):
                raise ValueError(f"Category group {group.key} contains unknown categories")
            grouped.extend(group.categories)
        if set(grouped) != category_keys or len(grouped) != len(category_keys):
            raise ValueError("Every category must belong to exactly one category group")

    def groups(self):
        """Return the decoder-defined non-overlapping category partitions."""
        return self.category_groups

    def group(self, key):
        for group in self.groups():
            if group.key == key:
                return group
        raise ValueError(f"Unknown category group: {key}")

    def parse_parameters(self, values):
        return {
            item.key: item.parse(values[item.key])
            for item in self.parameters
        }

    def defaults(self):
        return self.parse_parameters({item.key: item.default for item in self.parameters})


@dataclass(frozen=True)
class FrameBatch:
    received: np.ndarray
    transmitted_symbols: np.ndarray

    def validate(self):
        if self.received.ndim != 2 or 0 in self.received.shape:
            raise ValueError("received must have non-empty (frames, bits) shape")
        if self.transmitted_symbols.shape != self.received.shape:
            raise ValueError("received and transmitted shapes must match")
        if not np.issubdtype(self.received.dtype, np.floating):
            raise ValueError("received must be floating-point")
        if not np.all(np.isfinite(self.received)):
            raise ValueError("received must be finite")
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
    fields: Mapping[str, np.ndarray]
    active: np.ndarray


@dataclass(frozen=True)
class StepResult:
    fields: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class CategorySamples:
    """One category; observations may also occur in other categories."""

    frame_indices: np.ndarray
    entity_indices: np.ndarray
    values: Mapping[str, np.ndarray]


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
    categories: Mapping[str, CategorySamples]
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
    parameters: Mapping[str, object]
    before: Metrics
    after: Metrics

    def to_dict(self):
        return {
            "from_iteration": self.from_iteration,
            "to_iteration": self.to_iteration,
            "parameters": dict(self.parameters),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class HistogramResult:
    edges: np.ndarray
    counts: Mapping[str, np.ndarray]
    values: Mapping[str, np.ndarray]
    method: str
    normalization_count: Optional[int] = None


@dataclass(frozen=True)
class ComparisonHistory:
    name: str
    algorithm: str
    iterations: np.ndarray
    ber: np.ndarray
    fer: np.ndarray
    metadata: dict
