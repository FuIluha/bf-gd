"""Shared Tanner graph and decoder-independent batch stepping."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .models import CategorySamples, DecoderState, Metrics, StepObservation


@dataclass(frozen=True)
class TannerGraph:
    block_length: int
    n_checks: int
    edge_cn: np.ndarray
    edge_vn: np.ndarray
    check_offsets: np.ndarray

    @classmethod
    def from_pcm(cls, pcm):
        pcm = np.asarray(pcm, dtype=np.uint8)
        edge_cn, edge_vn = np.nonzero(pcm)
        edge_cn = np.ascontiguousarray(edge_cn, dtype=np.int32)
        edge_vn = np.ascontiguousarray(edge_vn, dtype=np.int32)
        degrees = np.bincount(edge_cn, minlength=pcm.shape[0])
        offsets = np.ascontiguousarray(
            np.concatenate(([0], np.cumsum(degrees))), dtype=np.int32,
        )
        graph = cls(pcm.shape[1], pcm.shape[0], edge_cn, edge_vn, offsets)
        graph.validate()
        return graph

    def validate(self):
        if self.block_length <= 0 or self.n_checks <= 0:
            raise ValueError("Tanner graph dimensions must be positive")
        if self.edge_cn.ndim != 1 or self.edge_vn.ndim != 1:
            raise ValueError("Tanner edge arrays must be one-dimensional")
        if self.edge_cn.size == 0 or self.edge_cn.size != self.edge_vn.size:
            raise ValueError("Tanner edge arrays have invalid lengths")
        if self.check_offsets.shape != (self.n_checks + 1,):
            raise ValueError("Tanner check offsets have an invalid shape")
        if self.check_offsets[0] != 0 or self.check_offsets[-1] != self.edge_cn.size:
            raise ValueError("Tanner offsets do not cover all edges")
        if np.any(np.diff(self.check_offsets) <= 0):
            raise ValueError("Degree-zero checks are not supported")
        if np.any(self.edge_vn < 0) or np.any(self.edge_vn >= self.block_length):
            raise ValueError("Tanner graph contains invalid variable indices")
        expected = np.repeat(np.arange(self.n_checks), np.diff(self.check_offsets))
        if not np.array_equal(self.edge_cn, expected):
            raise ValueError("Tanner edges must be grouped by check")
        if np.unique(self.edge_vn).size != self.block_length:
            raise ValueError("Degree-zero variables are not supported")

    def check_products(self, signs):
        return np.multiply.reduceat(
            signs[:, self.edge_vn], self.check_offsets[:-1], axis=1,
        ).astype(np.int8, copy=False)

    def satisfied(self, signs):
        return np.all(self.check_products(signs) == 1, axis=1)


def calculate_metrics(signs, transmitted):
    errors = signs != transmitted
    bit_errors = int(np.count_nonzero(errors))
    frame_errors = int(np.count_nonzero(np.any(errors, axis=1)))
    return Metrics(
        ber=bit_errors / errors.size,
        fer=frame_errors / errors.shape[0],
        bit_errors=bit_errors,
        frame_errors=frame_errors,
    )


class StepRandom:
    """Shared deterministic random arrays independent of worker partitioning."""

    def __init__(self, seed, iteration, frames):
        self.seed = int(seed)
        self.iteration = int(iteration)
        self.frames = int(frames)
        self._cache = {}
        self._lock = Lock()

    def uniform(self, frame_slice, tail_shape, stream=0):
        tail_shape = tuple(tail_shape)
        key = (int(stream), tail_shape)
        with self._lock:
            if key not in self._cache:
                rng = np.random.Generator(np.random.Philox(
                    np.random.SeedSequence([self.seed, self.iteration, int(stream)]),
                ))
                self._cache[key] = rng.random((self.frames, *tail_shape))
            return self._cache[key][frame_slice]

    def for_frame(self, frame):
        return _FrameRandom(self, frame)


class _FrameRandom:
    """RNG-shaped view of a precomputed, partition-independent frame stream."""

    def __init__(self, source, frame):
        self.source, self.frame = source, frame

    def random(self, size=None):
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        return self.source.uniform(self.frame, shape)


class BatchDecoderEngine:
    """Run any VisualizableDecoderBase subclass over a fixed frame batch."""

    def __init__(self, graph, batch, seed, workers=1):
        graph.validate()
        batch.validate()
        if graph.block_length != batch.block_length:
            raise ValueError("Frame batch and Tanner graph sizes differ")
        self.graph = graph
        self.batch = batch
        self.seed = int(seed)
        self.workers = max(1, int(workers))

    def initial_state(self, decoder, parameters):
        parts = [decoder.initial_state(received, parameters) for received in self.batch.received]
        if any(set(part) != set(parts[0]) for part in parts):
            raise ValueError("Decoder returned inconsistent initial state fields")
        fields = {key: np.stack([part[key] for part in parts]) for key in parts[0]}
        _validate_fields(fields, self.batch.frames)
        return DecoderState(fields=fields, active=np.ones(self.batch.frames, dtype=bool))

    def step(self, decoder, state, parameters, iteration):
        decoder.validate_parameters(parameters)
        _validate_fields(state.fields, self.batch.frames)
        if state.active.shape != (self.batch.frames,):
            raise ValueError("Invalid active-frame mask")
        decoded = np.asarray([
            decoder.is_decoded({key: value[frame] for key, value in state.fields.items()})
            for frame in range(self.batch.frames)
        ], dtype=bool)
        decision_frames = state.active & ~decoded
        active_indices = np.flatnonzero(decision_frames)
        random = StepRandom(self.seed, iteration, self.batch.frames)
        fields = {key: value.copy() for key, value in state.fields.items()}

        def work(frame):
            word_state = {key: value[frame] for key, value in state.fields.items()}
            return decoder.step_state(
                word_state, self.batch.received[frame], parameters,
                iteration, random.for_frame(int(frame)),
            )

        if self.workers == 1 or active_indices.size < 2:
            parts = [work(frame) for frame in active_indices]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, active_indices.size)) as executor:
                parts = list(executor.map(work, active_indices))
        diagnostics = {}
        if parts:
            expected_names = set(fields)
            diagnostic_names = set(parts[0].diagnostics)
            for frame, part in zip(active_indices, parts):
                if set(part.fields) != expected_names or set(part.diagnostics) != diagnostic_names:
                    raise ValueError("Decoder returned inconsistent step fields")
                for key, value in part.fields.items():
                    if np.asarray(value).shape != fields[key][frame].shape:
                        raise ValueError(f"State field {key} changed shape")
                    fields[key][frame] = value
            diagnostics = {
                key: np.stack([part.diagnostics[key] for part in parts])
                for key in diagnostic_names
            }
        decoded_after = np.asarray([
            decoder.is_decoded({key: value[frame] for key, value in fields.items()})
            for frame in active_indices
        ], dtype=bool)
        next_active = np.zeros(self.batch.frames, dtype=bool)
        next_active[active_indices] = ~decoded_after
        next_state = DecoderState(fields=fields, active=next_active)
        before = self._metrics(decoder, state.fields)
        after = self._metrics(decoder, fields)
        categories = {}
        if parts:
            local_categories = decoder.classify(
                {key: value[active_indices] for key, value in state.fields.items()},
                {key: value[active_indices] for key, value in fields.items()},
                diagnostics, np.ones(active_indices.size, dtype=bool),
                self.batch.transmitted_symbols[active_indices],
            )
            if not isinstance(local_categories, dict):
                raise ValueError("classify must return a category dictionary")
            for key, samples in local_categories.items():
                if not isinstance(samples, CategorySamples):
                    raise ValueError(f"Category {key} must contain CategorySamples")
                if np.any(samples.frame_indices < 0) or np.any(samples.frame_indices >= active_indices.size):
                    raise ValueError(f"Category {key} contains invalid local frame indices")
            categories = {
                key: CategorySamples(
                    active_indices[samples.frame_indices], samples.entity_indices,
                    samples.values,
                ) for key, samples in local_categories.items()
            }
        _validate_categories(categories, decoder.describe(), decision_frames)
        return next_state, StepObservation(categories, decision_frames, before, after)

    def _metrics(self, decoder, fields):
        signs = np.asarray(decoder.hard_decision(fields))
        if signs.shape != self.batch.received.shape or not np.all(np.isin(signs, (-1, 1))):
            raise ValueError("hard_decision must return one -1/+1 sign per bit")
        return calculate_metrics(signs, self.batch.transmitted_symbols)

def _validate_fields(fields, frames):
    if not fields or not all(isinstance(key, str) and key for key in fields):
        raise ValueError("Decoder state requires named arrays")
    for key, value in fields.items():
        if not isinstance(value, np.ndarray) or value.ndim < 1 or value.shape[0] != frames:
            raise ValueError(f"State field {key} must be a frame-indexed NumPy array")
        if value.dtype.hasobject:
            raise ValueError("Object arrays cannot be stored in decoder state")


def _validate_categories(categories, spec, decision_frames):
    allowed = {item.key for item in spec.categories}
    observables = {item.key for item in spec.observables}
    if not isinstance(categories, dict) or not set(categories).issubset(allowed):
        raise ValueError("classify returned an unknown category")
    for key, samples in categories.items():
        if not isinstance(samples, CategorySamples):
            raise ValueError(f"Category {key} must contain CategorySamples")
        count = samples.frame_indices.size
        if samples.entity_indices.shape != (count,) or set(samples.values) != observables:
            raise ValueError(f"Category {key} has invalid indices or observables")
        if np.any(samples.frame_indices < 0) or np.any(samples.frame_indices >= decision_frames.size):
            raise ValueError(f"Category {key} has invalid frame indices")
        if not np.all(decision_frames[samples.frame_indices]):
            raise ValueError(f"Category {key} contains stopped frames")
        if any(np.asarray(value).shape != (count,) for value in samples.values.values()):
            raise ValueError(f"Category {key} has invalid value lengths")
