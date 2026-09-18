"""Linear immutable decoder history and portable session persistence."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from threading import RLock
from types import MappingProxyType
from typing import Optional

import numpy as np

from .algorithms import BatchDecoderEngine, TannerGraph
from .models import ComparisonHistory, DecoderState, FrameBatch, Snapshot, TransitionRecord
from .supported import decoder_catalog, decoder_class


SESSION_FORMAT = "bf-gd-energy-explorer"
SESSION_VERSION = 2


@dataclass(frozen=True)
class SessionView:
    algorithm: str
    cursor: int
    snapshots: tuple
    transitions: tuple
    comparison: Optional[ComparisonHistory]
    metadata: dict


class ExplorerSession:
    def __init__(self, graph, batch, seed, workers=1, algorithm=None, metadata=None):
        self.graph, self.batch, self.seed = graph, batch, int(seed)
        self.workers = max(1, int(workers))
        self.metadata = dict(metadata or {})
        self.metadata["batch_digest"] = _batch_digest(batch)
        self.lock = RLock()
        self.comparison = None
        self.engine = BatchDecoderEngine(graph, batch, seed=self.seed, workers=self.workers)
        self._pcm = np.zeros((graph.n_checks, graph.block_length), dtype=np.uint8)
        self._pcm[graph.edge_cn, graph.edge_vn] = 1
        self.reset(algorithm or next(iter(decoder_catalog())))

    @property
    def decoder(self):
        return self._decoder

    @property
    def spec(self):
        return self.decoder.describe()

    def reset(self, algorithm, parameters=None):
        with self.lock:
            decoder_type = decoder_class(algorithm)
            parsed = decoder_type.validate_parameters(
                parameters or decoder_type.describe().defaults(),
            )
            decoder = decoder_type(
                None, pcm=self._pcm, block_length=self.graph.block_length,
                n_checks=self.graph.n_checks, n_iterations=1,
                is_systematic=False, **parsed,
            )
            state = _freeze_state(self.engine.initial_state(decoder, parsed))
            metrics = self.engine._metrics(decoder, state.fields)
            self.algorithm, self._decoder = algorithm, decoder
            self.snapshots = [Snapshot(0, state, metrics)]
            self.transitions, self.cursor = [], 0

    @property
    def current(self):
        return self.snapshots[self.cursor]

    def preview(self, parameters):
        with self.lock:
            parsed = self.decoder.validate_parameters(parameters)
            return self.engine.step(self.decoder, self.current.state, parsed, self.cursor)

    def step_forward(self, parameters):
        with self.lock:
            parsed = self.decoder.validate_parameters(parameters)
            next_state, observation = self.preview(parsed)
            del self.snapshots[self.cursor + 1:]
            del self.transitions[self.cursor:]
            self.transitions.append(TransitionRecord(
                self.cursor, self.cursor + 1, parsed, observation.before, observation.after,
            ))
            self.snapshots.append(Snapshot(self.cursor + 1, _freeze_state(next_state), observation.after))
            self.cursor += 1
            return observation

    def step_back(self):
        with self.lock:
            if self.cursor > 0:
                self.cursor -= 1
            return self.current

    def view(self):
        with self.lock:
            return SessionView(
                self.algorithm, self.cursor, tuple(self.snapshots), tuple(self.transitions),
                self.comparison, dict(self.metadata),
            )

    def to_bytes(self):
        with self.lock:
            names = list(self.snapshots[0].state.fields)
            metadata = {
                "format": SESSION_FORMAT, "version": SESSION_VERSION,
                "algorithm": self.algorithm, "cursor": self.cursor, "seed": self.seed,
                "workers": self.workers, "dataset": self.metadata, "fields": names,
                "transitions": [item.to_dict() for item in self.transitions],
            }
            arrays = {
                "metadata": np.asarray(json.dumps(metadata)),
                "received": self.batch.received,
                "transmitted_symbols": self.batch.transmitted_symbols,
                "edge_cn": self.graph.edge_cn, "edge_vn": self.graph.edge_vn,
                "check_offsets": self.graph.check_offsets,
                "states_active": np.stack([item.state.active for item in self.snapshots]),
            }
            for name in names:
                arrays[f"state_{name}"] = np.stack([
                    item.state.fields[name] for item in self.snapshots
                ])
            output = BytesIO()
            np.savez_compressed(output, **arrays)
            return output.getvalue()

    @classmethod
    def from_bytes(cls, payload, workers=None):
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            required = {"metadata", "received", "transmitted_symbols", "edge_cn", "edge_vn", "check_offsets", "states_active"}
            if not required.issubset(archive.files):
                raise ValueError("Session archive is incomplete")
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format") != SESSION_FORMAT or metadata.get("version") != SESSION_VERSION:
                raise ValueError("Unsupported session format/version")
            decoder_class(metadata["algorithm"])
            names = metadata["fields"]
            if not names or len(names) != len(set(names)) or any(
                not isinstance(name, str) or not name.isidentifier() for name in names
            ):
                raise ValueError("Invalid state field names")
            batch = FrameBatch(np.ascontiguousarray(archive["received"]), np.ascontiguousarray(archive["transmitted_symbols"]))
            batch.validate()
            graph = TannerGraph(
                batch.block_length, len(archive["check_offsets"]) - 1,
                np.ascontiguousarray(archive["edge_cn"], dtype=np.int32),
                np.ascontiguousarray(archive["edge_vn"], dtype=np.int32),
                np.ascontiguousarray(archive["check_offsets"], dtype=np.int32),
            )
            graph.validate()
            active = archive["states_active"]
            if active.ndim != 2 or active.shape[1] != batch.frames or active.shape[0] < 1:
                raise ValueError("Invalid saved active masks")
            states = {}
            for name in names:
                key = f"state_{name}"
                if key not in archive.files:
                    raise ValueError(f"Missing decoder state: {name}")
                states[name] = archive[key]
                if states[name].shape[0] != active.shape[0] or states[name].shape[1] != batch.frames:
                    raise ValueError(f"Invalid shape of decoder state: {name}")
            session = cls(
                graph, batch, int(metadata["seed"]),
                workers=int(metadata["workers"] if workers is None else workers),
                algorithm=metadata["algorithm"], metadata=metadata.get("dataset", {}),
            )
            session.snapshots = []
            for index in range(active.shape[0]):
                state = _freeze_state(DecoderState({name: states[name][index] for name in names}, active[index]))
                metrics = session.engine._metrics(session.decoder, state.fields)
                session.snapshots.append(Snapshot(index, state, metrics))
            rows = metadata["transitions"]
            if len(rows) != len(session.snapshots) - 1:
                raise ValueError("Transition and snapshot counts differ")
            session.transitions = []
            for index, row in enumerate(rows):
                if row["from_iteration"] != index or row["to_iteration"] != index + 1:
                    raise ValueError("Session history is not linear")
                session.transitions.append(TransitionRecord(
                    index, index + 1, session.decoder.validate_parameters(row["parameters"]),
                    session.snapshots[index].metrics, session.snapshots[index + 1].metrics,
                ))
            session.cursor = int(metadata["cursor"])
            if not 0 <= session.cursor < len(session.snapshots):
                raise ValueError("Saved cursor outside history")
            session.batch.received.setflags(write=False)
            session.batch.transmitted_symbols.setflags(write=False)
            return session

    def comparison_history(self, name):
        with self.lock:
            return ComparisonHistory(
                name, self.spec.title, np.arange(len(self.snapshots), dtype=np.int32),
                np.asarray([item.metrics.ber for item in self.snapshots]),
                np.asarray([item.metrics.fer for item in self.snapshots]), dict(self.metadata),
            )


def comparison_from_bytes(payload, name="comparison"):
    return ExplorerSession.from_bytes(payload, workers=1).comparison_history(name)


def compatible_datasets(left, right):
    return bool(left.get("batch_digest")) and left["batch_digest"] == right.get("batch_digest")


def _freeze_state(state):
    fields = {key: np.array(value, copy=True, order="C") for key, value in state.fields.items()}
    active = np.array(state.active, copy=True, dtype=bool, order="C")
    for value in fields.values():
        value.setflags(write=False)
    active.setflags(write=False)
    return DecoderState(MappingProxyType(fields), active)


def _batch_digest(batch):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(batch.received).view(np.uint8))
    digest.update(np.ascontiguousarray(batch.transmitted_symbols).view(np.uint8))
    return digest.hexdigest()
