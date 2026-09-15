"""Linear, immutable decoder history and portable session persistence."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from threading import RLock
from typing import Optional

import numpy as np

from .algorithms import BatchDecoderEngine, TannerGraph, calculate_metrics, initial_state
from .models import (
    Algorithm,
    ComparisonHistory,
    DecoderState,
    FrameBatch,
    FtgdbfParameters,
    PmgdbfParameters,
    Snapshot,
    TransitionRecord,
    parameters_from_dict,
)


SESSION_FORMAT = "bf-gd-energy-explorer"
SESSION_VERSION = 1


@dataclass(frozen=True)
class SessionView:
    algorithm: Algorithm
    cursor: int
    snapshots: tuple
    transitions: tuple
    comparison: Optional[ComparisonHistory]
    metadata: dict


class ExplorerSession:
    """Own one user's state; every forward transition creates a new snapshot."""

    def __init__(
        self,
        graph,
        batch,
        seed,
        workers=1,
        algorithm=Algorithm.FTGDBF,
        momentum_length=7,
        metadata=None,
    ):
        self.graph = graph
        self.batch = batch
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        self.metadata = dict(metadata or {})
        self.metadata["batch_digest"] = _batch_digest(batch)
        self.lock = RLock()
        self.comparison = None
        self.engine = BatchDecoderEngine(
            graph,
            batch,
            seed=self.seed,
            workers=self.workers,
        )
        self.reset(algorithm, momentum_length)

    def reset(self, algorithm, momentum_length=7):
        with self.lock:
            self.algorithm = Algorithm(algorithm)
            state = _freeze_state(
                initial_state(self.batch, self.algorithm, int(momentum_length))
            )
            metrics = calculate_metrics(state.x, self.batch.transmitted_symbols)
            self.snapshots = [Snapshot(0, state, metrics)]
            self.transitions = []
            self.cursor = 0

    @property
    def current(self):
        return self.snapshots[self.cursor]

    def preview(self, parameters):
        with self.lock:
            _validate_parameter_type(self.algorithm, parameters)
            return self.engine.step(
                self.current.state,
                parameters,
                self.cursor,
            )

    def step_forward(self, parameters):
        with self.lock:
            next_state, observation = self.preview(parameters)
            del self.snapshots[self.cursor + 1:]
            del self.transitions[self.cursor:]
            next_state = _freeze_state(next_state)
            next_snapshot = Snapshot(
                iteration=self.cursor + 1,
                state=next_state,
                metrics=observation.after,
            )
            self.transitions.append(TransitionRecord(
                from_iteration=self.cursor,
                to_iteration=self.cursor + 1,
                parameters=parameters,
                before=observation.before,
                after=observation.after,
            ))
            self.snapshots.append(next_snapshot)
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
                algorithm=self.algorithm,
                cursor=self.cursor,
                snapshots=tuple(self.snapshots),
                transitions=tuple(self.transitions),
                comparison=self.comparison,
                metadata=dict(self.metadata),
            )

    def to_bytes(self):
        with self.lock:
            metadata = {
                "format": SESSION_FORMAT,
                "version": SESSION_VERSION,
                "algorithm": self.algorithm.value,
                "cursor": self.cursor,
                "seed": self.seed,
                "workers": self.workers,
                "dataset": self.metadata,
                "transitions": [item.to_dict() for item in self.transitions],
                "has_ages": self.snapshots[0].state.ages is not None,
            }
            arrays = {
                "metadata": np.asarray(json.dumps(metadata)),
                "received": self.batch.received,
                "transmitted_symbols": self.batch.transmitted_symbols,
                "edge_cn": self.graph.edge_cn,
                "edge_vn": self.graph.edge_vn,
                "check_offsets": self.graph.check_offsets,
                "states_x": np.stack(
                    [item.state.x for item in self.snapshots],
                    axis=0,
                ),
                "states_active": np.stack(
                    [item.state.active for item in self.snapshots],
                    axis=0,
                ),
            }
            if metadata["has_ages"]:
                arrays["states_ages"] = np.stack(
                    [item.state.ages for item in self.snapshots],
                    axis=0,
                )
            output = BytesIO()
            np.savez_compressed(output, **arrays)
            return output.getvalue()

    @classmethod
    def from_bytes(cls, payload, workers=None):
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            required = {
                "metadata",
                "received",
                "transmitted_symbols",
                "edge_cn",
                "edge_vn",
                "check_offsets",
                "states_x",
                "states_active",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"session is missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata"].item()))
            _validate_metadata(metadata)
            batch = FrameBatch(
                received=np.ascontiguousarray(archive["received"]),
                transmitted_symbols=np.ascontiguousarray(
                    archive["transmitted_symbols"]
                ),
            )
            batch.validate()
            graph = TannerGraph(
                block_length=batch.block_length,
                n_checks=len(archive["check_offsets"]) - 1,
                edge_cn=np.ascontiguousarray(archive["edge_cn"], dtype=np.int32),
                edge_vn=np.ascontiguousarray(archive["edge_vn"], dtype=np.int32),
                check_offsets=np.ascontiguousarray(
                    archive["check_offsets"], dtype=np.int32
                ),
            )
            graph.validate()
            states_x = archive["states_x"]
            states_active = archive["states_active"]
            if states_x.ndim != 3 or states_x.shape[1:] != batch.received.shape:
                raise ValueError("saved decoder states have an invalid shape")
            if states_x.shape[0] == 0:
                raise ValueError("session history must contain an initial state")
            if states_active.shape != states_x.shape[:2]:
                raise ValueError("saved active masks have an invalid shape")
            if not np.all(np.isin(states_x, (-1, 1))):
                raise ValueError("saved decoder states are not binary signs")
            if metadata["has_ages"]:
                if "states_ages" not in archive.files:
                    raise ValueError("PMGDBF session does not contain ages")
                states_ages = archive["states_ages"]
                if states_ages.shape != states_x.shape:
                    raise ValueError("saved momentum ages have an invalid shape")
                if np.any(states_ages < 0):
                    raise ValueError("saved momentum ages must be non-negative")
            else:
                states_ages = None

            session = cls(
                graph=graph,
                batch=batch,
                seed=int(metadata["seed"]),
                workers=int(metadata["workers"] if workers is None else workers),
                algorithm=metadata["algorithm"],
                metadata=metadata.get("dataset", {}),
            )
            session.metadata["workers"] = session.workers
            session.snapshots = []
            for index in range(states_x.shape[0]):
                state = _freeze_state(DecoderState(
                    x=states_x[index],
                    active=states_active[index],
                    ages=None if states_ages is None else states_ages[index],
                ))
                session.snapshots.append(Snapshot(
                    iteration=index,
                    state=state,
                    metrics=calculate_metrics(
                        state.x,
                        batch.transmitted_symbols,
                    ),
                ))
            session.transitions = _restore_transitions(
                metadata["algorithm"],
                metadata["transitions"],
                session.snapshots,
            )
            session.cursor = int(metadata["cursor"])
            if not 0 <= session.cursor < len(session.snapshots):
                raise ValueError("saved cursor is outside the history")
            session.batch.received.setflags(write=False)
            session.batch.transmitted_symbols.setflags(write=False)
            return session

    def comparison_history(self, name):
        with self.lock:
            return ComparisonHistory(
                name=name,
                algorithm=self.algorithm.title,
                iterations=np.arange(len(self.snapshots), dtype=np.int32),
                ber=np.asarray(
                    [item.metrics.ber for item in self.snapshots],
                    dtype=np.float64,
                ),
                fer=np.asarray(
                    [item.metrics.fer for item in self.snapshots],
                    dtype=np.float64,
                ),
                metadata=dict(self.metadata),
            )


def comparison_from_bytes(payload, name="comparison"):
    session = ExplorerSession.from_bytes(payload, workers=1)
    return session.comparison_history(name)


def compatible_datasets(left, right):
    """Require exact fixed inputs before overlaying performance histories."""
    left_digest = left.get("batch_digest")
    right_digest = right.get("batch_digest")
    return bool(left_digest) and left_digest == right_digest


def _freeze_state(state):
    x = np.array(state.x, copy=True, dtype=np.int8, order="C")
    active = np.array(state.active, copy=True, dtype=bool, order="C")
    ages = None
    if state.ages is not None:
        ages = np.array(state.ages, copy=True, dtype=np.int16, order="C")
        ages.setflags(write=False)
    x.setflags(write=False)
    active.setflags(write=False)
    return DecoderState(x=x, active=active, ages=ages)


def _batch_digest(batch):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(batch.received).view(np.uint8))
    digest.update(np.ascontiguousarray(batch.transmitted_symbols).view(np.uint8))
    return digest.hexdigest()


def _validate_parameter_type(algorithm, parameters):
    expected = FtgdbfParameters if algorithm is Algorithm.FTGDBF else PmgdbfParameters
    if not isinstance(parameters, expected):
        raise ValueError(f"parameters do not belong to {algorithm.title}")


def _validate_metadata(metadata):
    if metadata.get("format") != SESSION_FORMAT:
        raise ValueError("file is not an energy explorer session")
    if metadata.get("version") != SESSION_VERSION:
        raise ValueError("unsupported session version")
    algorithm = Algorithm(metadata.get("algorithm"))
    if bool(metadata.get("has_ages")) != (algorithm is Algorithm.PMGDBF):
        raise ValueError("session state does not match its algorithm")
    if not isinstance(metadata.get("dataset"), dict):
        raise ValueError("session dataset metadata is invalid")
    if int(metadata.get("workers", 0)) <= 0:
        raise ValueError("session worker count is invalid")
    int(metadata.get("seed"))
    int(metadata.get("cursor"))
    if not isinstance(metadata.get("transitions"), list):
        raise ValueError("session transitions are invalid")


def _restore_transitions(algorithm, rows, snapshots):
    if len(rows) != len(snapshots) - 1:
        raise ValueError("transition and snapshot counts do not match")
    transitions = []
    for index, row in enumerate(rows):
        if row.get("from_iteration") != index or row.get("to_iteration") != index + 1:
            raise ValueError("session history is not linear")
        transitions.append(TransitionRecord(
            from_iteration=index,
            to_iteration=index + 1,
            parameters=parameters_from_dict(algorithm, row["parameters"]),
            before=snapshots[index].metrics,
            after=snapshots[index + 1].metrics,
        ))
    return transitions
