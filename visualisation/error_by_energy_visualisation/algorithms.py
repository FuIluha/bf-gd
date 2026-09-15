"""Batch-oriented, testable FTGDBF and PMGDBF stepping engines."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .models import (
    Algorithm,
    DecoderState,
    FtgdbfParameters,
    FrameBatch,
    Metrics,
    PmgdbfParameters,
    StepObservation,
)


MAX_STORED_AGE = np.iinfo(np.int16).max


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
        check_degrees = np.bincount(edge_cn, minlength=pcm.shape[0])
        if np.any(check_degrees == 0):
            raise ValueError("degree-zero checks are not supported")
        variable_degrees = np.bincount(edge_vn, minlength=pcm.shape[1])
        if np.any(variable_degrees == 0):
            raise ValueError("degree-zero variables are not supported")
        offsets = np.ascontiguousarray(
            np.concatenate(([0], np.cumsum(check_degrees))),
            dtype=np.int32,
        )
        result = cls(
            block_length=pcm.shape[1],
            n_checks=pcm.shape[0],
            edge_cn=edge_cn,
            edge_vn=edge_vn,
            check_offsets=offsets,
        )
        result.validate()
        return result

    def validate(self):
        if self.block_length <= 0 or self.n_checks <= 0:
            raise ValueError("Tanner graph dimensions must be positive")
        if self.edge_cn.ndim != 1 or self.edge_vn.ndim != 1:
            raise ValueError("Tanner graph edge arrays must be one-dimensional")
        if self.edge_cn.size == 0 or self.edge_cn.size != self.edge_vn.size:
            raise ValueError("Tanner graph edge arrays have invalid lengths")
        if self.check_offsets.shape != (self.n_checks + 1,):
            raise ValueError("Tanner graph check offsets have an invalid shape")
        if self.check_offsets[0] != 0 or self.check_offsets[-1] != self.edge_cn.size:
            raise ValueError("Tanner graph check offsets do not cover all edges")
        if np.any(np.diff(self.check_offsets) <= 0):
            raise ValueError("degree-zero checks are not supported")
        if np.any(self.edge_cn < 0) or np.any(self.edge_cn >= self.n_checks):
            raise ValueError("Tanner graph contains an invalid check index")
        if np.any(self.edge_vn < 0) or np.any(self.edge_vn >= self.block_length):
            raise ValueError("Tanner graph contains an invalid variable index")
        expected_checks = np.repeat(
            np.arange(self.n_checks, dtype=np.int32),
            np.diff(self.check_offsets),
        )
        if not np.array_equal(self.edge_cn, expected_checks):
            raise ValueError("Tanner graph edges must be grouped by check")
        if np.unique(self.edge_vn).size != self.block_length:
            raise ValueError("degree-zero variables are not supported")

    def check_products(self, values):
        edge_values = values[:, self.edge_vn]
        return np.multiply.reduceat(
            edge_values,
            self.check_offsets[:-1],
            axis=1,
        ).astype(np.int8, copy=False)

    def incident_check_sums(self, check_products):
        result = np.zeros(
            (check_products.shape[0], self.block_length),
            dtype=np.float64,
        )
        np.add.at(
            result.T,
            self.edge_vn,
            check_products[:, self.edge_cn].T,
        )
        return result

    def satisfied(self, values):
        return np.all(self.check_products(values) == 1, axis=1)


def calculate_metrics(x, transmitted_symbols):
    errors = x != transmitted_symbols
    bit_errors = int(np.count_nonzero(errors))
    frame_errors = int(np.count_nonzero(np.any(errors, axis=1)))
    return Metrics(
        ber=bit_errors / errors.size,
        fer=frame_errors / errors.shape[0],
        bit_errors=bit_errors,
        frame_errors=frame_errors,
    )


def initial_state(batch, algorithm, momentum_length):
    channel_signs = np.where(batch.received >= 0, 1, -1).astype(np.int8)
    active = np.ones(batch.frames, dtype=bool)
    if Algorithm(algorithm) is Algorithm.FTGDBF:
        return DecoderState(x=channel_signs, active=active)
    if int(momentum_length) <= 0:
        raise ValueError("momentum_length must be positive")
    # "Never flipped" must remain distinct when L is changed interactively.
    ages = np.full(channel_signs.shape, MAX_STORED_AGE, dtype=np.int16)
    return DecoderState(x=channel_signs, active=active, ages=ages)


class BatchDecoderEngine:
    """Apply one decoder transition to many fixed channel realizations."""

    def __init__(self, graph, batch, seed, workers=1):
        self.graph = graph
        self.batch = batch
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        batch.validate()
        graph.validate()
        if batch.block_length != graph.block_length:
            raise ValueError("frame batch and parity-check matrix do not match")

    def step(self, state, parameters, iteration):
        parameters.validate()
        self._validate_state(state)
        random_values = None
        if isinstance(parameters, PmgdbfParameters):
            random_values = self._random_values(iteration)

        slices = self._frame_slices()
        if len(slices) == 1:
            parts = [self._step_slice(state, parameters, slices[0], random_values)]
        else:
            with ThreadPoolExecutor(max_workers=len(slices)) as executor:
                futures = [
                    executor.submit(
                        self._step_slice,
                        state,
                        parameters,
                        frame_slice,
                        random_values,
                    )
                    for frame_slice in slices
                ]
                parts = [future.result() for future in futures]

        return self._combine(parts, state)

    def _step_slice(self, state, parameters, frame_slice, random_values):
        x = state.x[frame_slice]
        received = self.batch.received[frame_slice]
        transmitted = self.batch.transmitted_symbols[frame_slice]
        active = state.active[frame_slice]
        check_products = self.graph.check_products(x)
        decoded_now = np.all(check_products == 1, axis=1)
        decision_frames = active & ~decoded_now
        incident = self.graph.incident_check_sums(check_products)

        if isinstance(parameters, FtgdbfParameters):
            energy = parameters.alpha * x * received + incident
            threshold = np.zeros(x.shape[0], dtype=np.float64)
            flip_mask = energy <= 0.0
            next_ages = None
        else:
            ages = np.minimum(
                state.ages[frame_slice].astype(np.int32) + 1,
                MAX_STORED_AGE,
            ).astype(np.int16)
            momentum = np.zeros(x.shape, dtype=np.float64)
            within_memory = ages <= parameters.L
            rho = np.asarray(parameters.rho, dtype=np.float64)
            momentum[within_memory] = rho[ages[within_memory] - 1]
            energy = (
                parameters.alpha * x * received
                + incident
                + momentum
            )
            threshold = np.min(energy, axis=1) + parameters.delta
            selected = random_values[frame_slice] < parameters.p
            flip_mask = (energy <= threshold[:, None]) & selected
            next_ages = ages

        flip_mask &= decision_frames[:, None]
        next_x = x.copy()
        next_x[flip_mask] *= -1
        if next_ages is not None:
            next_ages = next_ages.copy()
            next_ages[flip_mask] = 0

        next_active = decision_frames & ~self.graph.satisfied(next_x)
        should_flip = x != transmitted
        correct_action = flip_mask == should_flip
        margin = energy - threshold[:, None]
        return {
            "x": next_x,
            "ages": next_ages,
            "active": next_active,
            "energy": energy,
            "threshold": threshold,
            "margin": margin,
            "should_flip": should_flip,
            "flip_mask": flip_mask,
            "correct_action": correct_action,
            "decision_frames": decision_frames,
        }

    def _combine(self, parts, previous_state):
        next_x = np.concatenate([part["x"] for part in parts], axis=0)
        next_active = np.concatenate([part["active"] for part in parts])
        if parts[0]["ages"] is None:
            next_ages = None
        else:
            next_ages = np.concatenate([part["ages"] for part in parts], axis=0)
        next_state = DecoderState(
            x=next_x,
            active=next_active,
            ages=next_ages,
        )
        before = calculate_metrics(
            previous_state.x,
            self.batch.transmitted_symbols,
        )
        after = calculate_metrics(next_x, self.batch.transmitted_symbols)
        observation = StepObservation(
            energy=np.concatenate([part["energy"] for part in parts], axis=0),
            threshold=np.concatenate([part["threshold"] for part in parts]),
            margin=np.concatenate([part["margin"] for part in parts], axis=0),
            should_flip=np.concatenate([part["should_flip"] for part in parts], axis=0),
            flip_mask=np.concatenate([part["flip_mask"] for part in parts], axis=0),
            correct_action=np.concatenate(
                [part["correct_action"] for part in parts], axis=0
            ),
            decision_frames=np.concatenate(
                [part["decision_frames"] for part in parts]
            ),
            before=before,
            after=after,
        )
        return next_state, observation

    def _random_values(self, iteration):
        seed = np.random.SeedSequence([
            self.seed,
            int(iteration),
            0x504D4744,
        ])
        generator = np.random.Generator(np.random.Philox(seed))
        return generator.random(self.batch.received.shape)

    def _frame_slices(self):
        workers = min(self.workers, self.batch.frames)
        boundaries = np.linspace(0, self.batch.frames, workers + 1, dtype=int)
        return [
            slice(int(boundaries[index]), int(boundaries[index + 1]))
            for index in range(workers)
            if boundaries[index] != boundaries[index + 1]
        ]

    def _validate_state(self, state):
        if state.x.shape != self.batch.received.shape:
            raise ValueError("decoder state shape does not match the frame batch")
        if state.active.shape != (self.batch.frames,):
            raise ValueError("active-frame mask has an invalid shape")
        if state.ages is not None and state.ages.shape != state.x.shape:
            raise ValueError("momentum age shape does not match the decoder state")
