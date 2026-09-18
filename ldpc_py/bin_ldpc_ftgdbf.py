import numpy as np

from .bin_ldpc import BinLdpcDecoderBase
from visualisation.error_by_energy_visualisation.base import VisualizableDecoderBase, masked_samples
from visualisation.error_by_energy_visualisation.models import (
    CategorySpec, DecoderSpec, ObservableSpec, ParameterSpec, StepResult,
)


class BinLdpcFtgdbfDecoder(BinLdpcDecoderBase, VisualizableDecoderBase):
    """Deterministic GDBF decoder with a fixed zero flipping threshold."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
        if self.alpha < 0:
            raise ValueError("Alpha must be non-negative")

        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)
        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)
        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks,
        )
        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def local_energy(self, x, received, check_syndromes, alpha=None):
        incident_syndrome_sums = np.bincount(
            self.edge_vn,
            weights=check_syndromes[self.edge_cn],
            minlength=self.block_length,
        )
        return (self.alpha if alpha is None else alpha) * x * received + incident_syndrome_sums

    @classmethod
    def describe(cls):
        return DecoderSpec(
            key="ftgdbf", title="FTGDBF",
            parameters=(ParameterSpec("alpha", "alpha", "float", 1.8, 0.0, 4.0, 0.01),),
            observables=(ObservableSpec("energy", "Энергия E", True),),
            categories=(
                CategorySpec("correct", "Верное действие", "#2e7d32"),
                CategorySpec("incorrect", "Ошибочное действие", "#c62828"),
            ),
        )

    def initial_state(self, received, parameters):
        return {"x": np.where(received >= 0, 1, -1).astype(np.int8)}

    def step_state(self, state, received, parameters, iteration, rng):
        x = state["x"]
        check_syndromes = self.bpsk_syndrome(x)
        energy = self.local_energy(x, received, check_syndromes, parameters["alpha"])
        flip = energy <= 0
        next_x = x.copy()
        next_x[flip] *= -1
        return StepResult({"x": next_x}, {"energy": energy, "flip": flip})

    def hard_decision(self, state):
        return state["x"]

    def is_decoded(self, state):
        return bool(np.all(self.bpsk_syndrome(self.hard_decision(state)) == 1))

    def classify(self, before, after, diagnostics, active, transmitted):
        correct = (diagnostics["flip"] == (before["x"] != transmitted)) & active[:, None]
        observed = np.broadcast_to(active[:, None], correct.shape)
        values = {"energy": diagnostics["energy"]}
        return {
            "correct": masked_samples(correct, values),
            "incorrect": masked_samples(observed & ~correct, values),
        }

    def decode(self, llr_in, llr_out, rng=None):
        received = llr_in.copy()
        parameters = {"alpha": self.alpha}
        state = self.initial_state(received, parameters)

        for iteration in range(self.n_iterations):
            if self.is_decoded(state):
                llr_out[:] = state["x"]
                return iteration

            state = self.step_state(state, received, parameters, iteration, rng).fields

        llr_out[:] = state["x"]
        return self.n_iterations

    def trace(self, llr_in):
        """Return pre-flip states and energies for one decoding trajectory."""
        received = llr_in.copy()
        parameters = {"alpha": self.alpha}
        state = self.initial_state(received, parameters)
        trajectory = []

        for iteration in range(self.n_iterations):
            if self.is_decoded(state):
                return True, trajectory

            result = self.step_state(state, received, parameters, iteration, None)
            trajectory.append((state["x"].copy(), result.diagnostics["energy"].astype(np.float32)))
            state = result.fields

        decoded = self.is_decoded(state)
        return decoded, trajectory
