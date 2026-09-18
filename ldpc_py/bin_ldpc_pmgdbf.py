import numpy as np
from .bin_ldpc import BinLdpcDecoderBase
from visualisation.error_by_energy_visualisation.base import VisualizableDecoderBase, masked_samples
from visualisation.error_by_energy_visualisation.models import (
    CategorySpec, DecoderSpec, ObservableSpec, ParameterSpec, StepResult,
)

MAX_AGE = np.iinfo(np.int32).max


class BinLdpcPmgdbfDecoder(BinLdpcDecoderBase, VisualizableDecoderBase):
    """Implementation of probabilistic momentum gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.alpha = kwargs["alpha"]
        self.p = kwargs["p"]
        rho = np.asarray(kwargs["rho"], dtype=np.float32)
        self.L = kwargs["L"]

        if len(rho) != self.L:
            raise ValueError("Momentum length must be equal to L")

        self.rho = np.concatenate((
            rho,
            np.array([0.0], dtype=np.float32),
        ))
        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)

        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        self.edges_count = len(self.edge_cn)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks
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

    @classmethod
    def describe(cls):
        return DecoderSpec(
            key="pmgdbf", title="PMGDBF",
            parameters=(
                ParameterSpec("delta", "delta", "float", 1.0, 0.0, 4.0, 0.01),
                ParameterSpec("alpha", "alpha", "float", 1.8, 0.0, 4.0, 0.01),
                ParameterSpec("p", "p", "float", 0.9, 0.0, 1.0, 0.01),
                ParameterSpec("rho", "rho (через запятую)", "float_list", [2, 2, 2, 2, 2, 1, 1]),
                ParameterSpec("L", "L", "int", 7, 1, 32, 1),
            ),
            observables=(
                ObservableSpec("energy", "Энергия E"),
                ObservableSpec("margin", "Запас E − E_threshold", True),
            ),
            categories=(
                CategorySpec("correct", "Верное действие", "#2e7d32"),
                CategorySpec("incorrect", "Ошибочное действие", "#c62828"),
            ),
        )

    @classmethod
    def validate_parameters(cls, parameters):
        parsed = super().validate_parameters(parameters)
        if len(parsed["rho"]) != parsed["L"]:
            raise ValueError("Длина rho должна совпадать с L")
        return parsed

    def initial_state(self, received, parameters):
        x = np.where(received >= 0, 1, -1).astype(np.int8)
        return {"x": x, "ages": np.full(x.shape, MAX_AGE, dtype=np.int32)}

    def step_state(self, state, received, parameters, iteration, rng):
        x = state["x"]
        ages = np.where(
            state["ages"] == MAX_AGE, MAX_AGE,
            np.minimum(state["ages"], parameters["L"]) + 1,
        ).astype(np.int32)
        momentum = np.zeros(x.shape, dtype=np.float32)
        within = ages <= parameters["L"]
        rho = np.asarray(parameters["rho"], dtype=np.float32)
        momentum[within] = rho[ages[within] - 1]
        checks = self.bpsk_syndrome(x)
        incident = np.bincount(
            self.edge_vn, weights=checks[self.edge_cn], minlength=self.block_length,
        )
        energy = parameters["alpha"] * x * received + incident + momentum
        threshold = np.min(energy) + parameters["delta"]
        flip = (energy <= threshold) & (rng.random(self.block_length) < parameters["p"])
        next_x = x.copy()
        next_x[flip] *= -1
        ages[flip] = 0
        return StepResult(
            {"x": next_x, "ages": ages},
            {"energy": energy, "margin": energy - threshold, "flip": flip},
        )

    def hard_decision(self, state):
        return state["x"]

    def is_decoded(self, state):
        return bool(np.all(self.bpsk_syndrome(self.hard_decision(state)) == 1))

    def classify(self, before, after, diagnostics, active, transmitted):
        correct = (diagnostics["flip"] == (before["x"] != transmitted)) & active[:, None]
        observed = np.broadcast_to(active[:, None], correct.shape)
        values = {key: diagnostics[key] for key in ("energy", "margin")}
        return {
            "correct": masked_samples(correct, values),
            "incorrect": masked_samples(observed & ~correct, values),
        }

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        received = llr_in.copy()
        parameters = {"delta": self.delta, "alpha": self.alpha, "p": self.p,
                      "rho": self.rho[:-1], "L": self.L}
        state = self.initial_state(received, parameters)
        for iteration in range(self.n_iterations): # iteration loop
            if self.is_decoded(state):
                llr_out[:] = state["x"]
                return iteration # exit the iteration loop;

            state = self.step_state(state, received, parameters, iteration, rng).fields

        llr_out[:] = state["x"]
        return self.n_iterations
