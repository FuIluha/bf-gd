"""Python implementation of the gradient-descent min-sum decoder."""

import numpy as np
from .bin_ldpc import BinLdpcDecoderBase
from visualisation.error_by_energy_visualisation.base import VisualizableDecoderBase, masked_samples
from visualisation.error_by_energy_visualisation.models import (
    CategoryGroupSpec, CategorySpec, DecoderSpec, ObservableSpec, ParameterSpec, StepResult,
)

class BinLdpcGdmsDecoder(BinLdpcDecoderBase, VisualizableDecoderBase):
    """Gradient-descent min-sum decoder with a smooth bipolar penalty."""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.channel_weight = float(kwargs["channel_weight"])
        self.bipolar_weight = float(kwargs["bipolar_weight"])

        if not np.isfinite(self.channel_weight) or self.channel_weight <= 0:
            raise ValueError("channel_weight must be finite and positive")
        if not np.isfinite(self.bipolar_weight) or self.bipolar_weight < 0:
            raise ValueError("bipolar_weight must be finite and non-negative")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0:
            raise ValueError("Learning rate decay must be non-negative")

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

        if np.any(check_degrees < 2):
            raise ValueError("GDMS requires check degree of at least two")

    @classmethod
    def describe(cls):
        return DecoderSpec(
            key="gdms", title="GDMS",
            parameters=(
                ParameterSpec("learning_rate", "learning_rate", "float", 0.05, 0.000001, 5.0, 0.01),
                ParameterSpec("learning_rate_decay", "learning_rate_decay", "float", 0.03, 0.0, 5.0, 0.01),
                ParameterSpec("channel_weight", "channel_weight", "float", 1.0, 0.000001, 5.0, 0.01),
                ParameterSpec("bipolar_weight", "bipolar_weight", "float", 0.1, 0.0, 5.0, 0.01),
            ),
            observables=(ObservableSpec("step", "Шаг сообщения Δq", True),),
            categories=(
                CategorySpec("toward_unchanged", "К истинному биту · решение не изменилось", "#2e7d32"),
                CategorySpec("toward_changed", "К истинному биту · решение изменилось", "#66bb6a"),
                CategorySpec("away_unchanged", "От истинного бита · решение не изменилось", "#ef6c00"),
                CategorySpec("away_changed", "От истинного бита · решение изменилось", "#c62828"),
            ),
            category_groups=(
                CategoryGroupSpec("direction", "По направлению и изменению решения", (
                    "toward_unchanged", "toward_changed", "away_unchanged", "away_changed",
                )),
            ),
        )

    def initial_state(self, received, parameters):
        x = np.asarray(received, dtype=np.float64).copy()
        return {"x": x, "q": x[self.edge_vn].copy()}

    def step_state(self, state, received, parameters, iteration, rng):
        next_x, next_q = self.update_state(
            received, state["x"], state["q"], iteration, parameters,
        )
        return StepResult({"x": next_x, "q": next_q}, {"step": next_q - state["q"]})

    def hard_decision(self, state):
        return np.where(state["x"] >= 0, 1, -1).astype(np.int8)

    def is_decoded(self, state):
        return bool(np.all(self.bpsk_syndrome(self.hard_decision(state)) == 1))

    def classify(self, before, after, diagnostics, active, transmitted):
        step = diagnostics["step"]
        toward = (step >= 0) == (transmitted[:, self.edge_vn] > 0)
        changed = (before["q"] >= 0) != (after["q"] >= 0)
        observed = np.broadcast_to(active[:, None], step.shape)
        values = {"step": step}
        return {
            "toward_unchanged": masked_samples(observed & toward & ~changed, values),
            "toward_changed": masked_samples(observed & toward & changed, values),
            "away_unchanged": masked_samples(observed & ~toward & ~changed, values),
            "away_changed": masked_samples(observed & ~toward & changed, values),
        }

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def check_to_variable_messages(self, edge_values):
        """Calculate min-sum check-to-variable messages."""
        edge_signs = np.where(edge_values < 0, -1, 1)
        edge_magnitudes = np.abs(edge_values)

        check_signs = np.multiply.reduceat(
            edge_signs,
            self.check_offsets[:-1],
        )
        first_minima = np.minimum.reduceat(
            edge_magnitudes,
            self.check_offsets[:-1],
        )
        is_first_minimum = (
            edge_magnitudes == first_minima[self.edge_cn]
        )
        first_minimum_counts = np.add.reduceat(
            is_first_minimum,
            self.check_offsets[:-1],
        )
        second_minima = np.minimum.reduceat(
            np.where(is_first_minimum, np.inf, edge_magnitudes),
            self.check_offsets[:-1],
        )

        use_second_minimum = (
            is_first_minimum
            & (first_minimum_counts[self.edge_cn] == 1)
        )
        extrinsic_magnitudes = np.where(
            use_second_minimum,
            second_minima[self.edge_cn],
            first_minima[self.edge_cn],
        )
        extrinsic_signs = check_signs[self.edge_cn] * edge_signs
        return extrinsic_signs * extrinsic_magnitudes

    @staticmethod
    def channel_bipolar_direction(y, state, channel_weight, bipolar_weight):
        """Negative gradient of the channel distance and bipolar penalty."""
        return (
            channel_weight * (y - state)
            - 4.0 * bipolar_weight * state * (state * state - 1.0)
        )

    def update_state(self, y, x, outgoing, iteration, parameters=None):
        parameters = parameters or {
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "channel_weight": self.channel_weight,
            "bipolar_weight": self.bipolar_weight,
        }
        incoming = self.check_to_variable_messages(outgoing)
        check_total = np.bincount(
            self.edge_vn, weights=incoming, minlength=self.block_length,
        )
        x_direction = self.channel_bipolar_direction(
            y, x, parameters["channel_weight"], parameters["bipolar_weight"],
        ) + check_total
        q_direction = self.channel_bipolar_direction(
            y[self.edge_vn], outgoing,
            parameters["channel_weight"], parameters["bipolar_weight"],
        ) + check_total[self.edge_vn] - incoming
        eta = parameters["learning_rate"] / np.sqrt(1 + parameters["learning_rate_decay"] * iteration)
        next_q = outgoing + eta * q_direction
        next_x = x + eta * x_direction
        if not np.all(np.isfinite(next_q)) or not np.all(np.isfinite(next_x)):
            raise FloatingPointError("Non-finite GDMS state")
        return next_x, next_q

    def decode(self, llr_in, llr_out, rng=None):
        y = llr_in.astype(np.float64, copy=True)
        parameters = {"learning_rate": self.learning_rate,
                      "learning_rate_decay": self.learning_rate_decay,
                      "channel_weight": self.channel_weight,
                      "bipolar_weight": self.bipolar_weight}
        state = self.initial_state(y, parameters)

        for iteration in range(self.n_iterations): # iteration loop
            if self.is_decoded(state):
                llr_out[:] = state["x"]
                return iteration # exit the iteration loop;

            state = self.step_state(state, y, parameters, iteration, rng).fields

        llr_out[:] = state["x"]
        return self.n_iterations
