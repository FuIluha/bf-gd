"""MPGD: alternating projected gradient steps on CN and VN objectives."""

import numpy as np

from .bin_ldpc import BinLdpcDecoderBase
from visualisation.error_by_energy_visualisation.base import VisualizableDecoderBase, masked_samples
from visualisation.error_by_energy_visualisation.models import (
    CategoryGroupSpec, CategorySpec, DecoderSpec, ObservableSpec, ParameterSpec, StepResult,
)


class BinLdpcMpgdDecoder(BinLdpcDecoderBase, VisualizableDecoderBase):
    """L2-regularized edge opinions, copied CN -> VN -> CN after each phase.

    These are bounded real opinions, not sum-product LLRs. Fixed projected steps
    need not increase either objective; no line search or momentum is applied.
    """

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self._reject_entropy_parameters(kwargs)
        defaults = self.describe().defaults()
        self.parameters = self.validate_parameters({
            key: kwargs.get(key, value) for key, value in defaults.items()
        })
        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)
        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)
        self.edges_count = len(self.edge_vn)
        degrees = np.bincount(self.edge_cn, minlength=self.n_checks)
        self.check_offsets = np.concatenate(([0], np.cumsum(degrees)))
        if np.any(degrees < 2):
            raise ValueError("MPGD requires check degree of at least two")
        if np.any(np.bincount(self.edge_vn, minlength=self.block_length) == 0):
            raise ValueError("MPGD requires every variable to have an incident edge")

    @staticmethod
    def _reject_entropy_parameters(parameters):
        if "tau_r" in parameters or "tau_v" in parameters:
            raise ValueError(
                "MPGD now uses L2: replace entropy parameters tau_r/tau_v with "
                "l2_r/l2_v and retune their values"
            )

    @classmethod
    def validate_parameters(cls, parameters):
        cls._reject_entropy_parameters(parameters)
        return super().validate_parameters(parameters)

    @classmethod
    def describe(cls):
        return DecoderSpec(
            key="mpgd", title="MPGD (L2)",
            parameters=(
                ParameterSpec("alpha", "alpha", "float", 1.0, 0.0),
                ParameterSpec("eta_r", "eta_R (CN)", "float", 0.1, 1e-12),
                ParameterSpec("eta_v", "eta_V (VN)", "float", 0.1, 1e-12),
                ParameterSpec("l2_r", "lambda_R (L2 CN)", "float", 0.1, 0.0),
                ParameterSpec("l2_v", "lambda_V (L2 VN)", "float", 0.1, 0.0),
                ParameterSpec("max_value", "Максимальный модуль сообщения", "float", 1.0, 1e-12),
            ),
            observables=(
                ObservableSpec("step", "Изменение s за цикл", True),
                ObservableSpec("cn_step", "Изменение в фазе CN", True),
                ObservableSpec("vn_step", "Изменение в фазе VN", True),
            ),
            categories=(
                CategorySpec("toward", "Шаг к истинному знаку", "#2e7d32"),
                CategorySpec("away", "Шаг от истинного знака", "#c62828"),
                CategorySpec("stationary", "Нулевой шаг", "#757575"),
            ),
            category_groups=(CategoryGroupSpec(
                "direction", "Направление изменения сообщения", ("toward", "away", "stationary"),
            ),),
        )

    def check_gradient(self, s):
        """Products excluding each recipient, including when inputs are zero."""
        result = np.empty_like(s)
        for start, end in zip(self.check_offsets[:-1], self.check_offsets[1:]):
            values = s[start:end]
            prefix = np.concatenate(([1.0], np.cumprod(values[:-1])))
            suffix = np.concatenate((np.cumprod(values[:0:-1])[::-1], [1.0]))
            result[start:end] = prefix * suffix
        return result

    def variable_totals(self, values):
        return np.bincount(self.edge_vn, weights=values, minlength=self.block_length)

    def objectives(self, s, u, received, parameters=None):
        """Local CN/VN scores, useful for mathematical checks and analysis."""
        p = self.parameters if parameters is None else parameters
        totals = self.variable_totals(u)
        r = np.multiply.reduceat(s, self.check_offsets[:-1]).sum()
        r -= 0.5 * p["l2_r"] * np.dot(s, s)
        v = p["alpha"] * np.dot(np.asarray(received, dtype=np.float64), totals)
        v += 0.5 * (np.dot(totals, totals) - np.dot(u, u))
        v -= 0.5 * p["l2_v"] * np.dot(u, u)
        return float(r), float(v)

    def initial_state(self, received, parameters):
        y = np.asarray(received, dtype=np.float64)
        if y.shape != (self.block_length,) or not np.all(np.isfinite(y)):
            raise ValueError("MPGD input must be a finite word of block_length values")
        channel = parameters["alpha"] * y
        if not np.all(np.isfinite(channel)):
            raise FloatingPointError("Non-finite MPGD channel field")
        limit = parameters["max_value"]
        s = np.clip(np.tanh(channel[self.edge_vn] / 2.0), -limit, limit)
        return {"s": s, "u": s.copy(), "x": y.copy()}

    def step_state(self, state, received, parameters, iteration, rng):
        p = parameters
        s = state["s"]
        limit = p["max_value"]
        direction_r = self.check_gradient(s) - p["l2_r"] * s
        raw_u = s + p["eta_r"] * direction_r
        if not np.all(np.isfinite(raw_u)):
            raise FloatingPointError("Non-finite MPGD CN step")
        u = np.clip(raw_u, -limit, limit)

        direction_v = (
            p["alpha"] * np.asarray(received, dtype=np.float64)[self.edge_vn]
            + self.variable_totals(u)[self.edge_vn] - u - p["l2_v"] * u
        )
        raw_s = u + p["eta_v"] * direction_v
        if not np.all(np.isfinite(raw_s)):
            raise FloatingPointError("Non-finite MPGD VN step")
        next_s = np.clip(raw_s, -limit, limit)
        x = self.variable_totals(next_s)
        # A zero vote follows the channel; x is never sent back to checks.
        x = np.where(x != 0, x, np.where(np.asarray(received) >= 0, 1.0, -1.0))
        return StepResult(
            {"s": next_s, "u": u, "x": x},
            {"step": next_s - s, "cn_step": u - s, "vn_step": next_s - u},
        )

    def hard_decision(self, state):
        return np.where(state["x"] >= 0, 1, -1).astype(np.int8)

    def is_decoded(self, state):
        products = np.multiply.reduceat(
            self.hard_decision(state)[self.edge_vn], self.check_offsets[:-1],
        )
        return bool(np.all(products == 1))

    def classify(self, before, after, diagnostics, active, transmitted):
        direction = diagnostics["step"] * transmitted[:, self.edge_vn]
        observed = np.broadcast_to(active[:, None], direction.shape)
        return {
            "toward": masked_samples(observed & (direction > 0), diagnostics),
            "away": masked_samples(observed & (direction < 0), diagnostics),
            "stationary": masked_samples(observed & (direction == 0), diagnostics),
        }

    def decode(self, llr_in, llr_out, rng=None):
        state = self.initial_state(llr_in, self.parameters)
        for iteration in range(self.n_iterations):
            if self.is_decoded(state):
                self._write_output(state["x"], llr_out)
                return iteration
            state = self.step_state(state, llr_in, self.parameters, iteration, rng).fields
        self._write_output(state["x"], llr_out)
        return self.n_iterations

    @staticmethod
    def _write_output(x, output):
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > np.finfo(output.dtype).max):
            raise FloatingPointError("MPGD decision exceeds the output floating-point range")
        output[:] = x
