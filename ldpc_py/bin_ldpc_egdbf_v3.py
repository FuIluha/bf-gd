"""E-GDBF V3: rounded gradient step with momentum on every edge."""

import numpy as np

from .bin_ldpc_egdbf import BinLdpcEgdbfDecoder


class GradientStepRounding:
    """Update rule shared by the Python and C++ V3 wrappers.

    s = beta * s + (1 - beta) * G[n->m] is the unrounded step kept on the
    edge; q = sign(q + eta * s), and a zero result keeps q.
    """

    def __init__(self, alist_filename, **kwargs):
        for name in ("delta", "theta", "rho", "L"):
            if name in kwargs:
                raise ValueError(f"E-GDBF V3 has no parameter {name!r}")
        self.eta = float(kwargs["eta"])
        self.beta = float(kwargs.get("beta", 0.0))
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not np.isfinite(self.beta) or not 0 <= self.beta < 1:
            raise ValueError("beta must be finite and in [0, 1)")
        super().__init__(alist_filename, rho=[], L=0, **kwargs)
        self.steps = np.zeros(self.edges_count, dtype=np.float64)

    def decode(self, llr_in, llr_out, rng=None):
        self.steps[:] = 0.0
        return super().decode(llr_in, llr_out, rng)

    def flip_candidates(
        self,
        y,
        variable_messages,
        check_messages,
        incoming_sums,
        ages,
    ):
        score = (
            self.alpha * y[self.edge_vn]
            + incoming_sums[self.edge_vn]
            - check_messages
        )
        self.steps *= self.beta
        self.steps += (1.0 - self.beta) * score
        position = variable_messages + self.eta * self.steps
        return variable_messages * position < 0

    def rule_arguments(self):
        return self.alpha, self.eta, self.beta, self.p


class BinLdpcEgdbfV3Decoder(GradientStepRounding, BinLdpcEgdbfDecoder):
    """Hard edge messages updated by a rounded momentum step."""
