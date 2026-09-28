"""E-GDBF V3: step q by the extrinsic gradient and round the result to +/-1."""

import numpy as np

from .bin_ldpc_egdbf import BinLdpcEgdbfDecoder


class GradientStepRounding:
    """Update rule shared by the Python and C++ V3 wrappers."""

    def __init__(self, alist_filename, **kwargs):
        for name in ("delta", "theta", "rho", "L"):
            if name in kwargs:
                raise ValueError(f"E-GDBF V3 has no parameter {name!r}")
        self.eta = float(kwargs["eta"])
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        super().__init__(alist_filename, rho=[], L=0, **kwargs)

    def flip_candidates(
        self,
        y,
        variable_messages,
        check_messages,
        incoming_sums,
        ages,
    ):
        """q+ = sign(q + eta * G[n->m]) differs from q; u = 0 keeps q."""
        score = (
            self.alpha * y[self.edge_vn]
            + incoming_sums[self.edge_vn]
            - check_messages
        )
        step_result = variable_messages + self.eta * score
        return variable_messages * step_result < 0

    def rule_parameter(self):
        return self.eta


class BinLdpcEgdbfV3Decoder(GradientStepRounding, BinLdpcEgdbfDecoder):
    """Hard edge messages updated by a rounded gradient step."""
