"""A metadata and step contract mixed into an existing Python decoder.

No numerical update lives here: ``decode`` and the UI must call the same
decoder-owned ``step_state`` implementation.
"""

from abc import ABC, abstractmethod

import numpy as np

from .models import CategorySamples, DecoderSpec


class VisualizableDecoderBase(ABC):
    """Abstract visualisation contract, without algorithm implementation."""

    @classmethod
    @abstractmethod
    def describe(cls) -> DecoderSpec:
        """Parameters, X values, categories, decomposition groups and labels."""

    @classmethod
    def validate_parameters(cls, parameters):
        """Cross-field validation; override if individual specs are insufficient."""
        return cls.describe().parse_parameters(parameters)

    @abstractmethod
    def initial_state(self, received, parameters):
        """Return named NumPy arrays for one received word."""

    @abstractmethod
    def step_state(self, state, received, parameters, iteration, rng):
        """Return StepResult for one word; decode and UI both call this."""

    @abstractmethod
    def hard_decision(self, state):
        """Return one BPSK sign (-1/+1) per bit for BER/FER and parity checks."""

    @abstractmethod
    def is_decoded(self, state):
        """Return whether one word satisfies the decoder's stopping rule."""

    @abstractmethod
    def classify(self, before, after, diagnostics, active, transmitted):
        """Classify a batch of active words; overlaps are permitted."""


def masked_samples(mask, values):
    """Build samples from a frame-by-entity mask and named X-value arrays."""
    frame_indices, entity_indices = np.nonzero(mask)
    return CategorySamples(
        frame_indices=frame_indices.astype(np.int32),
        entity_indices=entity_indices.astype(np.int32),
        values={key: np.asarray(value)[mask] for key, value in values.items()},
    )
