import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.algorithms import (
    BatchDecoderEngine,
    TannerGraph,
    calculate_metrics,
    initial_state,
)
from visualisation.error_by_energy_visualisation.models import (
    Algorithm,
    FrameBatch,
    FtgdbfParameters,
    PmgdbfParameters,
)


PCM = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
], dtype=np.uint8)


def batch():
    transmitted = np.asarray([
        [1, 1, 1, 1, 1, 1],
        [-1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int8)
    received = np.asarray([
        [-0.2, 1.1, 0.7, 1.0, 0.9, 0.8],
        [-0.8, 0.3, 0.9, -1.2, 1.1, 0.7],
        [1.2, 1.0, 0.8, 1.1, 0.6, 1.4],
    ], dtype=np.float32)
    return FrameBatch(received=received, transmitted_symbols=transmitted)


class TannerGraphTests(unittest.TestCase):
    def test_batch_products_and_incident_sums_match_scalar_math(self):
        graph = TannerGraph.from_pcm(PCM)
        values = np.asarray([
            [1, -1, 1, -1, 1, -1],
            [-1, -1, 1, -1, 1, 1],
        ], dtype=np.int8)
        products = graph.check_products(values)
        expected_products = np.asarray([
            [np.prod(row[check.astype(bool)]) for check in PCM]
            for row in values
        ], dtype=np.int8)
        np.testing.assert_array_equal(products, expected_products)

        sums = graph.incident_check_sums(products)
        expected_sums = np.asarray([
            [
                sum(expected_products[frame, PCM[:, bit].astype(bool)])
                for bit in range(PCM.shape[1])
            ]
            for frame in range(values.shape[0])
        ])
        np.testing.assert_array_equal(sums, expected_sums)


class DecoderEngineTests(unittest.TestCase):
    def test_ftgdbf_step_matches_direct_formula(self):
        frames = batch()
        graph = TannerGraph.from_pcm(PCM)
        state = initial_state(frames, Algorithm.FTGDBF, momentum_length=7)
        parameters = FtgdbfParameters(alpha=1.8)
        next_state, observation = BatchDecoderEngine(
            graph, frames, seed=4, workers=2
        ).step(state, parameters, iteration=0)

        for frame in range(frames.frames):
            x = state.x[frame]
            syndrome = np.asarray([
                np.prod(x[check.astype(bool)]) for check in PCM
            ])
            decoded = np.all(syndrome == 1)
            energy = parameters.alpha * x * frames.received[frame]
            energy += np.asarray([
                sum(syndrome[PCM[:, bit].astype(bool)])
                for bit in range(PCM.shape[1])
            ])
            expected = x.copy()
            if not decoded:
                expected[energy <= 0] *= -1
            np.testing.assert_allclose(observation.energy[frame], energy)
            np.testing.assert_array_equal(next_state.x[frame], expected)

    def test_pmgdbf_random_mask_is_independent_of_worker_count(self):
        frames = batch()
        graph = TannerGraph.from_pcm(PCM)
        state = initial_state(frames, Algorithm.PMGDBF, momentum_length=3)
        parameters = PmgdbfParameters(
            delta=1.0,
            alpha=1.8,
            p=0.67,
            rho=(2.0, 1.0, 0.5),
            L=3,
        )
        one = BatchDecoderEngine(graph, frames, seed=11, workers=1)
        many = BatchDecoderEngine(graph, frames, seed=11, workers=3)
        next_one, observation_one = one.step(state, parameters, iteration=2)
        next_many, observation_many = many.step(state, parameters, iteration=2)
        np.testing.assert_array_equal(next_one.x, next_many.x)
        np.testing.assert_array_equal(observation_one.flip_mask, observation_many.flip_mask)
        np.testing.assert_allclose(observation_one.energy, observation_many.energy)

    def test_pmgdbf_momentum_age_survives_runtime_l_change(self):
        frames = batch()
        graph = TannerGraph.from_pcm(PCM)
        state = initial_state(frames, Algorithm.PMGDBF, momentum_length=2)
        parameters = PmgdbfParameters(
            delta=0.0,
            alpha=1.8,
            p=1.0,
            rho=(9.0, 8.0),
            L=2,
        )
        next_state, _ = BatchDecoderEngine(
            graph, frames, seed=11, workers=1
        ).step(state, parameters, iteration=0)

        # Flipped bits start at age 0. All others retain the "never flipped"
        # sentinel and must not acquire momentum merely because L later grows.
        never_flipped = next_state.ages > parameters.L
        grown = PmgdbfParameters(
            delta=1.0,
            alpha=1.8,
            p=1.0,
            rho=(9.0, 8.0, 7.0, 6.0, 5.0, 4.0),
            L=6,
        )
        _, observation = BatchDecoderEngine(
            graph, frames, seed=11, workers=1
        ).step(next_state, grown, iteration=1)
        check_products = graph.check_products(next_state.x)
        no_momentum_energy = (
            grown.alpha * next_state.x * frames.received
            + graph.incident_check_sums(check_products)
        )
        np.testing.assert_allclose(
            observation.energy[never_flipped],
            no_momentum_energy[never_flipped],
        )

    def test_metrics_use_transmitted_symbols(self):
        frames = batch()
        signs = np.where(frames.received >= 0, 1, -1)
        metrics = calculate_metrics(signs, frames.transmitted_symbols)
        expected = signs != frames.transmitted_symbols
        self.assertEqual(metrics.bit_errors, int(expected.sum()))
        self.assertEqual(metrics.frame_errors, int(np.any(expected, axis=1).sum()))


if __name__ == "__main__":
    unittest.main()
