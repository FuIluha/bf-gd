import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.algorithms import BatchDecoderEngine, StepRandom, TannerGraph
from visualisation.error_by_energy_visualisation.models import FrameBatch
from ldpc_py.bin_ldpc_ftgdbf import BinLdpcFtgdbfDecoder
from ldpc_py.bin_ldpc_gdms import BinLdpcGdmsDecoder
from ldpc_py.bin_ldpc_pmgdbf import BinLdpcPmgdbfDecoder


PCM = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
], dtype=np.uint8)


def batch():
    received = np.asarray([
        [-0.2, 1.1, 0.7, 1.0, 0.9, 0.8],
        [-0.8, 0.3, 0.9, -1.2, 1.1, 0.7],
        [1.2, 1.0, 0.8, 1.1, 0.6, 1.4],
    ], dtype=np.float32)
    return FrameBatch(received, np.ones(received.shape, dtype=np.int8))


def make_decoder(decoder_type, pcm=PCM, parameters=None, iterations=1):
    params = parameters or decoder_type.describe().defaults()
    return decoder_type(
        None, pcm=pcm, block_length=pcm.shape[1], n_checks=pcm.shape[0],
        n_iterations=iterations, is_systematic=False, **params,
    )


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.graph = TannerGraph.from_pcm(PCM)
        self.batch = batch()

    def test_ftgdbf_step_matches_direct_formula(self):
        decoder = make_decoder(BinLdpcFtgdbfDecoder)
        engine = BatchDecoderEngine(self.graph, self.batch, seed=4, workers=2)
        parameters = decoder.describe().defaults()
        state = engine.initial_state(decoder, parameters)
        next_state, observation = engine.step(decoder, state, parameters, 0)
        for frame in range(self.batch.frames):
            x = state.fields["x"][frame]
            syndrome = np.asarray([np.prod(x[check.astype(bool)]) for check in PCM])
            energy = parameters["alpha"] * x * self.batch.received[frame] + np.asarray([
                sum(syndrome[PCM[:, bit].astype(bool)]) for bit in range(PCM.shape[1])
            ])
            expected = x.copy()
            if not np.all(syndrome == 1):
                expected[energy <= 0] *= -1
            np.testing.assert_array_equal(next_state.fields["x"][frame], expected)
        self.assertEqual(set(observation.categories), {"correct", "incorrect"})

    def test_pmgdbf_is_independent_of_worker_count(self):
        decoder = make_decoder(BinLdpcPmgdbfDecoder)
        params = decoder.validate_parameters({"delta": 1, "alpha": 1.8, "p": 0.67,
                                              "rho": [2, 1, 0.5], "L": 3})
        one = BatchDecoderEngine(self.graph, self.batch, seed=11, workers=1)
        many = BatchDecoderEngine(self.graph, self.batch, seed=11, workers=3)
        state = one.initial_state(decoder, params)
        next_one, obs_one = one.step(decoder, state, params, 2)
        next_many, obs_many = many.step(decoder, state, params, 2)
        for key in state.fields:
            np.testing.assert_array_equal(next_one.fields[key], next_many.fields[key])
        for key in obs_one.categories:
            np.testing.assert_array_equal(obs_one.categories[key].frame_indices,
                                          obs_many.categories[key].frame_indices)

    def test_pmgdbf_age_sentinel_survives_l_change(self):
        decoder = make_decoder(BinLdpcPmgdbfDecoder)
        engine = BatchDecoderEngine(self.graph, self.batch, seed=11)
        small = decoder.validate_parameters({"delta": 0, "alpha": 1.8, "p": 1,
                                             "rho": [9, 8], "L": 2})
        state = engine.initial_state(decoder, small)
        next_state, _ = engine.step(decoder, state, small, 0)
        never = next_state.fields["ages"] > small["L"]
        large = decoder.validate_parameters({"delta": 1, "alpha": 1.8, "p": 1,
                                             "rho": [9, 8, 7, 6, 5, 4], "L": 6})
        # Direct plugin diagnostics are useful here; the general observation
        # contract exposes only classified samples.
        result = decoder.step_state(
            {key: value[0] for key, value in next_state.fields.items()},
            self.batch.received[0], large, 1, StepRandom(11, 1, 3).for_frame(0),
        )
        products = self.graph.check_products(next_state.fields["x"])
        incident = np.asarray([
            np.bincount(self.graph.edge_vn, weights=row[self.graph.edge_cn], minlength=self.graph.block_length)
            for row in products
        ])
        no_momentum = large["alpha"] * next_state.fields["x"] * self.batch.received + incident
        np.testing.assert_allclose(result.diagnostics["energy"][never[0]], no_momentum[0][never[0]])

    def test_gdms_edge_step_matches_scalar_formula(self):
        decoder = make_decoder(BinLdpcGdmsDecoder)
        engine = BatchDecoderEngine(self.graph, self.batch, seed=4, workers=2)
        params = decoder.describe().defaults()
        state = engine.initial_state(decoder, params)
        next_state, observation = engine.step(decoder, state, params, 0)
        for frame in np.flatnonzero(observation.decision_frames):
            q = state.fields["q"][frame]
            x = state.fields["x"][frame]
            r = []
            for edge, (check, variable) in enumerate(zip(self.graph.edge_cn, self.graph.edge_vn)):
                neighbors = np.flatnonzero(self.graph.edge_cn == check)
                others = q[neighbors[neighbors != edge]]
                r.append(np.prod(np.where(others < 0, -1, 1)) * np.min(np.abs(others)))
            r = np.asarray(r)
            g = params["alpha"] * self.batch.received[frame] + np.bincount(
                self.graph.edge_vn, weights=r, minlength=self.graph.block_length)
            expected_q = q + params["learning_rate"] * (
                g[self.graph.edge_vn] - r - params["l2"] * q)
            expected_x = x + params["learning_rate"] * (g - params["l2"] * x)
            np.testing.assert_allclose(next_state.fields["q"][frame], expected_q)
            np.testing.assert_allclose(next_state.fields["x"][frame], expected_x)
        self.assertEqual(set(observation.categories), {
            "toward_unchanged", "toward_changed", "away_unchanged", "away_changed"})
        self.assertEqual(sum(len(item.frame_indices) for item in observation.categories.values()),
                         int(observation.decision_frames.sum()) * self.graph.edge_vn.size)

    def test_gdms_sign_zero_is_positive(self):
        decoder = make_decoder(BinLdpcGdmsDecoder)
        x = np.zeros((1, self.graph.block_length))
        np.testing.assert_array_equal(decoder.hard_decision({"x": x}), np.ones_like(x))

    def test_decode_and_visualizer_share_the_same_step(self):
        for decoder_type in (BinLdpcFtgdbfDecoder, BinLdpcPmgdbfDecoder, BinLdpcGdmsDecoder):
            with self.subTest(decoder=decoder_type.__name__):
                decoder = make_decoder(decoder_type)
                engine = BatchDecoderEngine(self.graph, self.batch, seed=23, workers=2)
                parameters = decoder.describe().defaults()
                state = engine.initial_state(decoder, parameters)
                stepped, _ = engine.step(decoder, state, parameters, 0)
                source = StepRandom(23, 0, self.batch.frames)
                for frame, received in enumerate(self.batch.received):
                    output = np.empty(self.batch.block_length, dtype=np.float64)
                    decoder.decode(received, output, rng=source.for_frame(frame))
                    np.testing.assert_allclose(output, stepped.fields["x"][frame])

    def test_pmgdbf_decode_matches_original_fixed_parameter_loop(self):
        parameters = {"delta": 1.1, "alpha": 0.45, "p": 0.96,
                      "rho": [0.5, 0.5, 0.25], "L": 3}
        decoder = make_decoder(BinLdpcPmgdbfDecoder, parameters=parameters, iterations=4)
        received = self.batch.received[0]
        actual = np.empty(self.batch.block_length, dtype=np.float64)
        actual_iterations = decoder.decode(received, actual, rng=np.random.default_rng(29))

        x = np.where(received >= 0, 1, -1).astype(np.int8)
        ages = np.repeat(parameters["L"] + 1, self.batch.block_length)
        rho = np.concatenate((np.asarray(parameters["rho"], dtype=np.float32),
                              np.asarray([0], dtype=np.float32)))
        rng = np.random.default_rng(29)
        expected_iterations = 4
        for iteration in range(4):
            checks = decoder.bpsk_syndrome(x)
            if np.all(checks == 1):
                expected_iterations = iteration
                break
            incident = np.bincount(self.graph.edge_vn, weights=checks[self.graph.edge_cn],
                                   minlength=self.graph.block_length)
            ages = np.minimum(ages, parameters["L"]) + 1
            energy = parameters["alpha"] * x * received + incident + rho[ages - 1]
            threshold = np.min(energy) + parameters["delta"]
            flip = (energy <= threshold) & (rng.random(self.batch.block_length) < parameters["p"])
            x[flip] *= -1
            ages[flip] = 0
        self.assertEqual(actual_iterations, expected_iterations)
        np.testing.assert_array_equal(actual, x)


if __name__ == "__main__":
    unittest.main()
