import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.algorithms import BatchDecoderEngine, StepRandom, TannerGraph
from visualisation.error_by_energy_visualisation.models import FrameBatch
from ldpc_py.bin_ldpc_ftgdbf import BinLdpcFtgdbfDecoder
from ldpc_py.bin_ldpc_gdms import BinLdpcGdmsDecoder
from ldpc_py.bin_ldpc_egdbf import BinLdpcEgdbfDecoder
from ldpc_py.cpp_bin_ldpc_egdbf import CppBinLdpcEgdbfDecoder
from ldpc_py.bin_ldpc_egdbf_v2 import BinLdpcEgdbfV2Decoder
from ldpc_py.cpp_bin_ldpc_egdbf_v2 import CppBinLdpcEgdbfV2Decoder
from ldpc_py.decoder_factory import create_decoder
from ldpc_py.bin_ldpc_pmgdbf import BinLdpcPmgdbfDecoder
from ldpc_py.bin_ldpc_tgdbf import BinLdpcTgdbfDecoder


PCM = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
], dtype=np.uint8)

EGDBF_RECEIVED = np.asarray([
    [-1.7657658, 1.1919596, 1.6477470, 0.3385023, 0.4917809, 2.4136950],
    [-0.7954280, 0.5324225, 1.4048856, -0.3198107, 1.3699480, 0.2621832],
], dtype=np.float32)


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

    def test_l_zero_disables_momentum(self):
        for decoder_type in (BinLdpcTgdbfDecoder, BinLdpcPmgdbfDecoder):
            with self.subTest(decoder=decoder_type.__name__):
                raw = decoder_type.describe().defaults()
                raw.update({"L": 0, "rho": [100, 100, 100]})
                if "p" in raw:
                    raw["p"] = 1.0
                parameters = decoder_type.validate_parameters(raw)
                self.assertEqual(parameters["rho"], [100.0, 100.0, 100.0])
                decoder = make_decoder(decoder_type, parameters=parameters)
                received = self.batch.received[0]
                state = decoder.initial_state(received, parameters)
                first = decoder.step_state(
                    state, received, parameters, 0, np.random.default_rng(1),
                )
                second = decoder.step_state(
                    first.fields, received, parameters, 1, np.random.default_rng(2),
                )
                x = first.fields["x"]
                checks = decoder.bpsk_syndrome(x)
                incident = np.bincount(
                    self.graph.edge_vn, weights=checks[self.graph.edge_cn],
                    minlength=self.graph.block_length,
                )
                energy = parameters["alpha"] * x * received + incident
                expected_margin = energy - np.min(energy) - (
                    parameters["delta"][1]
                    if decoder_type is BinLdpcTgdbfDecoder else parameters["delta"]
                )
                np.testing.assert_allclose(second.diagnostics["margin"], expected_margin)

    def test_tgdbf_bit_state_categories_overlap_action_categories(self):
        decoder = make_decoder(BinLdpcTgdbfDecoder)
        engine = BatchDecoderEngine(self.graph, self.batch, seed=4)
        parameters = decoder.describe().defaults()
        state = engine.initial_state(decoder, parameters)
        _, observation = engine.step(decoder, state, parameters, 0)

        self.assertEqual(set(observation.categories), {
            "correct", "incorrect", "bit_error", "bit_correct",
        })
        active_bits = int(observation.decision_frames.sum()) * self.graph.block_length
        self.assertEqual(
            len(observation.categories["bit_error"].frame_indices)
            + len(observation.categories["bit_correct"].frame_indices),
            active_bits,
        )
        self.assertEqual(
            len(observation.categories["correct"].frame_indices)
            + len(observation.categories["incorrect"].frame_indices),
            active_bits,
        )

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

    def test_egdbf_without_momentum_matches_zero_profile(self):
        for decoder_type in (BinLdpcEgdbfDecoder, CppBinLdpcEgdbfDecoder):
            for dtype in (np.float32, np.float64):
                with self.subTest(decoder=decoder_type.__name__, dtype=dtype):
                    common = dict(
                        pcm=PCM, block_length=PCM.shape[1], n_checks=PCM.shape[0],
                        n_iterations=5, is_systematic=False, alpha=1.8,
                    )
                    disabled = decoder_type(None, L=0, rho=[], **common)
                    zero_profile = decoder_type(None, L=3, rho=[0, 0, 0], **common)
                    for received in EGDBF_RECEIVED.astype(dtype):
                        actual = np.empty_like(received)
                        expected = np.empty_like(received)
                        actual_iterations = disabled.decode(received, actual)
                        expected_iterations = zero_profile.decode(received, expected)
                        self.assertGreater(actual_iterations, 0)
                        self.assertEqual(actual_iterations, expected_iterations)
                        np.testing.assert_array_equal(actual, expected)

    def test_egdbf_momentum_parameters_are_consistent(self):
        common = dict(
            pcm=PCM, block_length=PCM.shape[1], n_checks=PCM.shape[0],
            n_iterations=1, is_systematic=False, alpha=0.45,
        )
        for decoder_type in (BinLdpcEgdbfDecoder, CppBinLdpcEgdbfDecoder):
            with self.subTest(decoder=decoder_type.__name__):
                for params in (
                    {"L": -1, "rho": []}, {"L": 0, "rho": [1]},
                    {"L": 0, "rho": [], "delta": -0.1},
                    {"L": 0, "rho": [], "p": -0.1},
                    {"L": 0, "rho": [], "p": 1.1},
                ):
                    with self.assertRaises(ValueError):
                        decoder_type(None, **common, **params)

    def test_egdbf_threshold_matches_scalar_reference(self):
        edge_cn, edge_vn = np.nonzero(PCM)

        def reference(received, alpha, delta, rho, iterations, mean_threshold):
            channel = np.where(received >= 0, 1, -1)
            q = channel[edge_vn].copy()
            ages = np.full(len(q), len(rho) + 1, dtype=int)
            for iteration in range(iterations + 1):
                messages = np.asarray([
                    np.prod(q[(edge_cn == check) & (np.arange(len(q)) != edge)])
                    for edge, check in enumerate(edge_cn)
                ])
                full = alpha * received + np.asarray([
                    sum(messages[edge_vn == bit]) for bit in range(len(received))
                ])
                word = np.where(full > 0, 1, np.where(full < 0, -1, channel))
                if iteration == iterations or np.all([
                    np.prod(word[row.astype(bool)]) == 1 for row in PCM
                ]):
                    return iteration, word
                if rho:
                    ages = np.minimum(ages, len(rho)) + 1
                energies = np.asarray([
                    q[edge] * (alpha * received[bit] + sum(
                        messages[(edge_vn == bit) & (np.arange(len(q)) != edge)]
                    )) + (rho[ages[edge] - 1] if rho and 1 <= ages[edge] <= len(rho) else 0)
                    for edge, bit in enumerate(edge_vn)
                ])
                if mean_threshold:
                    minimum = min(
                        np.mean(energies[edge_vn == bit])
                        for bit in range(len(received))
                    )
                else:
                    minimum = min(energies)
                flipped = energies <= minimum + delta
                q[flipped] *= -1
                if rho:
                    ages[flipped] = 0

        for decoder_type, mean_threshold in (
            (BinLdpcEgdbfDecoder, False),
            (CppBinLdpcEgdbfDecoder, False),
            (BinLdpcEgdbfV2Decoder, True),
            (CppBinLdpcEgdbfV2Decoder, True),
        ):
            for params in (
                {"L": 0, "rho": [], "delta": 0.0},
                {"L": 3, "rho": [0.5, 0.25, 0.1], "delta": 0.4},
            ):
                with self.subTest(decoder=decoder_type.__name__, params=params):
                    decoder = decoder_type(
                        None, pcm=PCM, block_length=PCM.shape[1],
                        n_checks=PCM.shape[0], n_iterations=5,
                        is_systematic=False, alpha=1.8, **params,
                    )
                    for received in EGDBF_RECEIVED:
                        output = np.empty_like(received)
                        actual_iterations = decoder.decode(received, output)
                        expected_iterations, expected = reference(
                            received, 1.8, params["delta"], params["rho"],
                            5, mean_threshold,
                        )
                        self.assertGreater(actual_iterations, 0)
                        self.assertEqual(actual_iterations, expected_iterations)
                        np.testing.assert_array_equal(output, expected)

    def test_egdbf_v2_threshold_uses_bit_means(self):
        common = dict(
            pcm=PCM, block_length=PCM.shape[1], n_checks=PCM.shape[0],
            n_iterations=1, is_systematic=False, alpha=1.8,
            delta=0.25, L=0, rho=[],
        )
        v1 = BinLdpcEgdbfDecoder(None, **common)
        v2 = BinLdpcEgdbfV2Decoder(None, **common)
        for decoder_type in (BinLdpcEgdbfV2Decoder, CppBinLdpcEgdbfV2Decoder):
            with self.assertRaises(ValueError):
                decoder_type(None, **{**common, "p": 0.5})
        energies = np.full(v1.edges_count, 4.0)
        energies[np.flatnonzero(v1.edge_vn == 0)] = [-4.0, 4.0]
        energies[np.flatnonzero(v1.edge_vn == 1)] = [-2.0, -2.0]
        self.assertEqual(v1.energy_threshold(energies), -3.75)
        self.assertEqual(v2.energy_threshold(energies), -1.75)

        for algorithm, decoder_type in (
            ("edge-wise gradient descent bit-flipping v2", BinLdpcEgdbfV2Decoder),
            ("cpp edge-wise gradient descent bit-flipping v2", CppBinLdpcEgdbfV2Decoder),
        ):
            self.assertIsInstance(create_decoder(algorithm, None, **common), decoder_type)

    def test_egdbf_python_and_cpp_agree_with_momentum(self):
        common = dict(
            pcm=PCM, block_length=PCM.shape[1], n_checks=PCM.shape[0],
            n_iterations=5, is_systematic=False, alpha=1.8,
            L=3, rho=[0.5, 0.25, 0.1],
        )
        python_decoder = BinLdpcEgdbfDecoder(None, **common)
        cpp_decoder = CppBinLdpcEgdbfDecoder(None, **common)
        for received in EGDBF_RECEIVED:
            python_output = np.empty_like(received)
            cpp_output = np.empty_like(received)
            python_iterations = python_decoder.decode(received, python_output)
            cpp_iterations = cpp_decoder.decode(received, cpp_output)
            self.assertEqual(python_iterations, cpp_iterations)
            np.testing.assert_array_equal(python_output, cpp_output)

    def test_egdbf_probabilistic_flips_match_python_and_cpp(self):
        common = dict(
            pcm=PCM, block_length=PCM.shape[1], n_checks=PCM.shape[0],
            n_iterations=20, is_systematic=False, alpha=1.8,
            delta=0.4, L=0, rho=[],
        )
        for probability in (0.0, 0.3, 1.0):
            python_decoder = BinLdpcEgdbfDecoder(None, p=probability, **common)
            cpp_decoder = CppBinLdpcEgdbfDecoder(None, p=probability, **common)
            for dtype in (np.float32, np.float64):
                for received in EGDBF_RECEIVED.astype(dtype):
                    for seed in range(4):
                        with self.subTest(p=probability, dtype=dtype, seed=seed):
                            python_rng = np.random.default_rng(seed)
                            cpp_rng = np.random.default_rng(seed)
                            python_output = np.empty_like(received)
                            cpp_output = np.empty_like(received)
                            python_iterations = python_decoder.decode(
                                received, python_output, python_rng,
                            )
                            cpp_iterations = cpp_decoder.decode(
                                received, cpp_output, cpp_rng,
                            )
                            self.assertEqual(python_iterations, cpp_iterations)
                            np.testing.assert_array_equal(python_output, cpp_output)
                            self.assertEqual(
                                python_rng.bit_generator.random_raw(),
                                cpp_rng.bit_generator.random_raw(),
                            )

        received = EGDBF_RECEIVED[1]
        outputs = []
        for probability in (0.0, 1.0):
            decoder = BinLdpcEgdbfDecoder(
                None, p=probability, **{**common, "n_iterations": 3},
            )
            output = np.empty_like(received)
            decoder.decode(received, output)
            outputs.append(output)
        self.assertFalse(np.array_equal(*outputs))

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
