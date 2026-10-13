"""MPGD objective, zero-input, native parity, and tuning regression checks."""

import itertools
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ldpc_py.bin_ldpc_mpgd import BinLdpcMpgdDecoder
from ldpc_py.cpp_bin_ldpc_mpgd import CppBinLdpcMpgdDecoder
from ldpc_py.decoder_factory import create_decoder
from tuning import tune_mpgd
from visualisation.error_by_energy_visualisation.algorithms import BatchDecoderEngine, TannerGraph
from visualisation.error_by_energy_visualisation.models import FrameBatch
from visualisation.error_by_energy_visualisation.supported import decoder_catalog


PCM = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
], dtype=np.uint8)


def decoder(cls=BinLdpcMpgdDecoder, iterations=5, **params):
    return cls(None, pcm=PCM, block_length=6, n_checks=3,
               is_systematic=False, n_iterations=iterations, **params)


class MpgdTests(unittest.TestCase):
    def test_objective_gradients_by_finite_differences(self):
        dec = decoder(alpha=0.7, l2_r=0.3, l2_v=0.6, max_value=2.0)
        p = dec.parameters
        rng = np.random.default_rng(81)
        s, u = rng.uniform(-1.5, 1.5, (2, dec.edges_count))
        y = rng.normal(size=6)
        expected_r = dec.check_gradient(s) - p["l2_r"] * s
        expected_v = (p["alpha"] * y[dec.edge_vn]
                      + dec.variable_totals(u)[dec.edge_vn] - u
                      - p["l2_v"] * u)
        for edge in range(dec.edges_count):
            plus, minus = s.copy(), s.copy()
            plus[edge] += 1e-5
            minus[edge] -= 1e-5
            derivative = (dec.objectives(plus, u, y)[0] - dec.objectives(minus, u, y)[0]) / 2e-5
            self.assertAlmostEqual(derivative, expected_r[edge], places=8)
            plus, minus = u.copy(), u.copy()
            plus[edge] += 1e-5
            minus[edge] -= 1e-5
            derivative = (dec.objectives(s, plus, y)[1] - dec.objectives(s, minus, y)[1]) / 2e-5
            self.assertAlmostEqual(derivative, expected_v[edge], places=8)

    def test_synchronous_phases_match_scalar_reference(self):
        dec = decoder(alpha=0.7, eta_r=0.3, eta_v=0.2, l2_r=0.4, l2_v=0.6, max_value=1.25)
        p = dec.parameters
        y = np.array([-0.2, 1.1, 0.7, 1.0, 0.9, 0.8])
        state = dec.initial_state(y, p)
        state["s"] = np.array([0.0, -0.5, 0.0, 0.6, -0.3, 1.2, 0.2, -0.7, 0.0, 0.9])
        original = {key: value.copy() for key, value in state.items()}
        u = np.empty(dec.edges_count)
        for e, c in enumerate(dec.edge_cn):
            others = [state["s"][j] for j in range(dec.edges_count) if dec.edge_cn[j] == c and j != e]
            u[e] = np.clip(state["s"][e] + p["eta_r"] * (
                math.prod(others) - p["l2_r"] * state["s"][e]), -p["max_value"], p["max_value"])
        expected = np.empty_like(u)
        for e, v in enumerate(dec.edge_vn):
            other_sum = sum(u[j] for j in range(dec.edges_count) if dec.edge_vn[j] == v and j != e)
            expected[e] = np.clip(u[e] + p["eta_v"] * (
                p["alpha"] * y[v] + other_sum - p["l2_v"] * u[e]), -p["max_value"], p["max_value"])
        actual = dec.step_state(state, y, p, 0, None)
        np.testing.assert_allclose(actual.fields["u"], u, atol=1e-15)
        np.testing.assert_allclose(actual.fields["s"], expected, atol=1e-15)
        for key in state:
            np.testing.assert_array_equal(state[key], original[key])
        expected_vote = np.array([sum(expected[dec.edge_vn == v]) for v in range(6)])
        np.testing.assert_allclose(actual.fields["x"], expected_vote)

    def test_python_cpp_agree_with_different_bounds_and_zero_l2(self):
        rng = np.random.default_rng(92)
        words = np.vstack((rng.normal(size=(12, 6)), np.zeros(6),
                           [-1e6, 1e6, 0.0, 1.0, -1.0, 0.01]))
        for params in (
            {}, {"alpha": 0.0, "l2_r": 0.0, "l2_v": 0.0},
            {"alpha": 2.0, "eta_r": 1.5, "eta_v": 0.7, "l2_r": 4.0, "l2_v": 2.0, "max_value": 0.5},
            {"alpha": 0.75, "eta_r": 1.75, "eta_v": 0.8, "l2_r": 0.03, "l2_v": 0.3, "max_value": 2.0},
            {"l2_r": 0.0, "l2_v": 0.0, "max_value": 1e6},
        ):
            for limit in (1, 3, 20):
                py = decoder(iterations=limit, **params)
                cpp = decoder(CppBinLdpcMpgdDecoder, iterations=limit, **params)
                for dtype in (np.float32, np.float64):
                    for y in words.astype(dtype):
                        with self.subTest(params=params, limit=limit, dtype=dtype):
                            a, b = np.empty_like(y), np.empty_like(y)
                            self.assertEqual(py.decode(y, a), cpp.decode(y, b))
                            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                            # Reusing a decoder must not retain the previous word's state.
                            again = np.empty_like(y)
                            cpp.decode(y, again)
                            np.testing.assert_array_equal(b, again)

    def test_configurable_bound_applies_to_initialization_and_both_phases(self):
        y = np.array([-2.0, 1.1, 0.7, 1.0, 0.9, 0.8])
        dec = decoder(max_value=0.4, eta_r=5.0, eta_v=5.0, l2_r=0.0, l2_v=0.0)
        state = dec.initial_state(y, dec.parameters)
        self.assertLessEqual(np.max(np.abs(state["s"])), 0.4)
        after = dec.step_state(state, y, dec.parameters, 0, None).fields
        for key in ("s", "u"):
            self.assertLessEqual(np.max(np.abs(after[key])), 0.4)
        self.assertTrue(np.any(np.abs(after["s"]) == 0.4))

        dec = decoder(max_value=1e6, eta_r=10.0, eta_v=10.0, l2_r=0.0, l2_v=0.0)
        state = dec.initial_state(y, dec.parameters)
        state["s"][:] = 2.0
        after = dec.step_state(state, y, dec.parameters, 0, None).fields
        for key in ("s", "u"):
            self.assertGreater(np.max(np.abs(after[key])), 2.0)
            self.assertLess(np.max(np.abs(after[key])), 1e6)

    def test_output_overflow_is_reported_before_float32_conversion(self):
        y = np.array([-0.2, 1.1, 0.7, 1.0, 0.9, 0.8], dtype=np.float32)
        for cls in (BinLdpcMpgdDecoder, CppBinLdpcMpgdDecoder):
            dec = decoder(cls, iterations=1, max_value=1e50,
                          eta_r=1e20, eta_v=1e20, l2_r=0.0, l2_v=0.0)
            with self.assertRaises(FloatingPointError):
                dec.decode(y, np.empty_like(y))

    def test_codeword_sign_symmetry(self):
        bits = next(np.array(z) for z in itertools.product((0, 1), repeat=6)
                    if any(z) and not np.any(PCM @ z % 2))
        signs = 1 - 2 * bits
        y = np.array([-0.2, 1.1, 0.7, 1.0, 0.9, 0.8])
        for cls in (BinLdpcMpgdDecoder, CppBinLdpcMpgdDecoder):
            dec = decoder(cls, iterations=10)
            a, b = np.empty_like(y), np.empty_like(y)
            self.assertEqual(dec.decode(y, a), dec.decode(y * signs, b))
            np.testing.assert_allclose(b, a * signs, atol=1e-14)

    def test_visualizer_and_decode_use_identical_steps(self):
        dec = decoder(iterations=1)
        y = np.array([[-0.2, 1.1, 0.7, 1.0, 0.9, 0.8], np.ones(6)])
        batch = FrameBatch(y, np.ones_like(y, dtype=np.int8))
        engine = BatchDecoderEngine(TannerGraph.from_pcm(PCM), batch, seed=1, workers=2)
        before = engine.initial_state(dec, dec.parameters)
        frozen = {k: v.copy() for k, v in before.fields.items()}
        after, observation = engine.step(dec, before, dec.parameters, 0)
        for frame in range(2):
            out = np.empty(6)
            dec.decode(y[frame], out)
            np.testing.assert_allclose(after.fields["x"][frame], out)
        for key in frozen:
            np.testing.assert_array_equal(before.fields[key], frozen[key])
        self.assertEqual(sum(len(c.frame_indices) for c in observation.categories.values()), dec.edges_count)
        self.assertIs(decoder_catalog()["mpgd"], BinLdpcMpgdDecoder)

    def test_validation_and_factory(self):
        for key, value in (("eta_r", 0), ("eta_v", -1), ("alpha", -1),
                           ("l2_r", -1), ("l2_v", np.nan), ("eta_r", np.inf),
                           ("max_value", 0), ("max_value", -1), ("max_value", np.inf)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                decoder(**{key: value})
        for cls in (BinLdpcMpgdDecoder, CppBinLdpcMpgdDecoder):
            with self.assertRaisesRegex(ValueError, "L2"):
                decoder(cls, tau_r=0.02, tau_v=0.25)
        with self.assertRaisesRegex(ValueError, "L2"):
            BinLdpcMpgdDecoder.validate_parameters({"tau_r": 0.02})
        for name, cls in (("mpgd", BinLdpcMpgdDecoder), ("cpp mpgd", CppBinLdpcMpgdDecoder)):
            dec = create_decoder(name, None, pcm=PCM, block_length=6, n_checks=3,
                                 is_systematic=False, n_iterations=3)
            self.assertIsInstance(dec, cls)
        for pcm in (np.zeros_like(PCM), np.array([[1, 0], [0, 1]])):
            with self.assertRaises(ValueError):
                BinLdpcMpgdDecoder(None, pcm=pcm, block_length=pcm.shape[1],
                                   n_checks=pcm.shape[0], is_systematic=False, n_iterations=3)

    def test_tuner_grid_has_no_implicit_baseline(self):
        args = tune_mpgd.parse_args([
            "--alphas", "2", "--eta-rs", "0.2,0.2", "--eta-vs", "0.4",
            "--l2-rs", "0", "--l2-vs", "0.7", "--max-values", "1000000",
        ])
        tune_mpgd.validate_args(args)
        self.assertEqual(list(tune_mpgd.parameter_grid(args)), [
            dict(alpha=2.0, eta_r=0.2, eta_v=0.4, l2_r=0.0, l2_v=0.7, max_value=1e6),
        ])
        args = tune_mpgd.parse_args([])
        self.assertEqual(math.prod(len(x) for x in tune_mpgd.parameter_axes(args).values()), 109_760)
        self.assertIn(1e6, tune_mpgd.parameter_axes(args)["max_value"])

    def test_tuner_counts_early_wrong_codewords_and_fixed_error_limit(self):
        fake = SimpleNamespace(
            run=lambda snr, rng: SimpleNamespace(n_iter=1, fe_cum=0),
            codec=SimpleNamespace(decoder_impl=SimpleNamespace(output_ber=lambda *args: 0.5)),
            llr_out=np.ones(6), tx_bits=np.zeros(6),
        )
        tune_mpgd.init_worker({"codec": {}}, 0.8, 100, 3, 19)
        with patch.object(tune_mpgd, "LdpcExperimentSettings", lambda **config: config), \
             patch.object(tune_mpgd, "LdpcExperimentInstance", return_value=fake):
            result = tune_mpgd.evaluate_candidate((0, {}))
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["frame_errors"], 3)
        self.assertEqual(result["fer"], 1.0)
        self.assertTrue(result["target_reached"])


if __name__ == "__main__":
    unittest.main()
