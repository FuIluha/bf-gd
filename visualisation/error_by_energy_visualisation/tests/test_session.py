import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.algorithms import TannerGraph
from visualisation.error_by_energy_visualisation.models import (
    Algorithm,
    FrameBatch,
    FtgdbfParameters,
)
from visualisation.error_by_energy_visualisation.session import (
    ExplorerSession,
    compatible_datasets,
    comparison_from_bytes,
)


PCM = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1],
], dtype=np.uint8)


def make_session():
    received = np.asarray([
        [-0.2, 1.1, 0.7, 1.0, 0.9, 0.8],
        [-0.8, 0.3, 0.9, -1.2, 1.1, 0.7],
    ], dtype=np.float32)
    transmitted = np.ones_like(received, dtype=np.int8)
    batch = FrameBatch(received=received, transmitted_symbols=transmitted)
    return ExplorerSession(
        TannerGraph.from_pcm(PCM),
        batch,
        seed=7,
        workers=2,
        algorithm=Algorithm.FTGDBF,
        metadata={"code": "test"},
    )


class SessionTests(unittest.TestCase):
    def test_back_preserves_snapshot_and_new_step_truncates_future(self):
        session = make_session()
        first = FtgdbfParameters(alpha=1.0)
        second = FtgdbfParameters(alpha=2.0)
        session.step_forward(first)
        saved_state = session.snapshots[0].state.x.copy()
        session.step_forward(first)
        self.assertEqual(len(session.snapshots), 3)
        session.step_back()
        np.testing.assert_array_equal(session.snapshots[0].state.x, saved_state)
        session.step_forward(second)
        self.assertEqual(len(session.snapshots), 3)
        self.assertEqual(session.transitions[-1].parameters, second)
        with self.assertRaises(ValueError):
            session.snapshots[0].state.x[0, 0] = -1

    def test_npz_round_trip_preserves_history_and_cursor(self):
        session = make_session()
        session.step_forward(FtgdbfParameters(alpha=1.25))
        session.step_forward(FtgdbfParameters(alpha=1.5))
        session.step_back()
        restored = ExplorerSession.from_bytes(session.to_bytes(), workers=1)
        self.assertEqual(restored.algorithm, session.algorithm)
        self.assertEqual(restored.cursor, 1)
        self.assertEqual(len(restored.snapshots), 3)
        self.assertEqual(restored.transitions[1].parameters.alpha, 1.5)
        for expected, actual in zip(session.snapshots, restored.snapshots):
            np.testing.assert_array_equal(expected.state.x, actual.state.x)

    def test_comparison_requires_the_same_fixed_batch(self):
        session = make_session()
        comparison = comparison_from_bytes(session.to_bytes(), "same")
        self.assertTrue(compatible_datasets(session.metadata, comparison.metadata))
        changed = dict(comparison.metadata)
        changed["batch_digest"] = "different"
        self.assertFalse(compatible_datasets(session.metadata, changed))


if __name__ == "__main__":
    unittest.main()
