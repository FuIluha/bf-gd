import unittest
from pathlib import Path

import numpy as np

from visualisation.error_by_energy_visualisation.algorithms import TannerGraph
from visualisation.error_by_energy_visualisation.app import _rebuild_dataset_session
from visualisation.error_by_energy_visualisation.data import DatasetConfig, load_dataset
from visualisation.error_by_energy_visualisation.models import FrameBatch
from visualisation.error_by_energy_visualisation.session import ExplorerSession, compatible_datasets, comparison_from_bytes


PCM = np.asarray([
    [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1],
], dtype=np.uint8)


def make_session(algorithm="ftgdbf"):
    received = np.asarray([
        [-0.2, 1.1, 0.7, 1.0, 0.9, 0.8],
        [-0.8, 0.3, 0.9, -1.2, 1.1, 0.7],
    ], dtype=np.float32)
    batch = FrameBatch(received, np.ones(received.shape, dtype=np.int8))
    return ExplorerSession(TannerGraph.from_pcm(PCM), batch, seed=7, workers=2,
                           algorithm=algorithm, metadata={"code": "test"})


class SessionTests(unittest.TestCase):
    def test_back_preserves_snapshot_and_new_step_truncates_future(self):
        session = make_session()
        first, second = {"alpha": 1.0}, {"alpha": 2.0}
        session.step_forward(first)
        saved = session.snapshots[0].state.fields["x"].copy()
        session.step_forward(first)
        session.step_back()
        np.testing.assert_array_equal(session.snapshots[0].state.fields["x"], saved)
        session.step_forward(second)
        self.assertEqual(len(session.snapshots), 3)
        self.assertEqual(session.transitions[-1].parameters, second)
        with self.assertRaises(ValueError):
            session.snapshots[0].state.fields["x"][0, 0] = -1

    def test_round_trip_for_all_decoders(self):
        for key in ("ftgdbf", "pmgdbf", "gdms"):
            with self.subTest(key=key):
                session = make_session(key)
                session.step_forward(session.spec.defaults())
                session.step_back()
                restored = ExplorerSession.from_bytes(session.to_bytes(), workers=1)
                self.assertEqual(restored.algorithm, key)
                self.assertEqual(restored.cursor, 0)
                self.assertEqual(len(restored.snapshots), 2)
                for before, after in zip(session.snapshots, restored.snapshots):
                    for field in before.state.fields:
                        np.testing.assert_array_equal(before.state.fields[field], after.state.fields[field])

    def test_comparison_requires_same_batch(self):
        session = make_session()
        comparison = comparison_from_bytes(session.to_bytes(), "same")
        self.assertTrue(compatible_datasets(session.metadata, comparison.metadata))
        changed = dict(comparison.metadata, batch_digest="different")
        self.assertFalse(compatible_datasets(session.metadata, changed))

    def test_rebuild_dataset_resets_only_the_new_session(self):
        code = Path(__file__).resolve().parents[3] / "codes/ldpc_savin_4_8_12_24_Zc54.json"
        dataset = load_dataset(DatasetConfig(code, frames=2, snr_db=1.0, seed=7))
        previous = ExplorerSession(dataset.graph, dataset.batch, seed=7, metadata=dataset.metadata)
        previous.step_forward(previous.spec.defaults())
        rebuilt = _rebuild_dataset_session(previous, 3, -0.2)
        self.assertEqual(rebuilt.batch.frames, 3)
        self.assertEqual(rebuilt.metadata["snr_db"], -0.2)
        self.assertEqual(rebuilt.cursor, 0)
        self.assertEqual(rebuilt.algorithm, previous.algorithm)
        self.assertEqual(previous.batch.frames, 2)
        self.assertEqual(previous.cursor, 1)
        with self.assertRaises(ValueError):
            _rebuild_dataset_session(previous, 1.5, 0.0)


if __name__ == "__main__":
    unittest.main()
