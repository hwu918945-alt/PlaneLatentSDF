"""Regressions for final-coordinate band rejection and pair collisions."""
import copy
import unittest
import numpy as np
from .paired_geometry import Plane
from .paired_pool_coordinates import map_world, repair_coordinates


class CoordinateTests(unittest.TestCase):
    def fixture(self):
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        plane = Plane(5 * ring, 3 * ring, [-8, -8], [8, 8])
        rng = np.random.default_rng(19)
        banks = [plane.pairs(b, 32, rng) for b in (0, 1)]
        uv = np.concatenate([bank['q'].reshape(-1, 2) for bank in banks] + [plane.uniform(100, rng)])
        source = np.r_[np.tile([0, 1], 32), np.tile([2, 3], 32), np.full(100, 4)].astype(np.int8)
        return plane, banks, map_world(uv, 10., np.eye(3), .2), source

    def validate(self, plane, q, labels, source):
        actual, valid, distances = plane.classify(q[:, :2].astype(np.float64) * 10.)
        self.assertTrue(valid.all())
        self.assertGreater(float(distances.min()), .05)
        self.assertTrue(np.array_equal(labels, actual))
        expected = np.array([0, 2, 0, 1])
        self.assertTrue(np.array_equal(labels[source < 4], expected[source[source < 4]]))
        self.assertEqual(len(q), len(np.unique(q, axis=0)))

    def test_valid_pool_unchanged_including_rng(self):
        plane, banks, q, source = self.fixture()
        before = q.copy()
        rng = np.random.default_rng(29)
        state = copy.deepcopy(rng.bit_generator.state)
        q, labels, audit = repair_coordinates(plane, q, source, banks, 10., np.eye(3), .2, rng)
        self.assertTrue(np.array_equal(q, before))
        self.assertEqual(state, rng.bit_generator.state)
        self.assertEqual(audit['roi_points_retried'], 0)
        self.assertEqual(audit['boundary_pairs_retried'], [])
        self.validate(plane, q, labels, source)

    def test_roi_band_and_duplicate_rejections_are_reproducible(self):
        plane, banks, q, source = self.fixture()
        q[-1] = q[-2]
        # A point on the exclusion band in final training coordinates.
        q[-3] = [float(plane.curves[0][0, 0]) / 10., 0., .2]
        original = q.copy()
        first, labels, audit = repair_coordinates(
            plane, q.copy(), source, copy.deepcopy(banks), 10., np.eye(3), .2, np.random.default_rng(19))
        second, _, audit2 = repair_coordinates(
            plane, q.copy(), source, copy.deepcopy(banks), 10., np.eye(3), .2, np.random.default_rng(19))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(audit, audit2)
        self.assertEqual(audit['roi_points_retried'], 2)
        self.assertTrue(np.array_equal(first[:-3], original[:-3]))
        self.assertTrue(np.array_equal(first[-2], original[-2]))
        self.validate(plane, first, labels, source)

    def test_pair_collision_preserves_anchor_scale_and_two_sided_unit(self):
        plane, banks, q, source = self.fixture()
        arcs = [bank['arc'].copy() for bank in banks]
        scales = [bank['scale_id'].copy() for bank in banks]
        # Two final endocardial negatives quantize to one coordinate.
        q[67] = q[65]
        before = q.copy()
        q, labels, audit = repair_coordinates(
            plane, q, source, banks, 10., np.eye(3), .2, np.random.default_rng(19))
        self.assertEqual(len(audit['boundary_pairs_retried']), 1)
        row = audit['boundary_pairs_retried'][0]
        self.assertEqual((row['boundary'], row['unit']), (1, 1))
        self.assertTrue(row['anchor_preserved'])
        for bank, arc, scale in zip(banks, arcs, scales):
            self.assertTrue(np.array_equal(bank['arc'], arc))
            self.assertTrue(np.array_equal(bank['scale_id'], scale))
            self.assertTrue((bank['delta'] <= .4 * bank['clearance'] + 1e-12).all())
        self.assertTrue(np.array_equal(q[:66], before[:66]))
        self.assertTrue(np.array_equal(q[68:], before[68:]))
        self.validate(plane, q, labels, source)


if __name__ == '__main__':
    unittest.main()
