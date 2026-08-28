"""Tests for graph_matcher.py — neighborhood graph matching."""

from __future__ import annotations

import numpy as np
import pytest

from lunar_data_pipeline.crater_graph import build_graph
from lunar_data_pipeline.graph_matcher import (
    Correspondence,
    match_consistent,
    match_mutual_neighbor,
    match_nearest_neighbor,
    matches_to_pixel_pairs,
)


def _cluster():
    base = np.array([[0, 0], [100, 0], [0, -100], [70, 80], [-60, 50], [10, -40]])
    rad = np.array([30, 20, 25, 15, 28, 22])
    return base, rad


class TestMatchNearestNeighbor:
    def test_matches_identical_layout(self):
        base, rad = _cluster()
        gA = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        gB = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        m = match_nearest_neighbor(gA, gB, min_similarity=0.0)
        ids = {(x.idx_a, x.idx_b) for x in m}
        assert all((i, i) in ids for i in range(6))

    def test_matches_under_scale_and_rotation(self):
        # B is A scaled 1.5x and rotated 30 deg -> descriptors must agree.
        base, rad = _cluster()
        gA = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        th = np.radians(30)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        B = (R @ (base * 1.5).T).T
        gB = build_graph(B[:, 0], B[:, 1], rad * 1.5, meters_per_px=6.13, k=3)
        m = match_nearest_neighbor(gA, gB, min_similarity=0.0)
        ids = {(x.idx_a, x.idx_b) for x in m}
        assert all((i, i) in ids for i in range(6))

    def test_empty_graph(self):
        gA = build_graph([], [], [], 6.13, k=3)
        base, rad = _cluster()
        gB = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        assert match_nearest_neighbor(gA, gB) == []

    def test_matches_to_pixel_pairs_reorders_consistently(self):
        base, rad = _cluster()
        gA = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        gB = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        m = match_nearest_neighbor(gA, gB, min_similarity=0.0)
        pa, pb = matches_to_pixel_pairs(gA, gB, m)
        assert pa.shape == (len(m), 2)
        assert pb.shape == (len(m), 2)
        # For identical graphs the pixel coords must match exactly.
        np.testing.assert_allclose(pa, pb)


class TestMatchConsistent:
    def test_consistency_keeps_good_cluster_matches(self):
        base, rad = _cluster()
        gA = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        gB = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        m0 = [Correspondence(idx_a=i, idx_b=i, similarity=1.0) for i in range(6)]
        kept = match_consistent(gA, gB, m0)
        assert len(kept) >= 4  # most survive; structure is mutually consistent

    def test_consistency_drops_bad_permutation(self):
        # Asymmetric line (k=1): each node's single neighbour is well-defined.
        # x = [0, 50, 160] -> neighbours: 0->1, 1->0, 2->1.
        base = np.array([[0.0, 0.0], [50.0, 0.0], [160.0, 0.0]])
        rad = [20.0, 20.0, 20.0]
        gA = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=1)
        gB = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=1)

        good = [Correspondence(idx_a=i, idx_b=i, similarity=1.0) for i in range(3)]
        kept_good = match_consistent(gA, gB, good, topology_tol_ratio=0.5)
        assert len(kept_good) == 3  # identity survives

        bad = [
            Correspondence(idx_a=0, idx_b=2, similarity=0.5),
            Correspondence(idx_a=1, idx_b=0, similarity=0.5),
            Correspondence(idx_a=2, idx_b=1, similarity=0.5),
        ]
        kept_bad = match_consistent(gA, gB, bad, topology_tol_ratio=0.5)
        assert len(kept_bad) < len(bad)  # non-automorphism pruned

    def test_empty_input(self):
        assert match_consistent(build_graph([], [], [], 6.13),
                                build_graph([], [], [], 6.13), []) == []


def _central_constellation(offset_x=0.0):
    """Central crater (node 0) with 4 distinct neighbours, plus a far node."""
    cx = [offset_x + 0, 100 + offset_x, 0 + offset_x, 140 + offset_x, 0 + offset_x,
          600 + offset_x]
    cy = [0, 0, 120, 0, -200, 600]
    rad = [40, 20, 18, 25, 15, 30]
    return np.array(cx, dtype=float), np.array(cy, dtype=float), np.array(rad, dtype=float)


class TestMatchMutualNeighbor:
    def test_identical_layout_matches_all(self):
        cx, cy, rad = _central_constellation()
        gA = build_graph(cx, cy, rad, meters_per_px=6.13, k=4)
        gB = build_graph(cx, cy, rad, meters_per_px=6.13, k=4)
        m = match_mutual_neighbor(gA, gB, min_mutual_fraction=0.5)
        ids = {x.idx_a: x.idx_b for x in m}
        # Central node 0 must map back to itself.
        assert ids.get(0) == 0

    def test_rotation_and_scale_tolerated(self):
        # B is A scaled 1.5x and rotated 30 deg (no bearing used).
        cx, cy, rad = _central_constellation()
        gA = build_graph(cx, cy, rad, meters_per_px=6.13, k=4)
        th = np.radians(30)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = R @ (np.column_stack([cx, cy]) * 1.5).T
        gB = build_graph(pts[0], pts[1], rad * 1.5, meters_per_px=6.13, k=4)
        m = match_mutual_neighbor(gA, gB, min_mutual_fraction=0.5)
        ids = {x.idx_a: x.idx_b for x in m}
        assert ids.get(0) == 0

    def test_physical_mode_rejects_spectral_twin_decoy(self):
        # A's central node has an exact *spectral twin* B-node whose sorted
        # (dist_ratio, size_ratio) multiset is identical, but whose physical
        # distances are 4x larger (so every A-B neighbour pairing fails the
        # log-distance tolerance).  Ratio (scale-free) mode cannot tell them
        # apart; physical mode (same m/px in both images) rejects the decoy.
        cx_a, cy_a, rad_a = _central_constellation(offset_x=0.0)

        # True B set == A set.
        bx_t, by_t, br_t = _central_constellation(offset_x=0.0)
        # Decoy B set: same ratio/size multiset, physical distances x4, far away.
        k = 4
        cx_d = [9000 + v for v in (0, 100 * k, 0, 140 * k, 0, 1200)]
        cy_d = [0, 0, 120 * k, 0, -200 * k, 600]
        rad_d = np.array([40, 20, 18, 25, 15, 30])

        bx = np.concatenate([bx_t, np.array(cx_d)])
        by = np.concatenate([by_t, np.array(cy_d)])
        br = np.concatenate([br_t, rad_d])

        gA = build_graph(cx_a, cy_a, rad_a, meters_per_px=6.13, k=4)
        gB = build_graph(bx, by, br, meters_per_px=6.13, k=4)

        phys = {x.idx_a: x.idx_b for x in
                match_mutual_neighbor(gA, gB, min_mutual_fraction=0.5,
                                      neighbor_d_tol=0.5, physical=True)}
        # True central (node 0) must match B's true central (node 0), NOT the
        # decoy central (node 6).
        assert phys.get(0) == 0

        # Show why physical mode is needed: in ratio (scale-free) mode the
        # spectral-twin decoy is scored fully consistent with A's central, so
        # it cannot be rejected; physical distance disambiguates.
        from lunar_data_pipeline.crater_graph import node_neighborhoods
        from lunar_data_pipeline.graph_matcher import _neighbor_consistency
        nhs_a = node_neighborhoods(gA)
        nhs_b = node_neighborhoods(gB)
        nhr = _neighbor_consistency(nhs_a[0], nhs_b[6], d_tol=0.5, s_tol=0.6,
                                    physical=False)
        nhp = _neighbor_consistency(nhs_a[0], nhs_b[6], d_tol=0.5, s_tol=0.6,
                                    physical=True)
        assert nhr[1] >= 0.5   # ratio mode: decoy looks fully consistent
        assert nhp[1] < 0.5    # physical mode: decoy has no plausible partners

    def test_completely_different_constellation_rejected(self):
        # A's central crater should find NO good partner when B's neighbours
        # sit at a clearly different physical spacing.  Tighter log-distance
        # tolerance makes the rejection unambiguous.
        cx_a, cy_a, rad_a = _central_constellation()
        gA = build_graph(cx_a, cy_a, rad_a, meters_per_px=6.13, k=4)
        # B's central has neighbours ~8x farther than A's central.
        bx = np.array([0, 800, 0, 900, 0, 1300])
        by = np.array([0, 0, 900, 0, -1000, 0])
        br = np.array([40, 20, 18, 25, 15, 30])
        gB = build_graph(bx, by, br, meters_per_px=6.13, k=4)
        m = match_mutual_neighbor(gA, gB, min_mutual_fraction=0.75,
                                  neighbor_d_tol=0.3, physical=True)
        # A's central (node 0) must not find a perfect-structure partner.
        a0 = [c for c in m if c.idx_a == 0]
        assert all(c.mutual_fraction < 1.0 for c in a0)

    def test_empty_graph(self):
        gA = build_graph([], [], [], 6.13, k=4)
        cx, cy, rad = _central_constellation()
        gB = build_graph(cx, cy, rad, 6.13, k=4)
        assert match_mutual_neighbor(gA, gB) == []
