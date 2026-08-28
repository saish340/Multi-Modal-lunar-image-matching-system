"""Tests for crater_graph.py — neighborhood graph construction."""

from __future__ import annotations

import numpy as np
import pytest

from lunar_data_pipeline.crater_graph import (
    GraphEdge,
    build_graph,
    node_neighborhoods,
)


class TestBuildGraph:
    def test_k_nearest_neighbor_edges(self):
        # 4 craters in a square; each should connect to its 2 nearest.
        cx = [0.0, 100.0, 0.0, 100.0]
        cy = [0.0, 0.0, 100.0, 100.0]
        rad = [20.0, 20.0, 20.0, 20.0]
        g = build_graph(cx, cy, rad, meters_per_px=6.13, k=2)

        assert len(g.nodes) == 4
        # Each node has exactly min(k, n-1) = 2 neighbours.
        for n in g.nodes:
            assert len(g.neighbors_of(n.id)) == 2

    def test_physical_unit_distances(self):
        # 10px spacing at 6.13 m/px = 61.3 m.
        g = build_graph([0.0, 10.0], [0.0, 0.0], [5.0, 5.0],
                        meters_per_px=6.13, k=1)
        assert g.edges[0].dist_m == pytest.approx(61.3)

    def test_size_ratio_between_nodes(self):
        g = build_graph([0.0, 100.0], [0.0, 0.0], [20.0, 10.0],
                        meters_per_px=1.0, k=1)
        # min/max = 10/20 = 0.5
        assert g.edges[0].size_ratio == pytest.approx(0.5)

    def test_min_radius_m_filters_tiny_noise(self):
        # radii 5px at 6.13 m/px = 30.65 m < min_radius_m=100 -> dropped.
        # The 30px and 40px craters survive and are re-indexed to 0, 1.
        g = build_graph([0.0, 100.0, 200.0], [0.0, 0.0, 0.0],
                        [30.0, 5.0, 40.0], meters_per_px=6.13,
                        min_radius_m=100.0)
        assert len(g.nodes) == 2
        # Remaining radii must be 30px and 40px (the 5px noise filtered).
        radii = sorted(n.radius for n in g.nodes)
        assert radii == [30.0, 40.0]

    def test_single_node_graph(self):
        g = build_graph([0.0], [0.0], [10.0], meters_per_px=6.13, k=3)
        assert len(g.nodes) == 1
        assert g.edges == []

    def test_too_few_nodes_is_robust(self):
        g = build_graph([0.0, 10.0], [0.0, 0.0], [5.0, 5.0],
                        meters_per_px=6.13, k=5)
        for n in g.nodes:
            assert len(g.neighbors_of(n.id)) == 1  # only 1 other node

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            build_graph([0.0, 1.0], [0.0], [1.0, 1.0], meters_per_px=1.0)
        with pytest.raises(ValueError):
            build_graph([0.0], [0.0], [1.0], meters_per_px=0.0)
        with pytest.raises(ValueError):
            build_graph([0.0], [0.0], [1.0], meters_per_px=1.0, k=0)

    def test_edges_form_knn_in_decreasing_density(self):
        # Dense cluster of 3 + one far away: far node connects only to dense group.
        cx = [0.0, 10.0, 20.0, 1000.0]
        cy = [0.0, 0.0, 0.0, 0.0]
        g = build_graph(cx, cy, [10.0] * 4, meters_per_px=1.0, k=2)
        # Node 3 (far) has exactly 2 neighbours (the nearest of the cluster).
        assert len(g.neighbors_of(3)) == 2


class TestNodeNeighborhoods:
    def test_dist_ratio_is_scale_invariant(self):
        # Same layout at two global scales -> identical dist_ratio vectors.
        base = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]])
        rad = [20.0] * 5
        g1 = build_graph(base[:, 0], base[:, 1], rad, meters_per_px=6.13, k=3)
        g2 = build_graph(base[:, 0] * 2, base[:, 1] * 2, rad,
                         meters_per_px=6.13, k=3)
        n1 = node_neighborhoods(g1)[0]
        n2 = node_neighborhoods(g2)[0]
        np.testing.assert_allclose(n1.dist_ratio, n2.dist_ratio)

    def test_descriptor_is_fixed_length(self):
        g = build_graph([0.0, 50.0, 100.0], [0.0, 0.0, 0.0],
                        [20.0, 20.0, 20.0], meters_per_px=6.13, k=4)
        v = node_neighborhoods(g)[0].to_vector()
        assert v.shape == (2 * 4,)
