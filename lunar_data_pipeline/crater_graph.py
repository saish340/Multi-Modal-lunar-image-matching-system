"""Build local neighborhood graphs from detected craters.

Phase 4 (crater-neighborhood-graph matching): a crater's rim structure is a
property of the terrain, not of the local illumination, so it should be
reproducible across OHRC and TMC-2 even though pixel appearance is not.  This
module turns a set of detected craters (positions + radii) into a graph whose
nodes are craters and whose edges connect each crater to its K nearest
neighbours, decorated with scale/rotation-invariant physical features so the
graphs from the two instruments are directly comparable.

Features are computed in PHYSICAL units (metres), not pixels: each image's
known metres-per-pixel (from pds4_parser) normalises distances so the OHRC
(0.2 m/px) and TMC-2 (6.13 m/px) graphs live in the same space.  Bearing is
deliberately NOT used as a primary feature: the corner-homography between our
two images shows a ~2.7 deg rotation plus a y-axis flip, so raw bearing is not
reliably preserved across frames (see groundtruth.py limitation notes).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CraterNode:
    """A detected crater used as a graph node.

    ``cx``/``cy``/``radius`` are in IMAGE PIXELS (the detector's native frame);
    ``x_m``/``y_m``/``radius_m`` are the same point in physical metres computed
    with the image's metres-per-pixel, used for distance/size features.
    """

    id: int
    cx: float
    cy: float
    radius: float
    confidence: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    radius_m: float = 0.0

    @property
    def position_px(self) -> tuple[float, float]:
        return (self.cx, self.cy)


@dataclass
class GraphEdge:
    """Edge between two crater nodes in one image's neighborhood graph."""

    a: int  # node id (lower)
    b: int  # node id (higher)
    dist_m: float          # physical centre distance (m)
    size_ratio: float      # min(r_a, r_b) / max(r_a, r_b) in physical units, 0..1


@dataclass
class CraterGraph:
    """Neighborhood graph over one image's detected craters."""

    nodes: list[CraterNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    k: int = 4

    def node_ids(self) -> list[int]:
        return [n.id for n in self.nodes]

    def node(self, node_id: int) -> CraterNode:
        return next(n for n in self.nodes if n.id == node_id)

    def neighbors_of(self, node_id: int) -> list[tuple[int, GraphEdge]]:
        """Return [(neighbor_id, edge)] for the given node."""
        out = []
        for e in self.edges:
            if e.a == node_id:
                out.append((e.b, e))
            elif e.b == node_id:
                out.append((e.a, e))
        return out


@dataclass
class NodeNeighborhood:
    """The local neighborhood of a node, used as its matching descriptor.

    Contains the K nearest neighbours sorted by physical distance, with
    scale-invariant distance ratios and size ratios.
    """

    node_id: int
    # For each of the K neighbours, in order of increasing distance:
    neighbor_ids: list[int] = field(default_factory=list)
    dist_m: np.ndarray = None  # type: ignore[assignment]   physical distances
    dist_ratio: np.ndarray = None  # type: ignore[assignment]  d_k / median(d)
    size_ratio: np.ndarray = None  # type: ignore[assignment]  min(rn, r0)/max(rn, r0)
    node_radius_m: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Concatenated rotation-invariant descriptor for this neighborhood."""
        return np.concatenate([self.dist_ratio, self.size_ratio])


def _to_node(
    idx: int,
    cx: float,
    cy: float,
    radius: float,
    confidence: float,
    meters_per_px: float,
) -> CraterNode:
    return CraterNode(
        id=idx,
        cx=float(cx),
        cy=float(cy),
        radius=float(radius),
        confidence=float(confidence),
        x_m=float(cx) * meters_per_px,
        y_m=float(cy) * meters_per_px,
        radius_m=float(radius) * meters_per_px,
    )


def build_graph(
    cx: list[float] | np.ndarray,
    cy: list[float] | np.ndarray,
    radius: list[float] | np.ndarray,
    meters_per_px: float,
    *,
    k: int = 4,
    confidence: list[float] | np.ndarray | None = None,
    min_radius_m: float = 0.0,
) -> CraterGraph:
    """Build a K-nearest-neighbour graph from detected craters.

    Parameters
    ----------
    cx, cy, radius : array-like
        Crater centres (pixel) and radii (pixel) in the detector's native frame.
    meters_per_px : float
        Image resolution.  Converts pixel geometry to physical metres.
    k : int
        Number of nearest neighbours per node.
    confidence : array-like, optional
        Per-crater confidence (defaults to 1.0).
    min_radius_m : float
        Drop craters smaller than this physical radius before building the
        graph (used to remove tiny, noisy detections that add little structure).

    Returns
    -------
    CraterGraph
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if confidence is None:
        confidence = np.ones(len(cx))
    confidence = np.asarray(confidence, dtype=float)

    if not (len(cx) == len(cy) == len(radius) == len(confidence)):
        raise ValueError("cx, cy, radius, confidence must be same length")
    if meters_per_px <= 0:
        raise ValueError("meters_per_px must be positive")
    if k < 1:
        raise ValueError("k must be >= 1")

    radius_m = radius * meters_per_px
    keep = radius_m >= min_radius_m
    cx, cy, radius, confidence, radius_m = (
        cx[keep], cy[keep], radius[keep], confidence[keep], radius_m[keep],
    )

    nodes = [
        _to_node(i, float(cx[i]), float(cy[i]), float(radius[i]),
                 float(confidence[i]), meters_per_px)
        for i in range(len(cx))
    ]

    if len(nodes) < 2:
        return CraterGraph(nodes=nodes, k=k)

    pos = np.column_stack([cx * meters_per_px, cy * meters_per_px])  # metres
    edge_set: set[tuple[int, int]] = set()
    edges: list[GraphEdge] = []
    for i in range(len(nodes)):
        d2 = np.sum((pos - pos[i]) ** 2, axis=1)
        d2[i] = np.inf
        nbr_order = np.argsort(d2)[: min(k, len(nodes) - 1)]
        for j in nbr_order:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edge_set:
                continue
            edge_set.add((a, b))
            dist_m = math.sqrt(d2[j])
            size_ratio = (
                min(radius_m[i], radius_m[j]) / max(radius_m[i], radius_m[j])
                if max(radius_m[i], radius_m[j]) > 0 else 0.0
            )
            edges.append(GraphEdge(a=int(a), b=int(b), dist_m=dist_m,
                                   size_ratio=size_ratio))

    graph = CraterGraph(nodes=nodes, edges=edges, k=k)
    return graph


def _neighborhood(graph: CraterGraph, node_id: int) -> NodeNeighborhood:
    """Extract the sorted-K-neighbour descriptor for a node, padded to K slots."""
    nbr = graph.neighbors_of(node_id)
    k = graph.k
    node = graph.node(node_id)
    if not nbr:
        return NodeNeighborhood(node_id=node_id,
                                neighbor_ids=[-1] * k,
                                dist_m=np.full(k, np.inf),
                                dist_ratio=np.full(k, 3.0),
                                size_ratio=np.zeros(k),
                                node_radius_m=node.radius_m)
    nbr.sort(key=lambda t: t[1].dist_m)
    nbr = nbr[:k]
    ids = [nid for nid, _ in nbr]
    dist = np.array([e.dist_m for _, e in nbr], dtype=float)
    sratio = np.array([
        min(node.radius_m, graph.node(nid).radius_m)
        / max(node.radius_m, graph.node(nid).radius_m)
        for nid, _ in nbr
    ], dtype=float)
    med = np.median(dist) if len(dist) else 1.0
    dist_ratio = dist / med if med > 0 else np.ones_like(dist)

    # Pad to exactly K slots so the descriptor matrix is rectangular.
    if len(ids) < k:
        pad = k - len(ids)
        ids = ids + [-1] * pad
        dist = np.concatenate([dist, np.full(pad, np.inf)])
        dist_ratio = np.concatenate([dist_ratio, np.full(pad, 3.0)])
        sratio = np.concatenate([sratio, np.zeros(pad)])

    return NodeNeighborhood(
        node_id=node_id,
        neighbor_ids=ids,
        dist_m=dist,
        dist_ratio=dist_ratio,
        size_ratio=sratio,
        node_radius_m=node.radius_m,
    )


def node_neighborhoods(graph: CraterGraph) -> dict[int, NodeNeighborhood]:
    """Return {node_id: NodeNeighborhood} for every node in the graph."""
    return {n.id: _neighborhood(graph, n.id) for n in graph.nodes}
