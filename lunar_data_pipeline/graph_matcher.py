"""Match crater-neighborhood graphs between two images.

Given a graph from image A (OHRC) and one from image B (TMC-2), find crater
correspondences based on the STRUCTURAL similarity of their local
neighborhoods, not on pixel appearance.

Descriptor
----------
Each node's local neighborhood is summarized by its K nearest neighbours
sorted by physical distance, giving two scale & rotation-invariant vectors:

- ``dist_ratio``: neighbour distances normalized by their median (scale-free),
- ``size_ratio``: min/max radius ratio of each neighbour to the node.

These two vectors are concatenated into a 2K-length node descriptor. Because
they are sorted by distance and normalized, the descriptor is invariant to
rotation (no bearing used — the OHRC/TMC-2 frames have a ~2.7 deg rotation
plus a y-flip) and to resolution/scale (distances are ratio-normalized and in
physical metres).

Matching
--------
:func:`match_nearest_neighbor` gives each node in A its best-matching node in B
by descriptor similarity, applying a Lowe-style ratio test to reject ambiguous
matches (when the 2nd-best B candidate is nearly as similar).  This is the
simple, promoted approach; :func:`match_consistent` optionally prunes
correspondences whose matched neighbours are not mutually consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .crater_graph import CraterGraph, NodeNeighborhood, node_neighborhoods

logger = logging.getLogger(__name__)


@dataclass
class Correspondence:
    """A predicted crater-to-crater correspondence (node A -> node B)."""

    idx_a: int          # node id in graph A
    idx_b: int          # node id in graph B
    similarity: float   # 0..1 structural similarity
    ratio: float = 1.0  # Lowe-style: dissimilarity(A->best)/dissimilarity(A->2nd best)
    mutual_consistency: int = 0     # # of A's neighbours that found a B partner
    mutual_fraction: float = 0.0    # mutual_consistency / K


def _descriptor_matrix(graph: CraterGraph) -> tuple[list[int], np.ndarray, dict[int, NodeNeighborhood]]:
    """Return (node_ids, descriptor matrix (N, 2K), neighborhoods dict)."""
    nhs = node_neighborhoods(graph)
    ids = []
    rows = []
    empty = []
    for nid in sorted(nhs):
        nh = nhs[nid]
        ids.append(nid)
        rows.append(nh.to_vector())
    mat = np.array(rows, dtype=float) if rows else np.empty((0, 2 * graph.k))
    return ids, mat, nhs


@dataclass
class _CandidateSearch:
    """All A->B scores for the (possibly large) A x B matrix via row chunks."""

    ids_a: list
    mat_a: np.ndarray
    ids_b: list
    mat_b: np.ndarray

    def best_two_for_a(self, i: int, sigma: float) -> tuple[float, float, int]:
        """Return (best_sim, second_sim, best_b_pos) for A row i."""
        vec = self.mat_a[i]
        # dissimilarity = 1 - exp(-( (a-b)/sigma )^2 / 2 ) averaged over dims
        diff = self.mat_b - vec  # (N_b, 2K)
        sim = np.mean(np.exp(-((diff / sigma) ** 2) / 2.0), axis=1)
        order = np.argsort(sim)[::-1]
        best_pos = order[0]
        best_sim = sim[best_pos]
        second_sim = sim[order[1]] if len(order) > 1 else 0.0
        return float(best_sim), float(second_sim), int(best_pos)


def match_nearest_neighbor(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    *,
    ratio_test: float = 0.6,
    min_similarity: float = 0.25,
    sigma: float = 0.5,
    max_nodes_a: int | None = None,
    max_nodes_b: int | None = None,
) -> list[Correspondence]:
    """Match each node in A to its best node in B with a Lowe-style ratio test.

    Keeps an A->B assignment only if:

    - the best B similarity >= ``min_similarity``, and
    - it is unambiguous: ``(1 - best) < ratio_test * (1 - second)`` (Lowe's
      ratio test in dissimilarity space).

    The A and B node sets can be independently capped (e.g. by confidence) by
    trimming the graphs beforehand.
    """
    ids_a, mat_a, _ = _descriptor_matrix(graph_a)
    ids_b, mat_b, _ = _descriptor_matrix(graph_b)

    if len(ids_a) == 0 or len(ids_b) == 0 or mat_a.size == 0 or mat_b.size == 0:
        logger.warning("One graph has no usable nodes to match")
        return []

    search = _CandidateSearch(ids_a, mat_a, ids_b, mat_b)
    matches: list[Correspondence] = []
    for i in range(len(ids_a)):
        best_sim, second_sim, best_b_pos = search.best_two_for_a(i, sigma)
        if best_sim < min_similarity:
            continue
        if best_b_pos < 0:
            continue
        # Lowe ratio test in dissimilarity space.
        d_best = 1.0 - best_sim
        d_second = 1.0 - second_sim
        if d_second <= 0:
            pass  # perfect and unambiguous
        elif d_best >= ratio_test * d_second:
            continue
        matches.append(
            Correspondence(
                idx_a=ids_a[i],
                idx_b=ids_b[best_b_pos],
                similarity=best_sim,
                ratio=float(d_best / d_second) if d_second > 0 else 0.0,
            )
        )
    logger.info("Graph matching: %d A-nodes -> %d matches", len(ids_a), len(matches))
    return matches


def match_consistent(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    matches: list[Correspondence],
    *,
    topology_tol_ratio: float = 0.5,
) -> list[Correspondence]:
    """Prune matches that lack mutual-neighbour consistency.

    A crater in A matched to a crater in B is kept only if a comparable
    fraction of its graph neighbours map onto the matched crater's graph
    neighbours.  This imposes global structural consistency cheaply and is
    meant as a refinement pass over :func:`match_nearest_neighbor`.
    """
    if not matches:
        return []
    ba = node_neighborhoods(graph_a)
    bb = node_neighborhoods(graph_b)
    a_to_b = {m.idx_a: m.idx_b for m in matches}
    kept: list[Correspondence] = []
    for m in matches:
        na = ba.get(m.idx_a)
        nb = bb.get(m.idx_b)
        if na is None or nb is None or len(na.neighbor_ids) == 0 or len(nb.neighbor_ids) == 0:
            continue
        hit = 0
        for an in na.neighbor_ids:
            if an in a_to_b and a_to_b[an] in nb.neighbor_ids:
                hit += 1
        fraction = hit / len(na.neighbor_ids)
        if fraction >= 1.0 - topology_tol_ratio:
            kept.append(m)
    logger.info("Consistency pass: %d -> %d matches", len(matches), len(kept))
    return kept


def _neighbor_consistency(
    na: NodeNeighborhood,
    nb: NodeNeighborhood,
    *,
    d_tol: float,
    s_tol: float,
    physical: bool = False,
) -> tuple[int, float]:
    """Count how many of A's neighbours find a plausible, distinct B partner.

    Greedily matches A's specific neighbours to DISTINCT B neighbours using
    each neighbour's scale/rotation-invariant features (distance ratio + size
    ratio; optionally physical distance instead).  If *physical* is True the
    images are assumed to share metres-per-pixel and distances are compared in
    metres (more discriminative when applicable); otherwise distance-ratio
    (scale-free) is used.

    Returns ``(n_matched, fraction)`` where ``fraction = n_matched / K``.
    """
    na_ids = np.asarray(na.neighbor_ids)
    nb_ids = np.asarray(nb.neighbor_ids)
    da = np.asarray(na.dist_m if physical else na.dist_ratio, dtype=float)
    db = np.asarray(nb.dist_m if physical else nb.dist_ratio, dtype=float)
    sa = np.asarray(na.size_ratio, dtype=float)
    sb = np.asarray(nb.size_ratio, dtype=float)

    a_sel = np.where(na_ids != -1)[0]
    b_sel = np.where(nb_ids != -1)[0]
    da = da[a_sel]
    sa = sa[a_sel]
    db = db[b_sel]
    sb = sb[b_sel]

    m = len(da)
    n = len(db)
    if m == 0 or n == 0:
        return 0, 0.0

    log_da = np.log(da + 1e-9)
    log_db = np.log(db + 1e-9)

    matched = 0
    used_b = np.zeros(n, dtype=bool)
    # Match neighbours nearest-first (most discriminative first).
    for a in np.argsort(da):
        best_b = -1
        best_cost = np.inf
        for b in range(n):
            if used_b[b]:
                continue
            logerr = abs(log_da[a] - log_db[b])
            serr = abs(sa[a] - sb[b])
            if logerr > d_tol or serr > s_tol:
                continue
            cost = logerr + serr
            if cost < best_cost:
                best_cost = cost
                best_b = b
        if best_b >= 0:
            used_b[best_b] = True
            matched += 1

    return matched, (matched / m if m else 0.0)


def match_mutual_neighbor(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    *,
    ratio_test: float = 0.6,
    sigma: float = 0.5,
    n_candidates_per_node: int = 8,
    min_spectrum_similarity: float = 0.1,
    neighbor_d_tol: float = 0.6,
    neighbor_s_tol: float = 0.6,
    min_mutual_fraction: float = 0.5,
    physical: bool = False,
    use_ratio_test: bool = True,
) -> list[Correspondence]:
    """Two-pass mutual-neighbour-identity matching.

    Pass 1 (cheap filter): for each A node, use the sorted-spectrum
    descriptor to shortlist the top ``n_candidates_per_node`` B nodes that are
    spectrally similar.  This bounds the expensive second pass.

    Pass 2 (strict): for each candidate central match (A, B), count how many
    of A's SPECIFIC neighbours find a plausible, distinct partner among B's
    neighbours (:func:`_neighbor_consistency`).  A central match is kept only
    if the mutual-neighbour-consistency fraction meets
    ``min_mutual_fraction``, and (optionally) passes a Lowe-style ratio test
    between the best and second-best consistent B candidate.

    No bearing is used anywhere, so a global rotation (the ~2.7 deg found for
    our pair) and y-flip are tolerated inherently.
    """
    ids_a, mat_a, nh_a = _descriptor_matrix(graph_a)
    ids_b, mat_b, nh_b = _descriptor_matrix(graph_b)
    if len(ids_a) == 0 or len(ids_b) == 0 or mat_a.size == 0 or mat_b.size == 0:
        logger.warning("One graph has no usable nodes to match")
        return []

    candidates: dict[int, list[int]] = {}
    for i in range(len(ids_a)):
        vec = mat_a[i]
        diff = mat_b - vec
        sim = np.mean(np.exp(-((diff / sigma) ** 2) / 2.0), axis=1)
        order = np.argsort(sim)[::-1][:n_candidates_per_node]
        candidates[ids_a[i]] = [ids_b[j] for j in order
                                if sim[j] >= min_spectrum_similarity]

    matches: list[Correspondence] = []
    for a_id_a, cands in candidates.items():
        na = nh_a[a_id_a]
        scored: list[tuple[float, int, int, float]] = []  # (fraction, count, b_id, ratio)
        # Evaluate each candidate B and its best distinct second-best for ratio test.
        fracs: list[tuple[float, int, int]] = []
        for b_id in cands:
            nb = nh_b[b_id]
            count, frac = _neighbor_consistency(
                na, nb, d_tol=neighbor_d_tol, s_tol=neighbor_s_tol, physical=physical
            )
            fracs.append((frac, count, b_id))
        fracs.sort(key=lambda t: -t[0])
        if not fracs:
            continue
        best_frac, best_count, best_b = fracs[0]
        second_frac = fracs[1][0] if len(fracs) > 1 else 0.0
        if best_frac < min_mutual_fraction:
            continue
        if use_ratio_test and len(fracs) > 1:
            d_best = 1.0 - best_frac
            d_second = 1.0 - second_frac
            if d_second > 0 and d_best >= ratio_test * d_second:
                continue
        matches.append(Correspondence(
            idx_a=a_id_a,
            idx_b=best_b,
            similarity=best_frac,
            ratio=float((1.0 - best_frac) / (1.0 - second_frac))
            if second_frac < 1.0 else 0.0,
            mutual_consistency=best_count,
            mutual_fraction=best_frac,
        ))
    logger.info("Mutual-neighbour matching: %d A-nodes -> %d matches",
                len(ids_a), len(matches))
    return matches


def matches_to_pixel_pairs(
    graph_a: CraterGraph,
    graph_b: CraterGraph,
    matches: list[Correspondence],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert matches to aligned (N,2) pixel-centre arrays for image A and B.

    Coordinates are in each graph's native pixel frame.  Returns
    ``(points_a, points_b)`` in the order of *matches*.
    """
    pa, pb = [], []
    for m in matches:
        na = graph_a.node(m.idx_a)
        nb = graph_b.node(m.idx_b)
        pa.append((na.cx, na.cy))
        pb.append((nb.cx, nb.cy))
    return np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
