"""Factually verify whether a usable SAM-based crater detector exists.

Context: earlier sessions floated ``vam/Universal-Crater-Detection-with-SAM``
as a candidate for integration, but the claim was never verified. This module
queries the GitHub API and reports honestly whether any SAM-based crater
detector repo (a) actually exists, (b) has real code (not just a README),
(c) ships downloadable model weights, and (d) corresponds to the peer-reviewed
Giannakis et al. 2024 *Icarus* method (``arXiv:2304.07764``,
DOI ``10.1016/j.icarus.2023.115797``).

It never recommends integration on the strength of a repo name alone; the
verification result is the deliverable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("lunar_data_pipeline.sam_repo_check")

#: Target repo from the earlier (unverified) session claim.
PRIMARY_REPO = "vam/Universal-Crater-Detection-with-SAM"

#: GitHub API search terms for locating alternatives.
SEARCH_QUERIES = [
    "segment anything crater detection",
    "SAM lunar crater detection",
]

#: The peer-reviewed reference we are screening repos against.
GIANNAKIS_REFERENCE = {
    "authors": "Giannakis, Bhardwaj, Sam, Leontidis",
    "title": "A flexible deep learning crater detection scheme using Segment "
    "Anything Model (SAM)",
    "journal": "Icarus 408, 115797 (2024)",
    "arxiv": "arXiv:2304.07764",
    "doi": "10.1016/j.icarus.2023.115797",
}


#: Default URL opener (swappable in tests to avoid live network).
def _default_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "opencode-sam-check"})
    return urllib.request.urlopen(req, timeout=30).read()


_fetch: Callable[[str], bytes] = _default_fetch


def set_fetcher(fn: Callable[[str], bytes]) -> None:
    """Install a custom URL fetcher (used by tests to fake the API)."""
    global _fetch  # noqa: PLW0603
    _fetch = fn


def _get_json(url: str) -> dict | list:
    raw = _fetch(url)
    return json.loads(raw)


@dataclass
class RepoStatus:
    """Factual findings for one candidate repository."""

    full_name: str = ""
    exists: bool = False
    description: str | None = None
    archived: bool = False
    stargazers: int = 0
    default_branch: str | None = None
    files: list[str] = field(default_factory=list)
    has_requirements: bool = False
    has_weights: bool = False
    has_source_code: bool = False
    fetch_error: str | None = None

    @property
    def is_usable(self) -> bool:
        """Rough usability signal: real source + dependency manifest.

        Downloadability of actual *trained* model weights is a separate,
        stronger requirement tracked by ``has_weights``/``weights_note``.
        """
        return self.exists and self.has_source_code and self.has_requirements

    def to_dict(self) -> dict:
        return {
            "full_name": self.full_name,
            "exists": self.exists,
            "archived": self.archived,
            "stargazers": self.stargazers,
            "description": self.description,
            "default_branch": self.default_branch,
            "files": self.files,
            "has_requirements": self.has_requirements,
            "has_weights": self.has_weights,
            "has_source_code": self.has_source_code,
            "fetch_error": self.fetch_error,
            "is_usable": self.is_usable,
        }


_CODE_MARKERS = {".py", ".ipynb", ".sh", ".cpp", ".h", ".yaml", ".yml", ".cfg", ".toml"}
_REQUIREMENTS_MARKERS = {"requirements.txt", "environment.yml", "pyproject.toml", "setup.py"}
_WEIGHT_MARKERS = {".pth", ".ckpt", ".pt", ".onnx", ".bin", ".h5", ".wts", ".pdparams"}


def _classify_files(paths: list[str]) -> tuple[bool, bool, bool]:
    """Return (has_source_code, has_requirements, has_weights) from paths."""
    source = any(p.endswith(tuple(_CODE_MARKERS)) for p in paths)
    reqs = any(p.endswith("/" + r) or p == r for r in _REQUIREMENTS_MARKERS for p in paths)
    weights = any(p.endswith(tuple(_WEIGHT_MARKERS)) for p in paths)
    return source, reqs, weights


def check_repo(full_name: str) -> RepoStatus:
    """Check one GitHub repo via the API and classify its tree."""
    status = RepoStatus(full_name=full_name)
    try:
        meta = _get_json(f"https://api.github.com/repos/{full_name}")
        if isinstance(meta, dict) and meta.get("id") is None and meta.get("full_name") is None:
            status.fetch_error = str(meta.get("message", "not found"))
            return status
        status.exists = True
        status.description = meta.get("description")
        status.archived = bool(meta.get("archived"))
        status.stargazers = int(meta.get("stargazers_count") or 0)
        status.default_branch = meta.get("default_branch")
    except urllib.error.HTTPError as exc:
        status.fetch_error = f"HTTP {exc.code}"
        return status
    except Exception as exc:  # network / parse errors
        status.fetch_error = str(exc)
        return status

    if not status.default_branch:
        return status
    try:
        tree = _get_json(
            f"https://api.github.com/repos/{full_name}/git/trees/"
            f"{status.default_branch}?recursive=1"
        )
    except Exception as exc:
        status.fetch_error = str(exc)
        return status
    if not isinstance(tree, dict):
        return status

    paths = [t.get("path", "") for t in tree.get("tree", [])]
    files = sorted(p for p in paths)
    status.files = files
    source, reqs, weights = _classify_files(paths)
    status.has_source_code = source
    status.has_requirements = reqs
    status.has_weights = weights
    return status


def search_repos(query: str) -> list[dict]:
    """GitHub code/repo search for SAM crater detection; returns top items."""
    import urllib.parse

    q = urllib.parse.quote(query)
    try:
        data = _get_json(
            f"https://api.github.com/search/repositories?q={q}&sort=stars&per_page=8"
        )
    except Exception as exc:
        logger.warning("Search failed for %r: %s", query, exc)
        return []
    if not isinstance(data, dict):
        return []
    return [
        {
            "full_name": item.get("full_name"),
            "stars": item.get("stargazers_count"),
            "archived": item.get("archived"),
            "description": item.get("description"),
            "pushed_at": item.get("pushed_at"),
        }
        for item in data.get("items", [])
    ]


def build_report(primary: str = PRIMARY_REPO) -> dict:
    """Assemble the full factual verification report."""
    primary_status = check_repo(primary)

    searches = {q: search_repos(q) for q in SEARCH_QUERIES}

    verdict: str
    if not primary_status.exists:
        verdict = "does_not_exist"
    elif primary_status.is_usable and primary_status.has_weights:
        verdict = "usable_with_weights"
    elif primary_status.is_usable:
        verdict = "usable_no_weights"
    elif primary_status.has_source_code:
        verdict = "exists_source_no_weights"
    else:
        verdict = "exists_minimal_no_code"

    return {
        "schema_version": "1.0",
        "primary_repo": primary_status.to_dict(),
        "peer_reviewed_method": GIANNAKIS_REFERENCE,
        "verdict": verdict,
        "integration_recommendation": (
            "NOT recommended for direct integration yet. "
            + (
                "The named repo does not exist."
                if not primary_status.exists
                else "The named repo is a minimal demo (README + a single notebook "
                "+ image archive) with no released weights, no dependency manifest, "
                "and no trainable code, so it is not directly usable. The actual "
                "peer-reviewed method is the Giannakis et al. 2024 Icarus work, which "
                "would need SAM weights from Meta (downloadable) and the paper's "
                "shape-index post-processing implemented from scratch, plus a GPU."
            )
        ),
        "searches": searches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify SAM-based crater detector repos against the GitHub API."
    )
    parser.add_argument(
        "--repo",
        default=PRIMARY_REPO,
        help=f"Repo to check (default {PRIMARY_REPO}).",
    )
    parser.add_argument(
        "--output",
        type=__import__("pathlib").Path,
        default=__import__("pathlib").Path("sam_repo_report.json"),
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    report = build_report(primary=args.repo)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"Primary repo {args.repo}: exists={report['primary_repo']['exists']} "
          f"verdict={report['verdict']}")
    print(f"  source_code={report['primary_repo']['has_source_code']} "
          f"requirements={report['primary_repo']['has_requirements']} "
          f"weights={report['primary_repo']['has_weights']} "
          f"files={len(report['primary_repo']['files'])}")
    print(f"Integration recommendation: {report['integration_recommendation']}")
    print(f"Report written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
