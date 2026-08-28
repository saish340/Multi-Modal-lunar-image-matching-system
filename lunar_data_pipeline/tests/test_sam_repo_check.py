"""Unit tests for sam_repo_check using a faked GitHub API fetcher.

No live network calls: ``set_fetcher`` lets us supply canned API responses so
the classification logic is tested deterministically.
"""

from __future__ import annotations

import json

import pytest

from lunar_data_pipeline import sam_repo_check as src
from lunar_data_pipeline.sam_repo_check import (
    PRIMARY_REPO,
    RepoStatus,
    build_report,
    check_repo,
    search_repos,
    set_fetcher,
)


def _repo_payload(full_name: str, **overrides) -> dict:
    payload = {
        "full_name": full_name,
        "description": "some crater thing",
        "archived": False,
        "stargazers_count": 3,
        "default_branch": "main",
    }
    payload.update(overrides)
    return payload


def _tree_payload(files: list[str]) -> dict:
    return {"tree": [{"path": p, "type": "blob"} for p in files]}


@pytest.fixture
def api(full_name: str = "vam/Universal-Crater-Detection-with-SAM"):
    tree_files = ["README.md", "SAM_Image mask generation.ipynb", "images.zip"]

    def fetcher(url: str) -> bytes:
        if url.startswith("https://api.github.com/repos/"):
            if url.endswith("/git/trees/main?recursive=1"):
                return json.dumps(_tree_payload(tree_files)).encode()
            return json.dumps(_repo_payload(full_name)).encode()
        # search
        return json.dumps({"items": []}).encode()

    set_fetcher(fetcher)
    yield
    set_fetcher(src._default_fetch)


class TestCheckRepo:
    def test_minimal_repo_classification(self, api):
        status = check_repo(PRIMARY_REPO)
        assert status.exists is True
        assert status.has_source_code is True  # the .ipynb counts as code
        assert status.has_requirements is False
        assert status.has_weights is False
        assert status.is_usable is False  # no requirements -> not usable

    def test_repo_not_found(self):
        def fetcher(url: str):
            if "git/trees" in url:
                return json.dumps({"tree": []}).encode()
            return json.dumps({"message": "Not Found"}).encode()

        set_fetcher(fetcher)
        try:
            status = check_repo("does/not-exist")
            assert status.exists is False
            assert status.fetch_error == "Not Found"
        finally:
            set_fetcher(src._default_fetch)

    def test_usable_repo_with_weights(self):
        def fetcher(url: str):
            if "git/trees" in url:
                return json.dumps(
                    _tree_payload(
                        ["crater_detector.py", "requirements.txt", "model.pth"]
                    )
                ).encode()
            return json.dumps(_repo_payload("a/b")).encode()

        set_fetcher(fetcher)
        try:
            status = check_repo("a/b")
            assert status.has_source_code is True
            assert status.has_requirements is True
            assert status.has_weights is True
            assert status.is_usable is True
        finally:
            set_fetcher(src._default_fetch)

    def test_readme_only_yields_no_code(self):
        def fetcher(url: str):
            if "git/trees" in url:
                return json.dumps(_tree_payload(["README.md"])).encode()
            return json.dumps(_repo_payload("a/b")).encode()

        set_fetcher(fetcher)
        try:
            status = check_repo("a/b")
            assert status.has_source_code is False
            assert status.is_usable is False
        finally:
            set_fetcher(src._default_fetch)


class TestClassifyFiles:
    def test_markers(self):
        source, reqs, weights = src._classify_files(
            ["x.py", "requirements.txt", "weights.pth", "README.md"]
        )
        assert source is True and reqs is True and weights is True

    def test_nested_requirements(self):
        _, reqs, _ = src._classify_files(["sub/requirements.txt"])
        assert reqs is True


class TestSearchAndReport:
    def test_search_returns_items(self):
        data = [
            {
                "full_name": "x/y",
                "stargazers_count": 2,
                "archived": False,
                "description": "d",
                "pushed_at": "2026-01-01T00:00:00Z",
            }
        ]

        def fetcher(url: str):
            return json.dumps({"items": data}).encode()

        set_fetcher(fetcher)
        try:
            results = search_repos("segment anything crater")
            assert results[0]["full_name"] == "x/y"
        finally:
            set_fetcher(src._default_fetch)

    def test_build_report_verdict_minimal(self, api):
        report = build_report(PRIMARY_REPO)
        assert report["verdict"] == "exists_source_no_weights"
        assert report["integration_recommendation"].startswith("NOT recommended")
        assert "Giannakis" in report["integration_recommendation"]

    def test_build_report_verdict_nonexistent(self):
        def fetcher(url: str):
            return json.dumps({"message": "Not Found"}).encode()

        set_fetcher(fetcher)
        try:
            report = build_report("does/not-exist")
            assert report["verdict"] == "does_not_exist"
        finally:
            set_fetcher(src._default_fetch)
