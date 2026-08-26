"""Validate the RIFT2 reference implementation against the original authors' own
multi-modal test image pairs BEFORE running on our lunar data.

Usage:
    python run_rift_validation.py

Tests against:
- day-night pair (closest to our OHRC/TMC illumination problem)
- infrared-optical pair
- SAR-optical pair

Reports a clear pass/fail for each pair and overall.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add the cloned reference repo to the path
_REF_DIR = Path(__file__).resolve().parent.parent / "rift_reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from src.RIFT2 import RIFT2  # noqa: E402
from src.matcher_functions import match_keypoints_nn, outlier_removal  # noqa: E402

logger = logging.getLogger(__name__)

# Each test pair: (name, dir_name, min_expected_matches, description)
TEST_PAIRS = [
    ("day-night", "day-night", 10, "illumination extreme (closest to our problem)"),
    ("infrared-optical", "infrared-optical", 10, "cross-spectral thermal vs visible"),
    ("sar-optical", "sar-optical", 10, "SAR vs optical (strong NRD)"),
    ("optical-optical", "optical-optical", 20, "same-modality reference baseline"),
]


def validate_pair(
    rift2: RIFT2,
    img_dir: Path,
    pair_name: str,
    min_matches: int,
) -> tuple[bool, str]:
    """Run RIFT2 on one pair. Returns (pass, detail_string)."""
    img1_path = img_dir / "pair1.jpg"
    img2_path = img_dir / "pair2.jpg"

    if not img1_path.exists() or not img2_path.exists():
        return False, f"missing image files: {img1_path}, {img2_path}"

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return False, "cv2.imread returned None"

    t0 = time.time()
    try:
        kp1, des1, kp2, des2 = rift2(img1, img2)
    except Exception as exc:
        return False, f"RIFT2 raised {type(exc).__name__}: {exc}"
    t_rift = time.time() - t0

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return False, f"RIFT2 returned empty features (kp1={len(kp1)}, kp2={len(kp2)})"

    t1 = time.time()
    pts1, pts2, mutual_matches = match_keypoints_nn(
        des1, des2, kp1, kp2, lowes_ratio=0.95, mutual=False
    )
    t_match = time.time() - t1

    n_mutual = len(mutual_matches)

    t2 = time.time()
    try:
        inliers1, inliers2, mask = outlier_removal(pts1, pts2)
    except Exception as exc:
        n_inliers = 0
        mask = []
    else:
        n_inliers = len(inliers1)
    t_ransac = time.time() - t2

    passed = n_mutual >= min_matches
    detail = (
        f"RIFT2: {len(kp1)}+{len(kp2)} kps | "
        f"{n_mutual} matches (ratio 0.95) | "
        f"{n_inliers} inliers (MAGSAC) | "
        f"times: {t_rift:.1f}s detect+describe, {t_match:.3f}s match, {t_ransac:.3f}s ransac"
    )
    return passed, detail


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    images_dir = _REF_DIR / "images"
    if not images_dir.exists():
        logger.error("Reference images directory not found: %s", images_dir)
        return 2

    rift2 = RIFT2(npt=5000, patch_size=96, is_ori=1)

    results = []
    all_passed = True

    print("=" * 72)
    print("RIFT2 Validation Against Authors' Own Test Data")
    print("=" * 72)

    for name, dir_name, min_matches, desc in TEST_PAIRS:
        pair_dir = images_dir / dir_name
        print(f"\n[{name}] {desc}")
        print(f"  images: {pair_dir}")

        passed, detail = validate_pair(rift2, pair_dir, name, min_matches)
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {detail}")
        results.append((name, passed, detail))
        if not passed:
            all_passed = False

    print("\n" + "=" * 72)
    if all_passed:
        print("OVERALL: PASS -- RIFT2 reference implementation validated on all pairs")
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"OVERALL: FAIL on {', '.join(failed)}")
    print("=" * 72)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
