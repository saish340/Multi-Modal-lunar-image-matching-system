"""Extract text from a PDF, optionally filtering to pages containing keywords."""
from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    pdf_path = Path(sys.argv[1])
    keywords = [k.lower() for k in sys.argv[2:]] or []
    reader = PdfReader(str(pdf_path))
    print(f"PAGES: {len(reader.pages)}")
    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            print(f"--- page {idx + 1} ERROR {exc}")
            continue
        low = text.lower()
        if not keywords or any(k in low for k in keywords):
            print(f"===== PAGE {idx + 1} =====")
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
