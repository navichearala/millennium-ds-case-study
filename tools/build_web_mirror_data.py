"""Regenerate web-mirror/js/data.js from the pipeline output.

The browser mirror has no backend, so it ships the dataset as a JavaScript literal. That
snapshot has to be rebuilt whenever the pipeline re-runs, or the mirror quietly reports
different numbers from the Streamlit app and the JSON exports - which is exactly what
happened before this script existed. Run it after every `python src/pipeline.py`.

    python tools/build_web_mirror_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402  - needs the path insert above

TARGET = ROOT / "web-mirror" / "js" / "data.js"


def build() -> dict:
    candidates = json.loads(Path(config.CANDIDATES_JSON).read_text())

    # The mirror shows the source document behind each record, so a reviewer can audit an
    # extracted field without leaving the page. Keyed by candidate_id to match the app.
    raw_text: dict[str, str] = {}
    by_source = {c["source_file"]: c["candidate_id"] for c in candidates}
    for path in sorted(Path(config.RAW_TEXT_DIR).glob("*.txt")):
        match = next((cid for src, cid in by_source.items() if Path(src).stem == path.stem), None)
        if match is None:
            print(f"  warning: no candidate matches raw text {path.name}", file=sys.stderr)
            continue
        raw_text[match] = path.read_text(encoding="utf-8")

    missing = sorted({c["candidate_id"] for c in candidates} - set(raw_text))
    if missing:
        raise SystemExit(f"raw text missing for: {', '.join(missing)}")

    return {
        "candidates": candidates,
        "raw_text": raw_text,
        "generated": "pipeline output from outputs/candidates.json",
    }


if __name__ == "__main__":
    payload = build()
    TARGET.write_text(
        "window.PLATFORM_DATA = "
        + json.dumps(payload, separators=(",", ":"), default=str)
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {TARGET.relative_to(ROOT)} - {len(payload['candidates'])} candidates, "
          f"{len(payload['raw_text'])} source documents")
