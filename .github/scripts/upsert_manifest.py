#!/usr/bin/env python3
"""Append (or replace) a manifest into a single ``manifests.json``.

All manifests live in one file: ``manifests/manifests.json`` holds an array
under the ``"manifests"`` key. A new build *upserts* by ``image`` (same tag
replaces the previous entry, a new tag appends) and keeps the list newest-first.

No automatic pruning is performed — cleanup is manual (edit the file directly,
then the dashboard reflects whatever remains).

Usage:
    python3 upsert_manifest.py <manifests.json> <new_manifest.json>
"""
import json
import sys
from pathlib import Path


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <manifests.json> <new_manifest.json>", file=sys.stderr)
        return 2

    out_path = Path(args[0])
    new_path = Path(args[1])

    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get("manifests", [])
        except (json.JSONDecodeError, OSError):
            existing = []

    new = json.loads(new_path.read_text(encoding="utf-8"))
    key = new.get("image") or new_path.name

    # Upsert by image tag: drop any previous entry with the same image.
    existing = [m for m in existing if (m.get("image") or "") != key]
    existing.append(new)

    # Newest first.
    existing.sort(key=lambda m: m.get("generated_at") or "", reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"manifests": existing}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifests.json: {len(existing)} entry(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
