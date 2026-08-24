#!/usr/bin/env python3
"""Generate an image dependency manifest (JSON) for the sglang NPU images.

This script merges two sources of truth:

  1. ``--image-data``: a JSON file produced by running the built image
     (see ``.github/workflows/generate-image-manifest.yml``). It contains the
     *resolved* dependency inventory:
       - ``pip``        -> ``python3 -m pip list --format=json``
       - ``sglang``     -> git remote / branch / commit / describe / commit date
       - ``os``         -> distro / kernel / arch
       - ``cann``       -> Ascend toolkit path & version files

  2. ``--dockerfile``: static analysis of the Dockerfile itself. It records the
     *declared* build configuration even when the image cannot be inspected:
       - ``ARG`` default values (CANN_VERSION, PYTORCH_VERSION, SGLANG_TAG, ...)
       - ``git clone`` lines (sglang remote / branch / destination)
       - ``pip install`` lines and their ``name==version`` pins

The merged manifest is written to ``--output`` as JSON.

Usage:
    python3 generate_image_manifest.py \
        --image      swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B021 \
        --dockerfile docker/npu_kvtc.Dockerfile \
        --image-data image-data.json \
        --output     image-manifest.json
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

_PIN_RE = re.compile(r"([A-Za-z0-9_.\[\]+-]+(?:==|>=|<=|~=|!=|>|<)[A-Za-z0-9_.\[\]+.*-]+)")


def load_image_data(path):
    """Load runtime-extracted data from the image, or return {} if unavailable."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"::warning::Failed to parse image-data {path}: {exc}", file=sys.stderr)
    return {}


def parse_dockerfile(path):
    """Extract ARG defaults, git-clone lines and pip-install pins from a Dockerfile.

    ``${VAR}`` references are resolved against the ARG defaults so that the
    declared pins (e.g. ``torch==${PYTORCH_VERSION}``) are recorded with their
    concrete values.
    """
    declared = {"args": {}, "git_clones": [], "pip_installs": [], "pip_pins": []}
    if not path or not os.path.exists(path):
        return declared

    with open(path, encoding="utf-8") as fh:
        lines = [raw.rstrip("\n") for raw in fh]

    # Pass 1: collect ARG defaults (defined before use in these Dockerfiles).
    for line in lines:
        m = re.match(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*(.*))?$", line.strip())
        if m:
            declared["args"][m.group(1)] = (m.group(2) or "").strip().strip('"')

    def resolve(text):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda m: declared["args"].get(m.group(1), m.group(0)),
            text,
        )

    # Pass 2: git clones and pip installs (skip ARG/ENV/LABEL definitions).
    for line in lines:
        s = line.strip()
        if s.startswith(("ARG ", "ENV ", "LABEL ")):
            continue
        # git clone <remote> --branch <branch> <dir>
        m = re.search(r"git\s+clone\s+(\S+).*?(?:--branch|-b)\s+(\S+)\s+(\S+)", s)
        if m:
            declared["git_clones"].append(
                {"remote": resolve(m.group(1)), "branch": resolve(m.group(2)), "dir": m.group(3)}
            )
        # pip install (via ${PIP_INSTALL} or literal "pip install")
        if "${PIP_INSTALL}" in s or re.search(r"\bpip3?\s+install\b", s):
            declared["pip_installs"].append(s)
            for pin in _PIN_RE.findall(resolve(s)):
                if pin not in declared["pip_pins"]:
                    declared["pip_pins"].append(pin)

    return declared


def build_manifest(args):
    image_data = load_image_data(args.image_data)
    declared = parse_dockerfile(args.dockerfile)

    # Cross-reference the sglang clone (declared) with the resolved commit.
    sglang = image_data.get("sglang", {})
    if declared["git_clones"]:
        clone = declared["git_clones"][0]
        sglang.setdefault("remote", clone["remote"])
        sglang.setdefault("branch", clone["branch"])
        sglang.setdefault("dir", clone["dir"])

    pip_packages = image_data.get("pip", [])

    manifest = {
        "schema_version": 1,
        "image": args.image,
        "dockerfile": args.dockerfile or "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_args": declared["args"],
        "git_clones": declared["git_clones"],
        "declared_pip_installs": declared["pip_installs"],
        "declared_pip_pins": declared["pip_pins"],
        "sglang": sglang,
        "os": image_data.get("os", {}),
        "cann": image_data.get("cann", {}),
        "pip": {
            "count": len(pip_packages),
            "packages": pip_packages,
        },
    }
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Full image reference (registry/repo:tag)")
    parser.add_argument("--dockerfile", default="", help="Path to the Dockerfile")
    parser.add_argument("--image-data", default="", help="Path to runtime-extracted JSON")
    parser.add_argument("--output", default="image-manifest.json", help="Output manifest path")
    args = parser.parse_args(argv)

    manifest = build_manifest(args)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Wrote manifest to {args.output}")
    print(f"  image         : {manifest['image']}")
    print(f"  pip packages  : {manifest['pip']['count']}")
    print(f"  sglang commit : {manifest['sglang'].get('commit', '<unknown>')}")


if __name__ == "__main__":
    main()
