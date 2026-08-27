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

  3. ``--build-arg`` / ``--build-args-file``: the ``KEY=VALUE`` pairs the caller
     workflow actually passed to ``docker build``. These win over the Dockerfile
     ARG defaults, so the manifest reports what this image was *really* built
     with (e.g. ``DEVICE_TYPE=910b``) instead of the default (``a3``).

The merged manifest is written to ``--output`` as JSON.

Usage:
    python3 generate_image_manifest.py \
        --image      swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B021 \
        --dockerfile docker/npu_kvtc.Dockerfile \
        --image-data image-data.json \
        --build-arg  DEVICE_TYPE=a3 \
        --output     image-manifest.json
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

_PIN_RE = re.compile(r"([A-Za-z0-9_.\[\]+-]+(?:==|>=|<=|~=|!=|>|<)[A-Za-z0-9_.\[\]+.*-]+)")

# Build args that carry no useful information in the manifest:
#   TARGETARCH  docker builtin, differs per platform in a multi-arch build, so
#               a single manifest has no one correct value for it
#   APTMIRROR   defaults to "" and is only a build-time apt source tweak
# They stay available for ``${VAR}`` resolution, they are just not reported.
_HIDDEN_BUILD_ARGS = ("TARGETARCH", "APTMIRROR")


def load_image_data(path):
    """Load runtime-extracted data from the image, or return {} if unavailable."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"::warning::Failed to parse image-data {path}: {exc}", file=sys.stderr)
    return {}


def parse_build_args(entries):
    """Parse ``KEY=VALUE`` strings (docker ``--build-arg`` format) into a dict.

    Blank lines and ``#`` comments are skipped so the caller can feed in a
    heredoc'd workflow input verbatim. Entries with an *empty* value are
    dropped on purpose: an unset ``${{ matrix.foo }}`` expands to an empty
    string in GitHub Actions, and silently wiping a good Dockerfile default is
    worse than falling back to it.
    """
    parsed = {}
    for raw in entries:
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        if "=" not in entry:
            print(f"::warning::Ignoring build arg {entry!r} (expected KEY=VALUE)", file=sys.stderr)
            continue
        key, value = entry.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if not value:
            print(f"::warning::Ignoring empty build arg {key} (keeping Dockerfile default)", file=sys.stderr)
            continue
        parsed[key] = value
    return parsed


def load_build_args(path, extra):
    """Merge build args from a ``KEY=VALUE`` file and repeated ``--build-arg``."""
    overrides = {}
    if path:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                overrides.update(parse_build_args(fh))
        else:
            print(f"::warning::build-args-file {path} not found", file=sys.stderr)
    overrides.update(parse_build_args(extra or []))
    return overrides


def parse_dockerfile(path, overrides=None):
    """Extract ARG defaults, git-clone lines and pip-install pins from a Dockerfile.

    ``overrides`` (the args actually passed to ``docker build``) take precedence
    over the ARG defaults.

    ``${VAR}`` references are resolved against the resulting values so that the
    declared pins (e.g. ``torch==${PYTORCH_VERSION}``) are recorded with their
    concrete values.
    """
    declared = {"args": {}, "git_clones": [], "pip_installs": [], "pip_pins": []}
    if not path or not os.path.exists(path):
        declared["args"].update(overrides or {})
        return declared

    with open(path, encoding="utf-8") as fh:
        lines = [raw.rstrip("\n") for raw in fh]

    # Pass 1: collect ARG defaults (defined before use in these Dockerfiles).
    #
    # An ARG must be re-declared after FROM to be visible inside the build
    # stage, and that re-declaration carries no default (`ARG DEVICE_TYPE`).
    # Letting it overwrite the earlier `ARG DEVICE_TYPE=a3` would blank the
    # value, so a defaultless ARG only ever *introduces* a key.
    for line in lines:
        m = re.match(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*(.*))?$", line.strip())
        if m:
            name = m.group(1)
            default = (m.group(2) or "").strip().strip('"')
            if default or name not in declared["args"]:
                declared["args"][name] = default

    # What the build really used beats what the Dockerfile defaults to.
    declared["args"].update(overrides or {})

    def resolve(text):
        # Both ``${VAR}`` and bare ``$VAR`` — the Dockerfiles use either form
        # (e.g. ``git clone ... --branch $SGLANG_TAG``). Unknown names are left
        # untouched rather than blanked.
        def sub(m):
            name = m.group(1) or m.group(2)
            value = declared["args"].get(name)
            return m.group(0) if value is None else value

        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
            sub,
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
    overrides = load_build_args(args.build_args_file, args.build_arg)
    declared = parse_dockerfile(args.dockerfile, overrides)

    # Cross-reference the sglang clone (declared) with the resolved commit.
    sglang = image_data.get("sglang", {})
    if declared["git_clones"]:
        clone = declared["git_clones"][0]
        sglang.setdefault("remote", clone["remote"])
        sglang.setdefault("branch", clone["branch"])
        sglang.setdefault("dir", clone["dir"])

    pip_packages = image_data.get("pip", [])

    reported_args = {
        k: v for k, v in declared["args"].items() if k not in _HIDDEN_BUILD_ARGS
    }

    manifest = {
        "schema_version": 1,
        "image": args.image,
        "dockerfile": args.dockerfile or "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_args": reported_args,
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
    parser.add_argument(
        "--build-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Build arg actually passed to docker build (repeatable); overrides ARG defaults",
    )
    parser.add_argument(
        "--build-args-file",
        default="",
        help="File of KEY=VALUE build args, one per line; overrides ARG defaults",
    )
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
    build_args = manifest["build_args"]
    print(f"  cann/device   : {build_args.get('CANN_VERSION') or '<empty>'}"
          f" / {build_args.get('DEVICE_TYPE') or '<empty>'}")


if __name__ == "__main__":
    main()
