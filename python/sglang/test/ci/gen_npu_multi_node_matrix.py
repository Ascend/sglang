"""Generate the NPU multi-node e2e test matrix from CI registries.

Scans registered NPU test files for ``register_npu_ci(..., npu_multi_node={...})``
and groups cases that share the same resource profile (deployment +
prefill/decode/router sizes, or node size for mix) into batch matrix entries.
Cases in one entry run sequentially in a single k8s Pod allocation (Plan A).

The output is a JSON array consumable by GitHub Actions
``strategy.matrix.test_config: ${{ fromJson(...) }}``, e.g.:

    [
      {
        "name": "test_npu_kimi_k2_6_w4a8_1p1d_16p_batch",
        "prefill_size": 1,
        "decode_size": 1,
        "router_size": 1,
        "test_case": "test/registered/.../a.py test/registered/.../b.py",
        "test_type": "perf",
        "prefill_decode_deployment": "separation"
      }
    ]

Usage:
    python gen_npu_multi_node_matrix.py --test-dir test/registered/npu \
        --deployment separation [--filter kimi_k2_6]
"""

import argparse
import glob
import json
import os
import sys

from sglang.test.ci.ci_register import collect_tests


def _tc_name(test_case: str) -> str:
    """test/registered/.../test_npu_xxx.py -> test_npu_xxx"""
    base = test_case.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]


def _group_name(test_cases: list) -> str:
    """Derive a stable group name from the case file names.

    Single case: the case name itself. Multiple cases: the longest common
    name prefix trimmed to a full ``_``-separated token, plus ``_batch``.
    """
    names = [_tc_name(tc) for tc in test_cases]
    if len(names) == 1:
        return names[0]
    prefix = os.path.commonprefix(names)
    # Trim a partial trailing token (e.g. "..._in" from in64k/in128k).
    prefix = prefix[: prefix.rfind("_") + 1]
    if not prefix:
        prefix = "multi_node_"
    return f"{prefix}batch"


def build_matrix(test_dir: str, deployment: str, name_filter: str) -> list:
    pattern = os.path.join(test_dir, "**", "test_*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    registries = collect_tests(files, sanity_check=False)

    groups = {}
    for reg in registries:
        info = reg.npu_multi_node
        if not info or info.get("deployment") != deployment:
            continue
        if reg.disabled is not None:
            continue
        if name_filter and name_filter not in reg.filename:
            continue
        if deployment == "separation":
            key = (
                info["prefill_size"],
                info["decode_size"],
                info["router_size"],
            )
        else:
            key = (info["node_size"],)
        groups.setdefault(key, []).append(reg)

    matrix = []
    for key in sorted(groups):
        regs = sorted(groups[key], key=lambda r: r.filename)
        test_cases = [r.filename for r in regs]
        test_types = {r.npu_multi_node.get("test_type", "perf") for r in regs}
        if len(test_types) != 1:
            raise ValueError(
                f"mixed test_type within one resource group {key}: "
                f"{sorted(test_types)}; split the registrations."
            )
        entry = {
            "name": _group_name(test_cases),
            "test_case": " ".join(test_cases),
            "test_type": test_types.pop(),
            "prefill_decode_deployment": deployment,
        }
        if deployment == "separation":
            entry.update(
                {
                    "prefill_size": key[0],
                    "decode_size": key[1],
                    "router_size": key[2],
                }
            )
        else:
            entry["node_size"] = key[0]
        matrix.append(entry)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        default="test/registered/npu",
        help="root directory of registered NPU test cases",
    )
    parser.add_argument(
        "--deployment",
        required=True,
        choices=["separation", "mix"],
        help="which multi-node deployment mode to generate",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="only include cases whose path contains this substring",
    )
    args = parser.parse_args()

    matrix = build_matrix(args.test_dir, args.deployment, args.filter)
    json.dump(matrix, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
