"""Verify installed production sources and actual default A3 resources."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu
import sglang

manifest = json.loads(Path(__file__).with_name('production-manifest.json').read_text())
repo = Path(sglang.__file__).resolve().parents[2]
files = {}
for name, expected in manifest['files'].items():
    path = repo / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)
    files[name] = actual
for name in ('VLM_DIAGNOSTICS_DIR', 'VLM_SUPPORTED_ONLY', 'VLM_SUPPORTED_MATMUL',
             'VLM_CANN_ROUTE_BUDGET', 'VLM_DEVICE_LIMIT_CUBE', 'VLM_DEVICE_LIMIT_VECTOR'):
    assert not os.environ.get(name), name
assert not any(name.startswith('npu_vlm_') for name in sys.modules)
devices = []
for index in (0, 1):
    torch.npu.set_device(index)
    limits = torch_npu.npu.get_device_limit(index)
    assert limits == {'cube_core_num': 24, 'vector_core_num': 48}, limits
    devices.append({'index': index, 'limits': limits,
                    'properties': str(torch.npu.get_device_properties(index))})
record = {'package': sglang.__file__, 'files': files, 'devices': devices,
          'diagnostic_hooks_loaded': False}
Path('artifacts/npu-vlm/machine/production.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
