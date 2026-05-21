# Generated NPU Test Cases

This directory contains test cases generated based on the NPU vs GPU Test Gap Analysis report.

## Overview

- **Generated Date**: 2026-05-19 14:47:45
- **High Priority Tests**: 5
- **Medium Priority Tests**: 4
- **Low Priority Tests**: 0

## Test Files

### High Priority Tests (Core Functionality)

| Test File | Description |
|-----------|-------------|
| test_npu_lora_basic.py | LoRA adapter loading, batch inference |
| test_npu_lora_tp.py | LoRA with tensor parallelism (TP=2, TP=4) |
| test_npu_lora_radix_cache.py | LoRA with radix cache |
| test_npu_lora_multi_adapter.py | Multiple LoRA adapters |
| test_npu_pipeline_parallel.py | Pipeline parallelism (PP=2, PP=4) |

### Medium Priority Tests (Model Coverage and Features)

| Test File | Description |
|-----------|-------------|
| test_npu_gguf.py | GGUF quantization support |
| test_npu_engine_consistency.py | Engine vs Runtime consistency |
| test_npu_deterministic.py | Deterministic output with seed |
| test_npu_scheduler_control.py | Scheduler control (abort, pause/resume) |

## Prerequisites

1. Ascend NPU environment set up
2. SGLang installed with NPU support
3. Test models downloaded and paths updated in test files

## Configuration

Before running tests, update the following in each test file:

```python
# Update model paths
MODEL_NAME = "your/model/path"
ADAPTER_URL = "your/adapter/path"
GGUF_MODEL_Q4 = "your/gguf/model-Q4_K_M.gguf"
```

## Usage

### Run all generated tests:
```bash
pytest test/registered/ascend/generated_tests/ -v
```

### Run specific priority tests:
```bash
# High priority only
pytest test/registered/ascend/generated_tests/test_npu_lora_*.py -v

# Medium priority only
pytest test/registered/ascend/generated_tests/test_npu_gguf.py -v
```

### Run with specific markers:
```bash
pytest test/registered/ascend/generated_tests/ -v -m "not slow"
```

## Test Structure

Each test file follows this structure:
- **Test Class**: Contains all tests for a specific feature
- **Fixtures**: Server lifecycle management (start/stop)
- **Test Methods**: Individual test cases

## Notes

- Tests use placeholder model names - update before running
- Some tests (TP=8, PP=4) may require multiple NPUs
- Logprob thresholds may need tuning for specific models
- Tests are designed for integration testing, not unit testing

## Troubleshooting

### Server startup timeout
Increase timeout in `wait_for_server_ready()` call:
```python
wait_for_server_ready(timeout=600)  # 10 minutes
```

### Out of memory
Reduce batch size or use smaller models:
```python
MODEL_NAME = "smaller/model"  # Use a smaller model
```

### Connection refused
Ensure NPU drivers and CANN toolkit are properly installed.

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Include bilingual docstrings (English/Chinese)
3. Add proper error messages
4. Update this README
