import subprocess
data = subprocess.check_output(['git', 'show', 'HEAD:test/registered/ascend/basic_function/lora/generated_tests_20260521/test_npu_lora_overlap_loading.py'], cwd='d:\\transplant518test\\sglang518')
print(f'Total length: {len(data)}')
print(f'Last 10 bytes: {data[-10:]}')
print(f'Last byte value: {data[-1]}')
print(f'Ends with LF (0x0a): {data[-1] == 0x0a}')
print(f'Ends with CRLF: {data[-2:] == b"\\r\\n"}')