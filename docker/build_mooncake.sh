#!/usr/bin/env bash
set -eo pipefail

# 在 sglang_ljx 容器中执行，只生成 whl；BUILD_MEMFABRIC=1 时先编译 MemFabric。
PYTHON_BIN="${PYTHON_BIN:-python3}"
MOONCAKE_DIR="${MOONCAKE_DIR:-/mnt/share/Mooncake}"
MOONCAKE_DIR="$(cd -- "$MOONCAKE_DIR" && pwd)"
MEMFABRIC_DIR="${MEMFABRIC_DIR:-/mnt/share/memfabric_hybrid}"
BUILD_MEMFABRIC="${BUILD_MEMFABRIC:-1}"
BUILD_DIR="${BUILD_DIR:-$MOONCAKE_DIR/build/memfabric-wheel}"
WHEEL_DIR="${WHEEL_DIR:-$MOONCAKE_DIR/dist}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS:-16}"

PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"

# --no-build-isolation 要求构建依赖安装在同一个 Python 环境中。
case "${INSTALL_DEPS:-1}" in
  1)
    "$PYTHON_BIN" -m pip install \
      'scikit-build-core>=1.0' 'pybind11>=2.13' 'numpy>=1.24' \
      'setuptools>=61' 'pip>=23' wheel
    ;;
  0) ;;
  *) echo "INSTALL_DEPS must be 0 or 1" >&2; exit 2 ;;
esac
if ! "$PYTHON_BIN" -c 'import scikit_build_core.build, pybind11, numpy'; then
  echo "Python build dependencies are unavailable in $PYTHON_BIN." >&2
  echo "Run this script with INSTALL_DEPS=1 to install them." >&2
  exit 1
fi


mkdir -p "$BUILD_DIR" "$WHEEL_DIR"
BUILD_DIR="$(cd -- "$BUILD_DIR" && pwd)"
WHEEL_DIR="$(cd -- "$WHEEL_DIR" && pwd)"

if [[ "$BUILD_MEMFABRIC" == 1 ]]; then
  if [[ ! -f "$MEMFABRIC_DIR/script/build_and_pack_run.sh" ]]; then
    echo "MemFabric build script not found: $MEMFABRIC_DIR/script/build_and_pack_run.sh" >&2
    exit 1
  fi
  (
    # MemFabric 使用 python3 和 PYTHON_HOME，统一到 Mooncake 选择的 Python。
    unset multiple_python
    export PYTHON_HOME="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
    export PATH="$PYTHON_HOME/bin:$PATH"
    export MF_BUILD_JOBS="${BUILD_JOBS:-16}"
    bash "$MEMFABRIC_DIR/script/build_and_pack_run.sh" \
      --build_mode RELEASE --build_open_abi OFF --build_python ON --xpu_type NPU \
      --build_hcom OFF --build_hcom_rdma ON --build_etcd_backend OFF
  )
  cp "$MEMFABRIC_DIR"/output/memfabric_hybrid/wheel/memfabric_hybrid-*.whl "$WHEEL_DIR/"
elif [[ "$BUILD_MEMFABRIC" != 0 ]]; then
  echo "BUILD_MEMFABRIC must be 0 or 1" >&2
  exit 2
fi

# 当前适配在运行时 dlopen MemFabric，编译 Mooncake 不需要其头文件或预先安装的包。
export CMAKE_PREFIX_PATH="$("$PYTHON_BIN" -m pybind11 --cmakedir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# 优先复用现有 YLT，缺失时下载 Mooncake 当前基线使用的依赖版本。
YLT_VERSION=7801bc9ad9021781f15217552214e325a1cf7373
YLT_SOURCE_DIR="${YLT_SOURCE_DIR:-$MOONCAKE_DIR/extern/yalantinglibs}"
if [[ ! -f "$YLT_SOURCE_DIR/CMakeLists.txt" ]]; then
  YLT_SOURCE_DIR="$BUILD_DIR/thirdparty/yalantinglibs-$YLT_VERSION"
  if [[ ! -f "$YLT_SOURCE_DIR/CMakeLists.txt" ]]; then
    mkdir -p "$YLT_SOURCE_DIR"
    curl -k --fail --location --retry 3 \
      "https://codeload.github.com/alibaba/yalantinglibs/tar.gz/$YLT_VERSION" \
      | tar xz --strip-components=1 -C "$YLT_SOURCE_DIR"
  fi
fi

YLT_SOURCE_DIR="$(cd -- "$YLT_SOURCE_DIR" && pwd)"
cd "$MOONCAKE_DIR"
"$PYTHON_BIN" -m pip wheel . --no-deps --no-build-isolation --verbose \
  --wheel-dir "$WHEEL_DIR" -Cbuild-dir="$BUILD_DIR" \
  -Ccmake.define.FETCHCONTENT_SOURCE_DIR_YALANTINGLIBS="$YLT_SOURCE_DIR" \
  -Ccmake.define.WITH_STORE=ON -Ccmake.define.USE_TENT=ON -Ccmake.define.USE_MEMFABRIC=ON \
  -Ccmake.define.USE_CUDA=OFF -Ccmake.define.USE_ASCEND=OFF -Ccmake.define.USE_ASCEND_DIRECT=OFF \
  -Ccmake.define.WITH_STORE_RUST=OFF -Ccmake.define.WITH_STORE_RS=OFF -Ccmake.define.WITH_STORE_GO=OFF \
  -Ccmake.define.BUILD_SHARED_LIBS=ON -Ccmake.define.BUILD_UNIT_TESTS=OFF \
  -Ccmake.define.BUILD_EXAMPLES=OFF -Ccmake.define.BUILD_BENCHMARK=OFF \
  -Ccmake.define.ENABLE_DEBUG_SYMBOLS=OFF

printf '\nWheel 输出目录：%s\n' "$WHEEL_DIR"
ls -lh "$WHEEL_DIR"/*.whl
