#!/usr/bin/env bash
# Measure the bytes and FLOPs that Q4_0 / IQ4_NL / Q8_0 MUL_MAT kernels move
# and retire on the gfx906 backend. Builds a small ggml benchmark and runs it.
#
# usage: benches/ai-gfx906/run.sh [extra args for mul-mat-ai]

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
build="${BUILD:-$root/build}"
bin="$build/bin"

if [[ ! -f "$bin/libggml.so" ]]; then
    echo "error: $bin/libggml.so not found, build llama.cpp first" >&2
    exit 1
fi

"${CXX:-g++}" -O2 -std=c++17 \
    -I"$root/ggml/include" \
    "$here/mul-mat-ai.cpp" \
    -L"$bin" -lggml -lggml-base -Wl,-rpath,"$bin" \
    -o "$bin/mul-mat-ai"

exec "$bin/mul-mat-ai" --backend-dir "$bin" "$@"
