#!/usr/bin/env bash

# Source this file so the exports remain active in the current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this script with: source ./setup_env.sh" >&2
    exit 2
fi

_prepend_path() {
    local variable="$1"
    local directory="$2"
    local current="${!variable:-}"

    case ":${current}:" in
        *":${directory}:"*) ;;
        *) export "${variable}=${directory}${current:+:${current}}" ;;
    esac
}

if [[ -n "${TILELANG_ROOT:-}" ]]; then
    _tilelang_candidates=("${TILELANG_ROOT}")
else
    _tilelang_candidates=(
        "/app/tilelang-metax"
        "/data/tilelang-metax"
    )
fi

TILELANG_ROOT=""
for _candidate in "${_tilelang_candidates[@]}"; do
    if [[ -f "${_candidate}/tilelang/__init__.py" && \
          -f "${_candidate}/build/lib/libtilelang.so" ]]; then
        TILELANG_ROOT="${_candidate}"
        break
    fi
done

if [[ -z "${TILELANG_ROOT}" ]]; then
    echo "Could not find a built TileLang development tree." >&2
    echo "Set TILELANG_ROOT before sourcing this script, for example:" >&2
    echo "  TILELANG_ROOT=/app/tilelang-metax source ./setup_env.sh" >&2
    unset _tilelang_candidates _candidate
    unset -f _prepend_path
    return 1
fi

export TILELANG_ROOT
export MACA_PATH="${MACA_PATH:-/opt/maca}"

_prepend_path PYTHONPATH "${TILELANG_ROOT}"
_prepend_path LD_LIBRARY_PATH "${MACA_PATH}/lib"
_prepend_path LD_LIBRARY_PATH "${MACA_PATH}/mxgpu_llvm/lib"
_prepend_path PATH "${MACA_PATH}/mxgpu_llvm/bin"

echo "TileLang environment configured:"
echo "  source: ${TILELANG_ROOT}"
echo "  native: ${TILELANG_ROOT}/build/lib/libtilelang.so"
echo "  python: $(command -v python)"

unset _tilelang_candidates _candidate
unset -f _prepend_path
