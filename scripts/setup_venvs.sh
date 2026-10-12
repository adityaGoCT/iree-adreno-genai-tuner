#!/usr/bin/env bash
# Two Python 3.12 environments (IREE's macOS wheels are cp311/cp312 only):
#   .venv          IREE 3.11.0  - compiler matching the 3.11 runtime (stock Qualcomm driver)
#   .venv-iree312  IREE 3.12.0  - compiler matching the 3.12 runtime (Mesa Turnip)
# Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON=(torch==2.14.1 transformers==5.19.0 iree-turbine==3.9.0 mako numpy)
uv venv "$ROOT/.venv" --python 3.12
uv pip install --python "$ROOT/.venv/bin/python" "${COMMON[@]}" iree-base-compiler==3.11.0 iree-base-runtime==3.11.0
uv venv "$ROOT/.venv-iree312" --python 3.12
uv pip install --python "$ROOT/.venv-iree312/bin/python" "${COMMON[@]}" iree-base-compiler==3.12.0 iree-base-runtime==3.12.0
"$ROOT/.venv/bin/iree-compile" --version | head -2
"$ROOT/.venv-iree312/bin/iree-compile" --version | head -2
