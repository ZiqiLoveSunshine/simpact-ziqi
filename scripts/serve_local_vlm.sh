#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve a local vision-language model for the simpact closed loop.
#
# Starts llama.cpp's `llama-server` with an OpenAI-compatible /v1 endpoint plus a
# multimodal projector, which is all `SIMPACT_VLM_BACKEND=openai` needs — the
# propose/verify/regress loop then runs with NO API key (see docs/LOCAL_VLM.md).
#
# Usage:
#   scripts/serve_local_vlm.sh [--model PATH] [--mmproj PATH] [--port N] [--ctx N]
#
#   LLAMA_SERVER=/path/to/llama-server   (default: ~/llama.cpp/build/bin/llama-server)
#   MODEL_DIR=/path/to/ggufs             (default: ~/models/qwen3vl)
#
# Then, in another shell:
#   export SIMPACT_VLM_BACKEND=openai
#   MUJOCO_GL=egl uv run python scripts/optimize.py --task push --out_dir /tmp/push_loop
# ---------------------------------------------------------------------------
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3vl}"
MODEL="$MODEL_DIR/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
PORT=8080
CTX=32768
ALIAS=qwen3vl
# Qwen-VL needs >=1024 image tokens to ground reliably (llama.cpp warns at load);
# simpact's propose/verify steps ARE grounding tasks — object positions in a frame.
IMG_MIN_TOKENS=1024

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="$2";  shift 2 ;;
    --mmproj) MMPROJ="$2"; shift 2 ;;
    --port)   PORT="$2";   shift 2 ;;
    --ctx)    CTX="$2";    shift 2 ;;
    --alias)  ALIAS="$2";  shift 2 ;;
    --image-min-tokens) IMG_MIN_TOKENS="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

for f in "$LLAMA_SERVER" "$MODEL" "$MMPROJ"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

echo "serving $(basename "$MODEL") on http://127.0.0.1:$PORT/v1  (ctx=$CTX)"
exec "$LLAMA_SERVER" \
  -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port "$PORT" \
  --alias "$ALIAS" \
  -ngl 99 -c "$CTX" \
  --image-min-tokens "$IMG_MIN_TOKENS" \
  --temp 0.7 --top-p 0.8 --top-k 20
