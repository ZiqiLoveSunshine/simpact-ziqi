# Running the closed loop on a local VLM (no API key)

The propose → rollout → verify ↔ regress loop is provider-agnostic: every VLM
caller resolves its backend through one seam,
`simpact/generator/vlm.py::default_generate`. Setting `SIMPACT_VLM_BACKEND=openai`
points the whole loop at any OpenAI-compatible `/chat/completions` server — so a
**self-hosted** model can replace Gemini with no code change and no `GOOGLE_API_KEY`.

This is what makes the repo runnable offline, and what lets you swap in a model you
can audit. It is **not** a claim of parity: the paper's numbers are Gemini 2.5 Pro.

## 1. Pick a model

The loop asks a lot of a VLM: read a tabletop photo, reason about metric `(x, y, z)`
coordinates in a stated world frame, and emit **strict nested JSON**. Requirements:

- **vision** with reasonable resolution (scene photo + rollout renders are 640×480),
- **multi-image** in one request (the regress step sends every rollout's after-image),
- **long context** — the propose prompt alone is ~4 KB before images,
- **spatial grounding** — this is the part small models fail first.

[Qwen3-VL](https://huggingface.co/Qwen) is the recommended family: it is explicitly
trained for spatial perception (object position, viewpoint, occlusion; 2D/3D
grounding for embodied AI), ships official GGUFs, and is supported by llama.cpp's
multimodal (`mtmd`) stack.

| model | GGUF | + mmproj | VRAM | notes |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | Q8_0 · 8.7 GB | 1.16 GB | ~11 GB | fast; good for validating the plumbing |
| Qwen3-VL-8B-Instruct | Q4_K_M · 5.0 GB | 1.16 GB | ~7 GB | for 8–12 GB cards |
| Qwen3-VL-32B-Instruct | Q4_K_M · 19.8 GB | 1.20 GB | ~23 GB | best that fits a 32 GB card |

Always pair the LLM GGUF with its **`mmproj-*`** file — that is the vision encoder.
Without it `llama-server` loads as a text-only model and every image is dropped.

## 2. Build llama.cpp and fetch the weights

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120        # 120 = RTX 50xx (Blackwell); 89 = 40xx, 86 = 30xx
cmake --build ~/llama.cpp/build -j --target llama-server
```

```bash
mkdir -p ~/models/qwen3vl && cd ~/models/qwen3vl
R=https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main
curl -L --fail -C - -O "$R/Qwen3VL-8B-Instruct-Q8_0.gguf"
curl -L --fail -C - -O "$R/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
```

## 3. Serve

```bash
bash scripts/serve_local_vlm.sh          # defaults to the 8B above on port 8080
```

or directly:

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3vl/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --mmproj ~/models/qwen3vl/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  --host 127.0.0.1 --port 8080 --alias qwen3vl -ngl 99 -c 32768
```

## 4. Point simpact at it

```bash
export SIMPACT_VLM_BACKEND=openai
export SIMPACT_VLM_BASE_URL=http://127.0.0.1:8080/v1
export SIMPACT_VLM_MODEL=qwen3vl

MUJOCO_GL=egl uv run python scripts/optimize.py --task push \
  --scene examples/push_real2sim/0103_push_0 --out_dir /tmp/push_local
```

| env var | default | effect |
|---|---|---|
| `SIMPACT_VLM_BACKEND` | `gemini` | `openai` selects the local path |
| `SIMPACT_VLM_BASE_URL` | `http://127.0.0.1:8080/v1` | server root (must include `/v1`) |
| `SIMPACT_VLM_MODEL` | `local-vlm` | model name / `--alias` |
| `SIMPACT_VLM_API_KEY` | `no-key` | sent as `Authorization: Bearer …` |
| `SIMPACT_VLM_TEMPERATURE` | `0.7` | lower ⇒ more reliable JSON, less proposal diversity |
| `SIMPACT_VLM_MAX_TOKENS` | `8192` | propose emits 3 plans with reasoning — don't cut this low |
| `SIMPACT_VLM_JSON_MODE` | `1` | send `response_format={"type":"json_object"}` |
| `SIMPACT_VLM_SCHEMA_MODE` | `1` | compile the task's primitive whitelist into the grammar |
| `SIMPACT_VLM_IMAGE_FORMAT` | `PNG` | `JPEG` to cut request size on slow links |

**Leave both grammar switches on.** `SIMPACT_VLM_JSON_MODE=1` removes the markdown
fences and prose preambles that otherwise break parsing. `SIMPACT_VLM_SCHEMA_MODE=1`
goes further: when the task declares a primitive whitelist (`TaskSpec.allowed_prims`),
it is compiled into the decoding grammar, so a forbidden primitive cannot be sampled
at all. This is not cosmetic — measured on Qwen3-VL-8B against the push task:

| | propose calls yielding a valid plan |
|---|---|
| JSON mode only | **0 / 12** — every sample used `GRASP`/`RELEASE`, which push forbids |
| + schema mode | **6 / 6** |

The prompt already says "ONLY use PUSH, LIFT, DESCEND"; small models simply do not
honour it. The constraint is identical to the one `ProposalSet.validate` applies
afterwards, so this only moves the check earlier — it never widens what is accepted,
and the Gemini path is untouched.

Schema mode also **bounds the arrays** (≤3 proposals, ≤10 actions). Without a bound
the model emits proposals until it hits `max_tokens` and the JSON is truncated
mid-object; the cap just matches what the prompt asks for.

## Gotchas

- **Context.** The regress step's request grows with the number of rollouts (text +
  one image each). `-c 32768` is a safe floor; raise it if you raise `--max_iters`.
- **`--alias` must match `SIMPACT_VLM_MODEL`,** or the server 404s on the model name.
- **Quantization hurts coordinate arithmetic** more than it hurts captioning. If plans
  parse but the numbers look random, go up in quant before going up in parameters.
- **The measured gates do not care which VLM you use.** `alignment_gate` and
  `coverage_gate` read the rollout's object positions and particle cloud, so a
  local-model run is scored exactly like a Gemini run — which is what makes the
  comparison meaningful.
