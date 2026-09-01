# SSD expert streaming

SSD expert streaming reduces the resident memory of supported mixture-of-experts
models by keeping a bounded expert bank in memory and loading exact routed experts
on demand. The feature is experimental and disabled by default. When disabled, no
model classes are patched, no checkpoint reader is opened, and normal model loading
and evaluation are unchanged.

## Residency modes

- `soft_reap` pins experts listed in a Soft-REAP manifest and uses the cache for
  remaining routes.
- `cache_only` starts with no pinned routed experts and learns the working set at
  runtime.

Both modes execute exact router choices. They do not substitute, predict, or
preselect experts. Shared experts and the non-expert model trunk remain resident.
For Qwen4-Exp, enabling expert streaming forces the large PLE tensors onto their
existing mmap/SSD path instead of materializing them as resident arrays.

The implementation discovers layouts from safetensor metadata. It supports stacked
and per-expert tensors, fused or separate gate/up projections, and quantized affine
weights with scales and biases. Execution is selected from shape and dtype rather
than a hard-coded quant name.

## Experimental settings

| Setting | Default | Purpose |
| --- | ---: | --- |
| `expert_streaming_enabled` | `false` | Opt in to SSD expert streaming. |
| `expert_streaming_mode` | `soft_reap` | Manifest pinning or cache-only residency. |
| `expert_streaming_manifest` | unset | Soft-REAP JSON manifest; required by `soft_reap`. |
| `expert_streaming_cache_experts` | `32` | Dynamic resident expert slots per layer. |
| `expert_streaming_scratch_experts` | `32` | Cold execution slots per layer for prompt batches. |
| `expert_streaming_cache_policy` | `route_frequency` | `route_frequency` or `lru` eviction. |
| `expert_streaming_fast_resource_loading` | `true` | Use the Metal I/O loader. |
| `expert_streaming_direct_io` | `true` | Permit direct loads into idle Metal buffers. |
| `expert_streaming_native_demand` | `true` | Resolve exact decode demand at native evaluation. |
| `expert_streaming_decode_scratch_as_cache` | `true` | Lend scratch slots to the decode cache. |
| `expert_streaming_io_coalescing_kib` | `64` | Maximum gap merged into one storage read. |

The Admin dashboard and macOS model editor expose these settings under
Experimental. Profiles include all of them. Native demand and direct I/O require
Fast Resource Loading.

## I/O and weight format

Fast Resource Loading can issue staged Metal I/O reads for any indexed layout.
Direct I/O is used only when the destination is known idle and the on-disk tensor
representation exactly matches the runtime bank. Otherwise the loader safely
stages the read and copies or converts it.

For the DeepSeek V4 Flash mixed 2.4-bit checkpoint, keep packed quantized weights
packed and store routed-expert affine metadata (`scales` and `biases`) as FP16.
This matches the FP16 expert bank and permits the fastest direct path. BF16 affine
metadata remains supported through staged BF16-to-FP16 conversion, but cannot use
that direct path. The conversion utility is:

```bash
python scripts/convert_deepseek_v4_streaming_layout.py /path/to/model
python scripts/convert_deepseek_v4_streaming_layout.py /path/to/model --apply
```

The first command is a dry run. Use a checkpoint copy unless in-place conversion
is intentional. The utility validates tensor metadata and writes a conversion
report.

## Runtime behavior

Prompt processing groups routes into a preallocated execution bank so cold experts
do not evict the hot cache for every prompt group. During token generation, native
demand releases the Python GIL while the Metal graph is evaluated, loads exact
missing rows, and publishes them before dispatch. At the prefill/decode boundary,
the scratch bank can be deterministically reclassified as extra cache and restored
before the next prefill.

Runtime statistics include cache hits and misses, just-in-time loads, SSD bytes and
operations, native-demand callbacks, direct-load counts, I/O timing, and elastic
cache transitions. Admin benchmark snapshots include these counters.

## Benchmarking

`benchmarks/soft_reap_streaming.py` reports projected residency, validates direct
row reads against MLX checkpoint loading, and can run an end-to-end generation:

```bash
python benchmarks/soft_reap_streaming.py \
  --model /path/to/model \
  --streaming-mode cache_only \
  --cache-experts 74 \
  --scratch-experts 12 \
  --full-load
```

Cache and scratch sizes are deployment budgets, not model constants. Tune them for
expert-row size, available memory, prompt/decode mix, and storage latency. Compare
warmed repeated runs and record just-in-time loads alongside PP and TG throughput;
a smaller resident budget is not automatically faster.
