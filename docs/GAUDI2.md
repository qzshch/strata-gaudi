# Gaudi2 port

This fork preserves Project Maya's history and MIT license. The starting revision is
`mw00/project-maya@8412d5edc36e714ea4001a56c5bb65a7b2228d41`. The first target is
GLM-5.3-Flash, pure text, C1, ordinary autoregressive generation, MTP disabled, on one
96 GiB Gaudi2 and a 64 GiB Host. No complete HPU model execution exists at this stage.

## Packed-expert planning

The native `strata-hpu-plan` command reads actual GGUF directories, including metadata-only
shards and experts whose gate/up/down tensors are in different shards. It records exact
compressed matrix offsets and row byte sizes; it does not decode or duplicate all weights.
It rejects incomplete shard sets, duplicate tensors, missing expert roles, mismatched shapes,
unsupported byte geometries, overflow and payloads outside the source files. Main layers and
optional NextN tensors are separated using the artifact's trunk/NextN metadata.

Each routed expert has exactly one resident owner: compressed device bank, compressed Host
bank, or file fallback. Device experts never consume a second Host-bank slot. Allocation
alignment is charged. The file is still the persistent source of every expert; logical file
ownership does not mean its pages consume no RAM. The separate file-cache allowance is
deducted from Host capacity, and the OS can use more cache transiently or under other workloads.

For the initial plan:

```
device expert capacity = device budget - device reserve - packed/BF16 transfer slots
Host expert capacity = Host budget - Host reserve - file-cache allowance - packed transfer slots
```

The default budgets are 96 GiB device, 64 GiB Host, 27 GiB device reserve, 16 GiB Host reserve,
8 GiB file-cache allowance, two slots and 64 rows per tile. The reserve values are explicit
planning assumptions. They must be replaced with measured free memory and real dense/state/
recipe/workspace requirements before execution. `nonexpert_packed_bytes` reports artifact
representation size, which is not the execution allocation size. The planner does not prove
that those weights fit the reserve, allocate HBM, constrain the OS page cache, or predict speed.

Each slot accounts for the largest compressed row tile and the largest corresponding BF16
tile, bounded independently of the number of experts in a layer. A whole GLM expert layer
expanded to BF16 would be 13.5 GiB; no such full-layer buffer is part of this plan. Byte geometry
support does not advertise a working TPC decoder or numerical parity for that type.

Build without a GPU SDK or downloaded ggml:

```sh
cmake -S . -B build-hpu-host \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_HPU_TOOLS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=ON
cmake --build build-hpu-host --config Release --target strata-hpu-plan hpu_expert_plan_test hpu_tile_stream_test hpu_packed_decode_test hpu_glm_routing_test
ctest --test-dir build-hpu-host -C Release -R '^hpu_' --output-on-failure
```

On Windows with the Visual Studio generator, executables are under
`build-hpu-host/src/hpu/Release/`; on a single-configuration Unix generator they are under
`build-hpu-host/src/hpu/`.

```sh
build-hpu-host/src/hpu/strata-hpu-plan \
  --host-gib 64 --hbm-gib 96 --host-reserve-gib 16 \
  --hbm-reserve-gib 27 --page-cache-gib 8 --tile-rows 64 --slots 2 \
  --placements build-hpu-host/placements.csv model-00001-of-00003.gguf \
  model-00002-of-00003.gguf model-00003-of-00003.gguf
```

The summary is JSON, the optional placement table is CSV. Without a measured hot-expert
profile the command uses deterministic layer/expert order, which is reproducible but is not a
performance policy. The library accepts an explicit priority list. There is no HPU model CLI
flag that silently dispatches to CUDA, HIP, Python or a fake runtime.

## Bounded file tiles and completion lifetime

`read_packed_rows` seeks to a matrix's exact shard/expert offset and reads only complete rows
into caller storage. `PackedTileStream` owns a fixed number of fixed-size pageable Host buffers.
It publishes a generation-tagged ticket only after the complete file read succeeds. A busy slot
applies backpressure, and an old ticket cannot finish or discard a newer use of the same slot.
Read failure, missing files, invalid row ranges and undersized slots cannot publish partial data.

The consumer must submit its ticket with a real completion-wait callback. `finish`, explicit
`close` and destruction wait before recycling or freeing any submitted buffer. Discarding a
submitted tile is refused: request cancellation alone does not establish DMA completion. A
failed explicit wait leaves the slot in flight for retry. Completion callbacks must not throw
during shutdown. These callbacks are the intended connection to a SynapseAI event wait; this
implementation does not claim native device events, pinned memory, or hardware overlap.

File reads are serialized inside the staging stream and can overlap external consumer work
in the other slot. Ready-buffer views require caller lifetime coordination until submission.
The file-byte counter reports successfully requested bytes, without claiming physical NVMe
traffic. Host bank loading can transiently duplicate file-backed pages with anonymous copies;
the later loader needs incremental allocation and suitable page-reclaim/advice, followed by
actual peak-memory measurements. The planner's file-cache allowance is not an OS memory cap.

## Exact host decoding for HPU tiles

This reference requires a little-endian Host with IEEE-754 binary32; other Hosts fail at
compile time instead of interpreting the packed codebooks in the wrong byte order.

`decode_packed_bf16` now consumes row-major packed tiles of Q2_K, Q3_K, IQ2_S, IQ3_S and IQ4_XS
and produces caller-owned row-major BF16 bits using round-to-nearest, ties-to-even. There is
also a pre-rounding FP32 reference entry point. Unsupported types, partial blocks, short
buffers, size overflow and overlapping input/output are rejected before decoding. Only one
256-float block is expanded on the stack; neither a full expert nor a full FP32 tile is allocated.

The implementation reuses Maya's existing scalar Q3_K/IQ4_XS decoding and its exact vendored
ggml codebooks. IQ2_S/IQ3_S execute the existing CUDA decoder's 32 logical lane mappings on the
Host; Q2_K follows the same pinned ggml reference. The source and codebooks remain under the
existing `third_party/ggml/LICENSE`. This is a correctness reference and an input-preparation
path for the future HPU backend, not a TPC kernel or MME implementation.

Ten deterministic synthetic cases cover all five formats, multiple rows, 512/4096 columns,
random code/sign/scale bits and half-scale zero/subnormal/normal/negative/max-finite values.
Expected FP32 and BF16 results come independently from pinned gguf-py; the binary fixture
contains no model weights. Tests check 48,640 FP32 values and BF16 bit patterns, canary guards,
unaligned packed input and the real `PackedTileStream` file-row → decoded BF16 path. They also
exercise BF16 midpoint, signed-zero, infinity and NaN conversion separately. Ordinary CTest
needs no NumPy, ggml download or GPU SDK. Linux and Windows CI run the same offline checks.

To regenerate the goldens, use NumPy and a llama.cpp checkout at
`third_party/ggml/VERSION.txt`, then run:

```sh
python tools/hpu/generate_decode_vectors.py --gguf-py /path/to/llama.cpp/gguf-py
```

The generator verifies the oracle source SHA-256 and records source revision, seed, output
hash and NumPy version in `tests/hpu/fixtures/packed_decode_vectors.json`.

## Fixed GLM routing reference

`select_glm53_experts` accepts one token's 288 FP32 logits and correction biases.
It implements the published single-group, top-8 sigmoid router: correction bias affects
selection only, while original sigmoid scores are normalized by their selected sum plus
`1e-20` and multiplied by 2.5. The helper uses fixed stack storage and FP32 arithmetic.
It rejects non-finite inputs. Very negative logits can saturate to zero in FP32; an all-zero
selection therefore produces zero weights. Equal corrected scores use ascending expert ID
as a deterministic tie break; framework top-k tie choices and reduction order can differ.

The profile comes from the [official model config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)
and the [fixed Transformers implementation](https://github.com/huggingface/transformers/blob/944e22eb226890dec60299386579275a66b3b44d/src/transformers/models/glm5_next/modeling_glm5_next.py#L144).
Tests use an independent long-double sigmoid/all-pairs rank oracle with distinct selection
boundaries, plus explicit bias-only selection, tie, saturation and invalid-input cases.
This is a Host reference for later route dispatch, not a router GEMM, device kernel, shared
expert computation or CPU/HPU result merger. It is not wired into the generation runner.

## Bounded single-expert forward reference

`expert_forward_reference` now evaluates one token through a routed expert's gate/up →
clamped SwiGLU → down path. Its three matrices can use different supported GGUF formats.
It reads and decodes bounded row tiles through `PackedTileStream`, retaining only one decoded
FP32 tile, an FF-width activation vector and a hidden-width candidate output. The caller owns
the stream and scratch span and must explicitly budget their payload bytes plus the codec's
one-block stack scratch. File-stream/allocator metadata and OS file cache remain in the
application's separate Host reserve; this is not an OS memory cap. Failure at any read or
arithmetic step leaves the caller's output unchanged. Successful in-place input/output is
supported, while scratch aliasing is refused.

This Host reference's decoded FP32 tile uses four bytes per value; it must not reuse the
planner's two-byte device BF16-tile allowance. `payload_workspace_bytes` reports only the
declared CPU payload, including the actual staging-slot allocation, not process RSS or peak
OS page-cache/allocator usage. The stream must provide one sufficiently large slot; aggregate
bytes spread across undersized slots do not satisfy the requirement.

`codec_stack_bytes=1024` charges the explicit 256-float expansion. Small local packed-block
structs and ordinary call-stack overhead are still in the Host reserve; this is not an exact
measurement or cap on the process stack.

The named profile is **`packed_fp32_fp64_accum`**: exact packed → FP32 weights; left-to-right
FP64 dot sums cast to FP32; SwiGLU evaluated from those FP32 gate/up values in FP64 and cast
to FP32; then a FP64 down dot cast to FP32. This deliberately supplies a numerical reference.
Maya's production CPU lane uses GGML activation quantization and vector dots, and a future
HPU BF16 MME profile has different storage/reduction boundaries. This function is not either
production path and makes no throughput or full-model parity claim.

The GLM contract is `gate=min(gate,limit)`, `up=clamp(up,-limit,+limit)`, then `silu(gate)*up`,
with limit 10 by default. Negative gate values have no lower clamp; there is no 1.702 alpha.
The result is one **unweighted** expert output: router renormalization/scaling and shared
expert merging belong outside this function. This agrees with Maya's `glm_ffn.cu` and the
[fixed Transformers activation](https://github.com/huggingface/transformers/blob/944e22eb226890dec60299386579275a66b3b44d/src/transformers/models/glm5_next/modeling_glm5_next.py#L136).

Tests construct sparse synthetic packed matrices from independent gguf-py golden blocks,
then compare to a separate sparse FP64 matrix/activation oracle that never calls this
decoder or forward function. They cover all five formats, mixed gate/up/down types,
non-divisible row tiles, upper-only gate clipping, both-sided up clipping, workspace and
alias failures, missing files, non-finite/overflow inputs, late down-read failure and real
4096-hidden/2048-FF geometry. Each sparse matrix consumes less than 1 MiB of physical fixture
storage; no real checkpoint or large zero-filled file is created. These are expert-module
checks, not generation-runner or accelerator validation.

## GLM KDA core reference

`kda_reference` fixes the audited GLM-5.3-Flash linear-attention profile (34 layers, 64 heads
of 128, 4-tap short conv, gate lower bound −5.0) and implements the four KDA core stages as
Host FP32 references: causal per-channel conv + SiLU with an explicit `(d_conv−1) × d_inner`
state, the safe-path decay gate `lb·sigmoid(exp(A_log)·g)` in log space, the gated delta rule
scan, and the RMSNorm×weight×sigmoid(g2) output gate. The scan applies FLA l2 norm (eps 1e-6
inside the sqrt) to q/k, scales q by `head_dim^-0.5`, decays the FP32 state's key-axis rows by
`exp(gate)`, updates `S += k̂ ⊗ β(v − Sᵀk̂)` and reads `o = Sᵀq̂`. The semantics were
cross-checked against the fixed-commit FLA fused Triton kernel (which computes the gate and
l2 norms in-kernel and always exponentiates the log-space value), the transformers
`glm5_next` modules, llama.cpp's graph and Maya's CUDA kernels.

The scan takes caller scratch holding one state image plus one output image; state and output
commit together only after the whole scan validates, so an overflowing intermediate leaves
both untouched. Processing tokens one at a time is bit-identical to one multi-token call,
including the final state; the conv state has the same split continuity. A zero q vector
produces exact zero outputs. A finite but out-of-range beta overflows the state mid-scan and
is rejected with both planes intact — the guard between a caller-side sigmoid bug and silent
state corruption.

The references are the recurrent (decode) algorithm only: chunk prefill, the qkv/f_a/f_b/g_a/
g_b/beta projections (plain linears owned by the future model layer), GGUF weight binding and
any HPU execution are explicitly not implemented. Tests use an independent long-double oracle
assembled from the FLA naive formulation in a different association order, plus small-profile
continuity/failure cases and a 64×128-dimension split-continuity smoke; they are not
generation-runner, hardware or long-context validation.

## GLM KDA chunk prefill reference

`kda_chunk_reference` implements the FLA naive chunk algorithm with the audited GLM-5.3
semantics, sharing the recurrent reference's exact input contract (post-conv q/k/v, log-space
gate, already-sigmoided beta, in-function l2 norm and query scale). Tokens must be a multiple
of the chunk width (64 in production; the caller pads tails with g=0/q=k=v=beta=0 rows, which
the tests prove inert bitwise). Per chunk and head: a rebased per-chunk cumsum of the log gate;
the decayed key Gram computed from differences only — every exponent is
`gsum[later] - gsum[earlier] <= 0`, so the factored product form that overflows fp32 on long
chunks is never used; the unit-diagonal strict-lower system solved by exact forward
substitution (no Neumann approximation); the LATER token's beta scaling the Gram rows and the
EARLIER (column) token's beta scaling the u/w operands, with the (T+I) diagonal carrying the
own token's beta; Aqk keeping its diagonal; and the chunk-affine state update. The index
conventions were pinned by a numpy translation of the FLA naive source verified against the
naive recurrent scan to 1e-16 in float64 before the C++ port; the C++ was then debugged to
that Python reorganization (one transposed Gram orientation and one beta stride slip, both
caught by the per-token oracle diff).

Tests compare every chunk width (2/4/8) against an independent long-double RECURRENT oracle
(proving the reorganization, not a shared bug), against the fp32 recurrent reference within
the association band, split-call versus whole-call continuation, bitwise-inert zero padding,
the full 64×128 GLM profile at 128 tokens against the recurrent reference, and the invalid
size/NaN family. Scratch is caller-owned: one state image plus one output image (same
validate-then-commit contract as the scan) plus one reusable per-head work plane of
`chunk*(chunk + 6*head_dim + 1)` values. This is the reference algorithm only: no varlen
packing, no TPC/MME layout, and the abotsis HPU lazy-path miscompile list (stride-0 expand
into batched matmul, Hillis-Steele scan) still applies to any future device port.

## GLM mHC reference

`mhc_reference` fixes the audited mHC geometry (4 streams, 20 Sinkhorn iterations, hc_eps 1e-6,
unweighted RMS eps 1e-5 over the flattened 16384-wide stream plane) and implements the three
pieces a decoder layer needs. `mhc_pre_reference` normalizes the flattened streams, runs the
24-lane mix projection in FP32, and produces pre = sigmoid+scaled+base then +eps,
post = 2·sigmoid(...), and comb from the flat checkpoint layout `comb[dst + hc*src]` as a
softmax over dst followed by the pinned Sinkhorn order — one column normalization over src,
then 19 rounds of row-then-column, each denominator carrying hc_eps — plus the
pre-weighted stream collapse. `mhc_post_reference` applies
`out[dst] = post[dst]*block + sum_src comb[src,dst]*residual[src]` per token (staged, so an
out aliasing the residual never mixes values), and `mhc_head_reference` is the final
unweighted mean. Semantics were cross-checked against transformers, llama.cpp's graph, Maya's
reference and sglang's wiring; the abotsis parity test's documented history — an internal
mirror silently using softmax for pre shipped a TPC kernel oracle-tested to the wrong
function — is why these tests derive their oracle from the transformers module directly, keep
`pre sums != 1` as an invariant, and probe the flat comb layout with a one-hot base. The
iters=1-vs-20 comparison pins that Sinkhorn actually iterates while gates stay iteration-free.

The mHC references are pure per-token math: no projections beyond the mix linear, no GGUF
binding, no bf16 mix rounding (the HPU production envelope of ~1e-2 relative on gate outputs
is a later device-path concern), no HPU execution.

## GLM DSA indexer and absorbed-MLA reference

`dsa_reference` implements the audited DSA selection chain and the absorbed sparse
attention as Host FP32 references, matching the implementation-level research notes
(fixed commits, cross-checked transformers x llama.cpp x Maya). `dsa_pool_reference`
compresses COMPLETE pools of 4 tokens with the parity-pinned PER-KEY-ELEMENT softmax
(gates + learned ape), `dsa_score_reference` applies the two-stage ReLU scoring with
the transformers scale placement (head_dim^-0.5 inside the ReLU, heads^-0.5 on the
weights; llama/Maya pre-multiply the combined factor - same product, different
rounding), and `dsa_select_reference` ranks the visible pools (pool visible iff its
last member <= the query's global position) stable-descending with ties by lower pool
index, places each pool at its rank slots, appends the incomplete tail in position
order (the tail includes the query token itself) and pads with -1 to the fixed 2051
width. `dsa_attention_reference` consumes the selection rows in absorbed form: scores
are q_abs . latent scaled by qk_nope^-0.5 (the 256^-0.5 model constant, NOT
kv_lora^-0.5), the softmax and context accumulation exclude -1 slots entirely, and
W_V_B applies after the attention per head from the GGUF {kv_lora, v_head, heads}
layout. NoPE only; the layer's projections (q_a/q_b/kv_a/wk_b/wv_b application,
LayerNorm'd indexer keys) stay with the future model layer.

Tests cover: long-double pool/score oracles; the ReLU sign boundary; rank/tie/slot
order, every tail width, the top-pools-zero start-of-sequence row, and visibility
boundaries; the absorbed attention against an independent long-double oracle on
hand-built selections, on a full selection (the dense-equivalence case), and with
duplicate cells; an end-to-end pool->score->select->attend chain asserting the
selection budget and causality; and the invalid-input family. The GLM constants
(2051 width, 512 pools, 32x128 indexer, 512 latent / 64 heads / 256 nope) are pinned.
This is the reference selection and attention math only: streaming pool-state
maintenance (rep rows, edit invalidation - the llama hybrid-idx protocol) and HPU
execution remain future work.

## Remaining execution work

The packed planner is a preparation tool; the existing CUDA/HIP GLM runner does not use it.
Linux SynapseAI device allocation, pinned Host registration, DMA/event lifetime, recipe
compilation, bounded TPC quantized decoding and MME GEMMs still need to be wired into a real
GLM runner. Dense/shared experts and Host expert-result merging still need independent numerical
and continuation tests; the KDA recurrent/chunk references cover the decode and prefill
core math, mHC the stream mixing, and the DSA reference the selection chain and absorbed
attention (streaming pool-state maintenance is still open). In
particular, disabling the indexer is not a parity implementation of the model's sparse
attention. End-to-end quality, 256K context, peak RAM/HBM and throughput require actual
Gaudi2 validation.

The host tests are correctness checks for artifact layout, byte accounting and ownership;
they are not hardware tests or model-quality results.
