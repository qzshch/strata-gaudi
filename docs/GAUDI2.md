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
cmake --build build-hpu-host --config Release --target strata-hpu-plan hpu_expert_plan_test hpu_tile_stream_test
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

## Remaining execution work

The packed planner is a preparation tool; the existing CUDA/HIP GLM runner does not use it.
Linux SynapseAI device allocation, pinned Host registration, DMA/event lifetime, recipe
compilation, bounded TPC quantized decoding and MME GEMMs still need to be wired into a real
GLM runner. KDA FP32 state, sparse MLA/indexer, mHC, dense/shared experts and Host expert-result
merging need independent numerical and continuation tests. In particular, disabling the
indexer is not a parity implementation of the model's sparse attention. End-to-end quality,
256K context, peak RAM/HBM and throughput require actual Gaudi2 validation.

The host tests are correctness checks for artifact layout, byte accounting and ownership;
they are not hardware tests or model-quality results.
