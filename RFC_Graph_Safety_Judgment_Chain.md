# [RFC]: A Graph-Safety Judgment Chain for vllm-gr CUDA Graph Development

> state: RFC draft ｜ author: @DINGEde ｜ date: 2026-08-24 ｜ issue: #295

## Abstract

This RFC records a complete study of CUDA Graph execution semantics — twelve course modules, from Stream/Event FIFO up to the physical geometry of beam search — whose **ultimate goal is to improve graph formation for the beam-decode stage in vllm-gr**. Its deliverable is a reusable **graph-safety judgment chain**: a four-layer conceptual framework plus six safety questions, refined into a **Graph-Safety Harness** (four layers of invariants and a ten-step falsification loop), and finally applied to beam search to turn "is this code graph-safe?" into "what specific constraints must beam decode satisfy before it can be captured?"

This revision additionally reviews **PR #294 ("FULL CUDA Graph for beam decode via startup dummy capture")**, which has since landed the first working beam-decode FULL graph. The review maps PR #294 back onto the judgment chain — validating the framework's predictions — and then identifies the concrete boundaries that remain, producing a prioritized set of next-step directions for the beam-decode graph goal.

## 1. Background and Motivation

vllm-gr relies heavily on CUDA Graph for decode performance (see #225 Beam-decode FULL CUDA Graph, #278 Prefill Full Graph Capture, #286 Full ACLGraph Offload). The **ultimate goal of this study is to improve graph formation for the beam-decode stage** — a problem first framed by #225: beam search keeps multiple candidate beams per step whose physical layout (geometry) shifts as beams fork and collapse, which breaks the fixed topology and fixed addresses that a FULL graph requires.

Before approaching that problem, I needed a precise mental model of *why* CUDA Graph works and *when* code is graph-safe. This study builds that model from first principles, following the chain:

```
Stream FIFO → cross-stream happens-before → multi-stream DAG capture
            → Dispatcher → BatchDescriptor → CUDAGraphWrapper → replay
```

While this study was in progress, the team merged two directly relevant changes: **PR #292** (Beam Batch Contracts, the `#288` contract slice) and **PR #294** (the first beam-decode FULL graph). This RFC therefore ends not with an abstract "next study the beam problem", but with a grounded review of what #294 built, what it deliberately left out, and where the beam-decode graph effort goes next.

## 2. Goals and Non-Goals

### 2.1 Goals

- Document the CUDA Graph execution model (Stream / Event / DAG / three phases).
- Establish a reusable graph-safety judgment chain — **four layers + six questions + a Harness** — as the analytical tool for the ultimate goal.
- Map learned concepts onto real vllm-gr source code, both upstream vLLM 0.22.1 and the vllm-gr beam patch layer.
- Apply the chain to beam search, deriving a **beam-specific constraint list** from the five blockers in #225.
- Review PR #294 against the chain, and state the **next-step improvement directions** for the beam-decode graph goal.

### 2.2 Non-Goals

- This RFC does **not** propose a specific implementation change in this revision.
- It does **not** yet contain a detailed per-subtask work breakdown for the next phase; that is a follow-on design RFC.

## 3. Core Concepts

### 3.1 Execution model: Stream, Event, and the DAG that is not bound to a stream

- **Stream** is an *ordered submission sequence* (FIFO), not a "parallelism switch". Launching onto multiple streams enables parallelism but guarantees nothing by itself; ordering is the hardware's only promise.
- **Event** establishes cross-stream *happens-before*: recording an event on stream B and waiting on it from stream A guarantees A observes all of B's side effects.

**Insight — a graph is a DAG, not a stream-bound object.** During capture on stream `Sc`, the FIFO order of `Sc` is *translated* into directed edges of the DAG. After capture, the graph treats two orderings separately: **internal order** (fixed by DAG edges) and **external order** (determined by whichever stream you replay onto). This is why replaying onto a different stream `Se` keeps the same operations and the same addresses, while only the stream changes — and why "captured on `Sc`" never implies "must replay on `Sc`".

**Three phases** map to three distinct objects: `cudaGraph_t` (the node/dependency template) → `cudaGraphExec_t` (the instantiated executable snapshot) → launch × N.

### 3.2 Reuse contract, and the memory layers behind "fixed address"

The graph fixes **topology, pointers, shape / stride / dtype / layout**; it does **not** fix the *values* stored at those addresses.

**Insight — address fixed ≠ value fixed.** KV cache *contents* change every round while the graph replays fine, because the graph reads fixed *addresses*, not fixed values. This is the single most counter-intuitive point, and it is the whole basis of the reuse contract.

**Going one level deeper: what actually guarantees "fixed address"?** It is not one mechanism but four, layered bottom-up:

| Layer | Object | Lifetime governed by |
|---|---|---|
| native allocation | `cudaMalloc` / `cudaFree` | host API + sync |
| stream-ordered allocator | `cudaMallocAsync` / `cudaFreeAsync` | stream order + event + memory pool |
| graph memory node | alloc/free nodes inside the graph | graph edges |
| PyTorch graph-private pool | caching allocator's private pool | CUDAGraph + pool token + captured tensor refs |

Two subtleties worth stating explicitly:

1. **"Fixed address" is a *virtual* address.** Physical pages may be reused or remapped underneath; correctness depends only on the virtual address staying valid within its lifetime and on DAG-protected access order — never on physical pages staying put.
2. **PyTorch's private pool (layer 4) exists to *avoid* triggering layer 3.** By pre-allocating a pool and *reusing* memory during capture, PyTorch prevents any real allocation from happening mid-capture, so no graph memory node is ever needed. Pre-allocating workspace is therefore an **engineering choice for determinism**, not a workaround for "CUDA Graph cannot allocate".

### 3.3 Four-layer framework, and the confusions it prevents

Analyzing any graph-related code requires first asking *which layer is responsible*:

| Layer | Objects | Answers the question |
|---|---|---|
| CUDA execution | Stream, Event, Graph, GraphExec | what physically happens on the GPU |
| PyTorch | static tensor, graph-private pool, RNG state | what the framework adds on top |
| vLLM strategy | CUDAGraphMode, Dispatcher, BatchDescriptor | when / which graph to use |
| model runtime | attention metadata, KV, LoRA, offload | what the forward actually reads/writes |

Three confusions this framework prevents (all three are mistakes I initially risked making):

1. **"CUDA Graph is a vLLM scheduling feature."** ❌ It is a CUDA execution-layer optimization; vLLM merely *uses* it. `CUDAGraphMode` is a vLLM software enum, not a CUDA concept.
2. **"Capture succeeded, so the code is safe."** ❌ Capture only proves a valid graph was formed (execution layer); whether replay is *correct* depends on the PyTorch layer (e.g. the caching allocator returning a different address on replay).
3. **"KV contents change, so the graph cannot be reused."** ❌ This confuses *mutable value* (layer 4) with *fixed address* (layer 1).

### 3.4 Six safety questions

The judgment chain's acceptance criteria — asked *in order*, before asking "can it capture":

1. Is the execution topology fixed this round?
2. Do params / addresses / shape / stride / dtype / layout satisfy the reuse contract?
3. Is ownership of metadata / workspace / KV / side-effects / RNG stable?
4. Do multi-streams have complete fork/join?
5. Are collectives in the same order across all ranks?
6. Does a signature mismatch fail closed (fall back to another graph or eager)?

### 3.5 Async copy traps: "Async" is not a performance conclusion

`cudaMemcpyAsync`'s "Async" guarantees only that the call does not block the CPU. It does **not** guarantee overlap. Two concrete consequences:

- **pinned vs pageable.** H2D with a pageable source forces the runtime to *stage* through an internal pinned buffer first — a synchronous, blocking step that silently kills the expected overlap. Only pinned host memory allows direct DMA.
- **lifetime red line.** Even with pinned memory, the source of H2D must not be modified/freed before the copy truly completes, and the destination of D2H must not be read before completion. "Async" never enforces this; the programmer must.

### 3.6 Fork/join and warmup: two practical capture pitfalls

**Fork/join (multi-stream capture).** Only the origin stream begins/ends capture; auxiliary streams are pulled in by waiting on a captured event (fork) and must be joined back before `EndCapture` by a second event (join). A branch that was forked but not joined — even if its result is only consumed by the *next* round — makes `EndCapture` fail with `cudaErrorStreamCaptureUnjoined`. The join is a graph dependency edge, not a CPU sync, not a stream destroy, and it does not serialize the two branches.

**Warmup.** Warmup's real job is to make *capture succeed*, not to make results correct: cuBLAS autotune — which allocates workspace, synchronizes, and trial-launches kernels — is illegal inside capture, so capturing a matmul cold fails with `cudaErrorStreamCaptureInvalidated`, while capturing it warm succeeds. Warmup drains lazy init / autotune *before* capture so capture records a clean, repeatable execution. It should run on a side stream with entry/exit stream handoff, covering: CUDA context / library handles / communicator init, kernel/module lazy load, Triton/cuBLAS/cuDNN autotune and plan selection, allocator paths and workspace requests, and custom-op first-registration / cache fill.

### 3.7 The PyTorch minimal loop, and why "capture round" is not "correct round"

`torch.cuda.CUDAGraph` never turns a dynamic graph static automatically; the *caller* must keep long-lived static input/output objects and update their **contents** in place before each replay. The minimal loop is:

```
side-stream warmup (drain lazy init / autotune)
→ capture with static inputs AND static outputs
→ replay: static_input.copy_(runtime_input); graph.replay(); result = static_output.clone()
```

Three consequences that generalize to vLLM:

1. **Tensor object / storage / metadata must be separated.** Rebinding a Python variable to a new tensor does *not* change the address baked into the graph's kernel args. One must check the Python object, underlying storage + `data_ptr()`, shape/stride/dtype/layout, view/alias relations, whether output is carried by static storage, metadata tensor address *and* contents, and RNG generator/state. Same shape does **not** imply reusable — a stride or layout change re-interprets the same addresses.
2. **The capture round's result is not trustworthy.** Capture *records* the graph; it does not *produce correct output*. Only after replay — with contents staged via `copy_` — does the output equal the eager path. This is the "capture succeeded ≠ graph-safe" insight (3.3) restated at the tensor level.
3. **Static output is a fixed, overwritten memory block.** `old = static_out` is aliasing, not copying; the next `replay()` overwrites the very block `old` points at. Preserving a prior round's result requires `clone()`, which copies into independent memory.

### 3.8 The Graph-Safety Harness: turning "capturable" into falsifiable assertions

Capture success proves "no immediate error this once", not "safe to replay". The reusable contract to verify is: *under an identical execution signature, addresses / shapes / layouts / topology stay compatible; every change enters only through static buffer contents; side-effects, output lifetimes, and cross-stream dependencies remain correct across repeated replays.* Four layers of invariants:

| Invariant layer | What must stay stable or controlled |
|---|---|
| Tensor contract | `data_ptr`, shape, stride, dtype, device, layout |
| Execution contract | kernel count/order, grid, algorithm plan, workspace capacity, Stream/Event DAG |
| Side-effect contract | KV write location, RNG advancement, atomic updates, collective order |
| Lifetime contract | static inputs/outputs/workspace/graph-pool alive through the last replay |

The ten-step loop: (1) eager reference → (2) multi-round side-stream warmup → (3) pre-allocate static inputs/outputs/metadata/state/workspace → (4) capture once → (5) replay same-signature different-content → (6) sentinel-check addresses/shapes before and after → (7) compare output + side-effect state vs eager → (8) record allocator delta across replay → (9) incompatible signature must fail closed → (10) lifetime negative test after owner destruction.

The three easiest negative experiments to miss: **stale data without an error** (drop the `copy_`; replay silently reads old addresses — pointer-stability is necessary, not sufficient), **static-output aliasing** (the `=` vs `clone()` trap above), and **swallowed signature miss** (shape `(16,)→(17,)` is a *topology* change — grid/launch args differ — so the correct policy is a new key / re-capture / eager, never "stuff it into the old graph").

### 3.9 The vLLM strategy layer: modes, BatchDescriptor, and the routing table

The course's concrete mapping onto upstream vLLM 0.22.1:

- **Five config modes, three runtime modes.** The config enum has `NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE`; the last two are *composite policies* (`FULL_DECODE_ONLY = (FULL, NONE)`, `FULL_AND_PIECEWISE = (FULL, PIECEWISE)`) that route batches at initialization and never appear as replay identities. The wrapper only ever sees `NONE / PIECEWISE / FULL`.
- **There is no `GraphKey` class.** The effective identity is `(runtime_mode, BatchDescriptor)`; the wrapper indexes `dict[BatchDescriptor, CUDAGraphEntry]`.
- **BatchDescriptor has five fields**: `num_tokens` (padded physical capacity), `num_reqs` (exact for FULL, relaxed to `None` for PIECEWISE), `uniform` (same query length across requests), `has_lora`, `num_active_loras`.
- **FULL keys are stricter than PIECEWISE.** FULL keeps attention *in-graph*, so its topology depends on `num_reqs`/`uniform`. PIECEWISE splits the attention op (and KV write) *out* of the graph (`_attention_ops` + `unified_kv_cache_update` as splitting ops), so initializing its keys runs `replace(batch_desc, num_reqs=None, uniform=False)` — relaxing exactly those two fields.
- **Routing order FULL → PIECEWISE → NONE** is a lookup priority, not a guaranteed three-layer fallback; a config with no PIECEWISE keys drops straight to `NONE`. Capture order is the reverse (PIECEWISE → FULL, large → small) to establish a big graph pool early.
- **`AttentionCGSupport`** is a four-level capability (`NEVER < UNIFORM_SINGLE_TOKEN_DECODE < UNIFORM_BATCH < ALWAYS`) taken as the *minimum* across backends; `NEVER` blocks FULL but may still allow PIECEWISE.
- **Capture vs replay are two chains.** `capture_model → get_capture_descs → _capture_cudagraphs → _warmup_and_capture → _dummy_run → set_forward_context(FULL) → wrapper.__call__(capture)`; runtime is `execute_model → _determine_batch_execution_and_padding → dispatch → set_forward_context → wrapper.__call__(replay)`. FULL covers only the model backbone forward — hidden-state selection, LM head logits, and `sample_tokens` stay **outside** the graph.
- **The dispatcher is the safety decision-maker; the wrapper blindly trusts ForwardContext.** Safe fallback must therefore happen at dispatcher selection time, not inside the wrapper. A nested `FULL_AND_PIECEWISE` wrapper passes through on mode mismatch — that pass-through is what lets two wrappers coexist with only one active per round.

### 3.10 Beam-specific constraints: physical geometry and persistent suffix KV

Beam search's fork/collapse/reorder is the load that most directly attacks the reuse contract. A beam's KV sequence splits into a **shared prefix** (all beams, one copy) and a **per-beam suffix** (each beam's decode segment). Its *physical geometry* is how those W suffixes are laid out in KV cache and how that layout changes each step.

Applying the six questions to beam decode:

| Question | Beam-decode status | Verdict |
|---|---|---|
| 1. topology fixed? | fork/collapse changes parent-child map and token count every step | ❌ varies |
| 2. reuse contract (addr/shape)? | suffix-KV gather `torch.empty` every step; reorder changes layout | ❌ varies |
| 3. ownership stable? | decode-KV reorder (in-place gather along beam axis) has delicate timing | ⚠️ risky |
| 4. fork/join complete? | cascade = 3 attention paths + LSE merge | ⚠️ must close |
| 5. collective order? | usually single-GPU; LSE merge is in-graph | ✅ fine |
| 6. fail-closed? | no BatchDescriptor matches → cannot route | ⚠️ missing |

The two fatal questions are **1 (topology)** and **2 (address)**. The root of the suffix problem is a scheduling-timing bug (#225): upstream `_update_after_schedule` advances `num_computed_tokens` before vllm-gr's `patched_schedule` builds beam metadata, so `cache_len == prefix_len` entering `_compute_beam_bounds` forces `b_past_suffix = 0` — always the "no historical suffix" branch, forcing a full recompute of the ever-growing decode segment each step (O(W·K²)). **Persistent suffix KV is thus a *prerequisite* for the beam graph, not an optional optimization**: if the suffix is recomputed and re-gathered every step, its address and layout never stabilize, and the graph cannot be captured at all.

#225's five blockers are exactly the six questions instantiated on beam:

| Blocker | Content | Question |
|---|---|---|
| 1. uniform-decode gating not satisfied | `_is_uniform_decode` rejects multi-token-per-beam | Q1 |
| 2. cascade forced to PIECEWISE | `dispatch_cudagraph(disable_full=use_cascade_attn)` | Q1/Q4 |
| 3. warmup never writes `BEAM_DATA_VAR` | `_dummy_run` builds standard shapes only | Q2 |
| 4. suffix-KV gather reallocates | `extract_suffix_kv` `torch.empty` per step | Q2 |
| 5. BatchDescriptor cannot match | per-request token count varies with W / decode_steps | Q6 |
| (6) decision on host | full `logits[W, vocab]` D2H for the beam decision | decision placement |

## 4. Mapping to vllm-gr Source

### 4.1 Upstream vLLM 0.22.1 (concept → code)

| Concept | Location |
|---|---|
| `CUDAGraphMode` (NONE/PIECEWISE/FULL + composites) | `config/compilation.py:53` |
| `BatchDescriptor` (5 fields) | `forward_context.py:30` |
| `CudagraphDispatcher.dispatch()` (FULL→PIECEWISE→NONE) | `v1/cudagraph_dispatcher.py:235` |
| `CUDAGraphWrapper.__call__` (capture/replay branch) | `compilation/cuda_graph.py:233` |
| `CUDAGraphEntry` (one graph/output per key) | `compilation/cuda_graph.py:128` |
| `AttentionCGSupport` (4-level capability) | `v1/attention/backend.py:606` |
| `CommonAttentionMetadata` (metadata example) | `v1/attention/backend.py:411` |
| `_attention_ops` / `splitting_ops` (PIECEWISE split) | `config/compilation.py:764` |
| `capture_model` / `execute_model` (two chains) | `v1/worker/gpu_model_runner.py:6820 / 4174` |

**Key insight:** the Dispatcher is the safety decision-maker; the wrapper blindly trusts ForwardContext. Safe fallback (to eager) must therefore happen at the Dispatcher selection stage, not inside the wrapper.

### 4.2 vllm-gr beam patch layer (PR #294 landed)

| Concept | Location |
|---|---|
| `_compute_beam_bounds` (beam geometry) | `vllm_gr/v1/beam/geometry.py:7` |
| `BeamAttentionContext` (KV pools, session registry, suffix buffers, **decode_step_buf**) | `vllm_gr/v1/beam/beam_attention_context.py` |
| `BeamAttentionPool` (5-D contiguous decode KV pool) | `vllm_gr/v1/beam/beam_attention_pool.py` |
| `cascade_attention` (prefix + suffix + LSE merge) | `vllm_gr/v1/attention/backends/beam_attn_gpu.py:87` |
| `cache_unshared_kv_kernel` / `copy_suffix_kv_kernel` (device-buffer-driven Triton) | `vllm_gr/v1/attention/backends/beam_graph_kernels.py` |
| `capture_beam_decode_graph` (startup dummy capture, re-entry-safe) | `vllm_gr/v1/worker/gpu_model_runner_patch.py` |
| `make_patched_determine_batch_execution_and_padding_gpu` (FULL gating) | `vllm_gr/v1/worker/gpu_model_runner_patch.py` |
| `_beam_full_graph_ready` / `_beam_graph_enabled_for_run` | `vllm_gr/v1/worker/gpu_model_runner_patch.py:128` |

## 5. Applying the Chain to Beam-decode: the Constraint List

Synthesizing §3.10, the beam-decode graph must satisfy, before capture is even attempted:

1. **Fix the topology.** Pin the beam *physical layout* to a fixed upper bound — fixed `max_seqlen_k` (full block-table capacity) and a full-width block-table view, so the captured graph's shape is prompt-length independent. Fork/collapse must be expressed as *content changes* (which beam occupies which fixed row), never as shape/address changes.
2. **Fix the addresses.** The decode-segment KV must live in a contiguous pool whose slices preserve the base address; no per-step `torch.empty`, no per-step Python rebuild of metadata. `BEAM_DATA_VAR` must be actively written at capture so the graph contains real beam geometry, not an empty branch.
3. **Move per-step scalars to device buffers.** Any quantity that varies across replays (e.g. `decode_step`) must be read from a device-side buffer by the kernel, not baked in as a host Python int — otherwise a graph captured at step=1 forever copies "step+1" rows.
4. **Stabilize the side-effect.** `select_unshared_kv` (in-place gather-reorder along the beam axis) must either run graph-externally or write to fixed addresses; the aux buffers that replace per-step `.clone()` must be sized from *configured upper bounds*, never from the first request's beam width.
5. **Fail closed.** Grant FULL only for an *exact* signature match (pure beam batch, exact width); a mismatch — mixed batch, unavailable session, width not in the captured set, config disabled — must downgrade to eager, never replay a wrong graph.
6. **Move the decision off the host** (the sixth, qualitatively different blocker): the `constrained_top_k + beam_select + select_unshared_kv` decision chain must become device-side eager so `logits[W, vocab]` never crosses D2H; only W `(token_id, parent)` pairs return to the host.

## 6. PR #294 Review: What Was Built, and What Remains

### 6.1 What PR #294 already did (and how it maps to the chain)

#294 took the **native vLLM startup-capture route** rather than #225's `BeamGraphRunner` subsystem: it captures the whole beam-decode forward (embedding → layers → lm_head) once at engine start with dummy inputs, then is **replay-only** for real requests. Mapping it to the judgment chain:

- **Topology (Q1)** — fixed via G1 constant-prefix geometry: a fixed `max_seqlen_k` upper bound and full-width block-table view make the captured graph's shape prompt-length independent. This is the "fixed upper bound" move from §5.1.
- **Address (Q2)** — `BeamAttentionContext` owns a 5-D contiguous decode KV pool plus fixed-address `suffix_k_buf`/`suffix_v_buf` and pre-allocated `aux_key_buf`/`aux_value_buf` (replacing per-step `.clone()`). `build_for_cudagraph_capture()` *reuses* `build()`, so eager and captured metadata paths are identical.
- **Per-step scalar (the key innovation)** — `decode_step` moved from a host Python int to `decode_step_buf` (a device-side int32). The new Triton kernels `cache_unshared_kv_kernel` and `copy_suffix_kv_kernel` read it from the buffer, so the same captured kernel works for every step. This is a textbook application of "change only through static buffer contents" (§3.8): the graph fixes the *address* of `decode_step_buf`, not its *value*.
- **Side-effect (Q3)** — `select_unshared_kv`'s aux buffers are sized from configured upper bounds, fixing the "wider later request wrote past the pool" illegal-access bug from the first-request-sizing mistake.
- **Fail-closed (Q6)** — the patched `_determine_batch_execution_and_padding` grants FULL only for a **pure** beam batch (`req_ids == beam keys`) at an **exact** width match (unpadded scheduled tokens), and otherwise downgrades to `NONE`. A native graph (which contains no beam kernels) is never replayed for beam tokens.

The framework's central prediction held: the blockers that prevented capture were exactly Q1/Q2, and the solution was exactly "fix the addresses + make the only varying quantity a device-buffer value." The measured result confirms the payoff of the chain's rationale — per-forward GPU time 52.2 ms → ~10 ms, kernel gap 142 µs → 1.7 µs, decode forward now a single `cudaGraphLaunch`.

### 6.2 Boundaries that remain (the next-step directions)

These are the places where #294 stops, each of which is a concrete next step toward the ultimate beam-decode graph goal. Ordered by priority:

1. **Width bucketing / padding.** #294 gates on an *exact* width match: a beam width of 100 does not replay a 128-row graph. The same "pad to a standard size" move the dispatcher already does for `num_tokens` (§3.9) has not been extended to beam width. Padding W to {128, 256, 512, 1024} would collapse the graph count and raise the hit rate — at the cost of a new correctness constraint: *virtual beams must not pollute KV or logits*, so every kernel in the graph must honor an `active_width` mask (the `beam_select` active-mask machinery from #288 is the natural home for this).
2. **Mixed beam + ordinary batches.** #294 grants FULL only for a *pure* beam batch. RFC #288's physical-batch model (multiple Sessions sharing one execution batch, §9 of #288) makes beam and non-beam requests coexist in one physical batch. Options: add a standard-attention path to the beam graph (larger, more complex graph), split the beam graph PIECEWISE-style (attention out of graph), or accept eager for mixed batches. This is a direct consequence of #288's "multi-microbatch physical batching", and the constraint list (§5.5) must be revisited accordingly.
3. **Decision on device (close the loop).** #294 captures only the forward; `constrained_top_k`, `beam_select`, and the host-side `argpartition`/Catalog logic still round-trip `logits[W, vocab]` through the host. #225's sixth blocker is therefore still open. With #288 moving beam decisions *below the Worker boundary* (Worker-owned `BeamSelector` / `ConstraintHeadExtension`), the device-side decision kernels become the natural next capture-boundary expansion — logits stay on device, only W `(token_id, parent)` pairs come back.
4. **Verify persistent suffix KV is actually landing.** #294 adds `decode_step` + `copy_suffix_kv_kernel`, which implies incremental decode is in play, but the eager substrate still routes through `_compute_beam_bounds` (`geometry.py`). Whether `b_past_suffix` now exceeds 0 — i.e. whether the per-step O(W·K²) full recompute is actually gone, or the graph only removed *launch* overhead while the KV recompute remains — is the highest-value correctness question to answer next. If `b_past_suffix` is still 0, the graph saves launches but not the recompute, and #225's Phase-1 persistent-suffix work is still outstanding.
5. **Capture-template and pool memory policy.** One graph per (width) plus a 5-D pool (~0.33 GiB per OneRec request) means template count and pool capacity need an LRU/cap policy — #225's open question 1, still unresolved. The `max_batch > 1` case (multiple sessions sharing one pool, partitioned by slot) is where the "neighbouring slot write" hazard the `copy_suffix_kv` docstring warns about actually materializes.
6. **NPU parity.** #294 is GPU-only (the NPU runner has no `_determine_batch_execution_and_padding`). #286 (Full ACLGraph Offload for NPU) is the follow-on: the same "per-step scalar → device buffer" idea must be re-derived for Ascend ACLGraph semantics.
7. **Gating identity after #288 lands.** #294's gating compares `req_ids` against beam keys and depends on `_current_beam_info`. Once #288's Worker-owned BeamRuntime (Session / epoch / item identity) replaces the request-id-based beam bookkeeping, the FULL-gating predicate should key off Session/epoch identity rather than `req_ids` set equality — otherwise mixed physical batching will silently mis-gate.

## 7. Next Steps

This RFC's judgment chain is complete; the remaining work is to turn §6.2 into a design RFC. The suggested order, following the chain's own "fix addresses before capture" principle:

1. **Verify persistent suffix KV** (§6.2.4): instrument `b_past_suffix` and confirm whether decode-segment recompute is actually eliminated on the eager substrate under #294. This is the correctness gate for everything downstream.
2. **Width bucketing** (§6.2.1): extend the exact-width gate to padded widths with an `active_width` mask, collapsing graph count.
3. **Device-side decision** (§6.2.3): in the direction of #288, move the decision chain below the Worker boundary and eliminate `logits` D2H.
4. **Mixed-batch handling** (§6.2.2) and **template/memory policy** (§6.2.5): resolve the physical-batching and pool-capacity questions #288 raises.
5. **NPU alignment** (§6.2.6) as a parallel track once the GPU semantics are settled.

Each of these reuses the same tool this RFC documents: ask *which layer* is responsible, run the six questions, and verify with the Harness before claiming graph-safety.

---

### Feedback Period.

_No response_

### CC List.

_No response_

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://vllm-gr.readthedocs.io), which can answer lots of frequently asked questions.
