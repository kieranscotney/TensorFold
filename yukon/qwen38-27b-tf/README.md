# Qwen3.8-27B TensorFold speed track: kieranscotney's finds

Code finds from my work on the Yukon challenge `GumbiiDigital/mlxfast-qwen38-27b-tf`
(Qwen3.8-27B, TensorFold Zig/Metal engine, DFlash speculative decoding, Apple M5 Max).
Every change keeps output token-for-token identical to the plain engine, which is what the challenge requires.

Each patch is relative to the TensorFold root (`zig/...`), taken from the challenge's vendored copy
(`Vendor/TensorFold`). The challenge tree moved quickly as solvers stacked on each other, so **each patch
applies to the submission it was built on**, listed in its section, not to this branch's `main`.
Ported code from other solvers is called out wherever a patch carries it.

Already upstream, not repeated here:
- In-place ring-cache attention for the DFlash drafter: ashhart/TensorFold#1727 (merged).
- Drafter head lane pipeline compiled at load: ashhart/TensorFold#1734.

Fuller write-up, measurements and dead ends: https://github.com/kieranscotney/mlxfast-qwen38-tf-work

Score = prefill_gain^0.25 × decode_gain^0.75. "pf" and "dec" below are local tokens/s on an M5 Max, alternating A/B runs.

## 1. Prompt projections take the lean prefetch (`01-prompt-lean.patch`)
- **Base:** submission `2d06a685` plus the head pre-compile commit.
- **Change:** `Key.lean` was `reg and !k.prompt`; it is now `reg`, so prompt-row reg projections use the codes-only
  prefetch (`TF_REG_LEAN`, kshitij-hash) that decode already used. One group of codes in flight; metadata and sums per group.
- **Result:** prefill 1028.5 → 1054.2 tok/s (+2.5%, every round), decode unchanged. Exact by construction.
  No codes in flight (LEAN0) loses 5.4% and two groups (LEAN2) gains nothing, so one group is optimal.
- **Official:** in `ea8f8271` (1.37849). Carried, with credit, in leader `2c41feb1` (1.53945) and later leaders.

## 2. Prompt-row SwiGLU in the gate/up epilogue (`02-prompt-row-swiglu.patch`)
- **Base:** submission `e75967a0`.
- **Change:** extends fkiene's decode-only `TF_SWIGLU` fused epilogue to prompt rows: 16-row blocks, split-K 1,
  K range without split offsets, Y/HS row offsets. Same bf16 rounding of gate/up and the same `h` expression as
  the unfused path, so results stay exact. The `quant_gpu.zig` hunk also carries the `prepareDecode` helper from #1734.
- **Result:** prefill 1105.2 → 1128.5 tok/s (+2.1%); with the other pieces of `23d85725`, decode +1.2%.
- **Official:** `23d85725` scored 1.52942. Carried, with credit, in leader `2c41feb1` (1.53945).

## 3. Drafter catches up on a prompt in 1024-row passes (`03-drafter-flush-1024.patch`)
- **Base:** submission `5317b10e` (my stack on `ea8f8271`).
- **Change:** the drafter absorbed the prompt's tap rows 128 at a time, which is nine frames and nine waits for a
  1065-token prompt. `flush_rows = 1024` makes that one or two passes.
- **Result:** decode window +0.15% (57.53 vs 57.44, 3 rounds). It also removes ~11 ms from the first draft round.
- **Official:** `f3f8f654` scored 1.38132. Carried, with credit, in leader `2c41feb1`.

## 4. Drafter top-k as a SIMD-shuffle tournament (`04-drafter-simd-topk.patch`)
- **Base:** submission `38c11ee6`.
- **Change:** `bf16_topk.metal` runs each rank's tournament inside a simdgroup with `simd_shuffle_xor`, then needs
  only one threadgroup-memory step (one barrier) per rank across simdgroups, in place of the many barrier rounds before.
- **Result:** small on its own (drafter lattice/top-k was ~97 µs of a 3.6 ms frame). Draft choices are unchanged:
  3×1024-token tapes identical to the base.
- **Official:** part of my promoted entry `07fd50da` (1.63957).
