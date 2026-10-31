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
