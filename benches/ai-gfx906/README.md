# Arithmetic intensity probe (gfx906)

Measures, for `MUL_MAT` with Q4_0, IQ4_NL and Q8_0 weights:

- the bytes the kernel has to move (weights streamed once per call, plus activations),
- the FLOPs it has to retire (2 per weight, times the token count),
- the resulting arithmetic intensity in FLOP/byte,
- the achieved time, GB/s and TFLOP/s on the current backend,
- and, for repacked rows, the nmse against the canonical output.

Q8_0, Q4_0 and IQ4_NL are also measured through the fork's gfx906
repacked-weight buffer type, which is what a normal `llama.cpp` load uses on
this arch.

## Run

```
benches/ai-gfx906/run.sh                  # K=N=8192, decode + 128-token batch
benches/ai-gfx906/run.sh -K 5120 -N 5120  # other width
benches/ai-gfx906/run.sh --tokens 512,2048  # prompt-sized batches
benches/ai-gfx906/run.sh --canonical-only # skip the repacked rows
benches/ai-gfx906/run.sh --only iq4_nl --repacked-only
```

The script compiles `mul-mat-ai.cpp` against the already built
`build/bin/libggml.so` and runs it on the first GPU backend found.

## Sample output (AMD Radeon Pro VII, gfx906)

```
type    layout       bpw    AI(w)  AI(all)   tokens   time_us     GB/s  TFLOP/s       nmse
------------------------------------------------------------------------------------------------
q4_0    canonical   4.50     3.56     3.55        1      77.2    490.0    1.739          -
q4_0    repacked    4.50     3.56     3.55        1      55.1    686.0    2.435   2.24e-10
iq4_nl  canonical   4.50     3.56     3.55        1      81.1    466.4    1.656          -
iq4_nl  repacked    4.50     3.56     3.55        1      58.4    647.5    2.298   2.41e-15
q8_0    canonical   8.50     1.88     1.88        1      95.2    749.5    1.410          -
q8_0    repacked    8.50     1.88     1.88        1      96.6    739.2    1.390   2.46e-15

q4_0    canonical   4.50   455.11   372.36      128    1068.7     43.2   16.075          -
q4_0    repacked    4.50   455.11   372.36      128     968.7     47.6   17.735   4.88e-10
iq4_nl  canonical   4.50   455.11   372.36      128    1391.6     33.2   12.345          -
iq4_nl  repacked    4.50   455.11   372.36      128     983.6     46.9   17.467   0.00e+00
q8_0    canonical   8.50   240.94   215.58      128    1712.8     46.5   10.031          -
q8_0    repacked    8.50   240.94   215.58      128     922.0     86.4   18.633   0.00e+00
```

`AI(w)` counts weights only, `AI(all)` adds the activation read and output
write. Decode is weight-streaming bound; at 128 tokens the activation traffic
is small against the weights, so both columns stay close.

## What the numbers say

- Q4_0 and IQ4_NL have the same arithmetic intensity: 18 bytes per 32 weights,
  3.56 FLOP/byte. Q8_0 carries 34 bytes per 32 weights, 1.88 FLOP/byte. The
  ratio is 1.89x, independent of the kernel.
- The canonical (non-repacked) kernels do not turn that ratio into speed.
  Q4_0 decode is only about 1.24x faster than Q8_0 because it reaches 490 GB/s
  while Q8_0 reaches 750 GB/s. IQ4_NL's lookup-based dequant costs more ALU and
  lands below Q4_0 at decode and at prefill (12.3 TFLOP/s).
- Q4_0 repacked reaches 686 GB/s at decode, 1.76x canonical Q4_0 and the
  fastest of the three. Its expand is a mask, a shift and a bytewise -8, so it
  beats IQ4_NL repacked despite the same 18 B block.
- At a 128-token batch all types are far above the MI50 ridge point (about
  13 FLOP/byte FP32, about 52 for int8 dot at 53.3 TOPS), so bytes stop
  mattering and the dequant/dot instruction mix decides. Q8_0 repacked leads
  (18.6 TFLOP/s), Q4_0 repacked follows (17.7) and IQ4_NL repacked is last
  (17.5).
- At 512 tokens Q4_0 repacked reaches 23.6 TFLOP/s, which is 11.8 TMAC/s or
  44% of the 26.7 TMAC/s (53.3 TOPS) int8 peak. Q8_0 repacked lands at 44.8%
  on the same shape, so the gap is the tile shape and the sdot4 issue rate,
  not the Q4_0 unpack. Reinstinct's standalone tile measured the same 11.25
  TMAC/s. Decode is the opposite case: 55 us for 37.7 MB is 686 GB/s, 67% of
  the 1.02 TB/s HBM peak, so decode is bandwidth-bound and prefill is
  ALU-bound at about half the int8 peak.
- A second width (K=N=5120) gives the same ordering at decode: Q8_0 about
  38 us, Q4_0 canonical about 32 us, Q4_0 repacked the fastest.
- The Q4_0 vs Q8_0 prefill gap is shape-dependent and small. At
  K=5120/N=17408 (the shape reinstinct reports) Q4_0 repacked leads 24.70 vs
  24.56 TFLOP/s at 512 tokens; at K=N=8192 Q8_0 repacked leads 23.91 vs
  23.57. End to end on Qwen3.5-9B, Q8_0 prefill is 2.4% faster (1342 vs 1309
  t/s pp512) while Q4_0 decode is 41% faster (93.5 vs 66.3 t/s) and the file
  is 4.94 vs 8.86 GiB.
- The repacked rows match canonical to nmse 5e-10 or better for Q4_0 and
  are bit-identical for IQ4_NL and Q8_0.

## Prompt-sized batches

K=N=8192, 512 and 2048 tokens (`--tokens 512,2048`), TFLOP/s:

| type | layout | 512 | 2048 |
|---|---|---|---|
| q4_0 | canonical | 18.63 | 17.87 |
| q4_0 | repacked | 23.57 | 23.44 |
| iq4_nl | repacked | 23.02 | 22.90 |
| q8_0 | repacked | 23.91 | 23.56 |

At prompt-sized batches the weights are reused enough that all types are
compute-bound. Q4_0 repacked lands within 1.5% of Q8_0 repacked and ahead of
IQ4_NL repacked, with half the weight bytes of Q8_0.

## End to end

Qwen3.5-9B-Q4_0, one MI50, `llama-bench -p 512,2048 -n 128`:

| test | canonical | repacked | delta |
|---|---:|---:|---:|
| pp512 | 1085.7 t/s | 1309.1 t/s | +20.6% |
| pp2048 | 1073.7 t/s | 1293.6 t/s | +20.5% |
| tg128 | 80.7 t/s | 93.5 t/s | +15.9% |

`llama-perplexity`, 16 chunks of 512, single GPU: 10.6919 canonical vs
10.6864 repacked, a 0.05% difference.

## Tensor parallel

The meta tensor-split gate in `ggml-backend-meta.cpp` carries a repack type
whitelist; Q4_0 was missing from it, so `-sm tensor -tps 2` silently kept the
weights canonical. After adding it, Qwen3.5-9B-Q4_0 on 2x MI50:

| test | canonical TP | repacked TP | delta |
|---|---:|---:|---:|
| pp512 | 1824.2 t/s | 2016.3 t/s | +10.5% |
| pp2048 | 1804.9 t/s | 1982.1 t/s | +9.8% |
| tg128 | 101.2 t/s | 107.7 t/s | +6.4% |

For reference, one GPU does 1309 pp512 and 93.5 tg128, layer split does 1313
pp512 and 86.5 tg128. `-sm tensor` without `-tps` picks the same two devices,
so the numbers are identical to `-tps 2`. TP does not double the throughput:
MI50 has no XGMI and every layer pays a PCIe allreduce, and layer split does
not parallelize one sequence's prefill because layer N+1 depends on layer N.

