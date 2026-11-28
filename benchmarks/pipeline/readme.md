# pipeline benchmark

Measures this complete, single-threaded operation:

```text
immutable interleaved input
  -> planar conversion
  -> brightness +20, saturated to [0, 255]
  -> gaussian blur, sigma 1.5 and radius 4
  -> owning interleaved output
```

Brightness leaves alpha unchanged; blur processes every channel, including alpha.
Borders replicate the image edges. Gray8, RGB8 and RGBA8 are measured at 256x256,
1920x1080 and 3840x2160. Inputs are deterministic random bytes generated outside timing.
No image decoding or disk access is timed.

## run

From the repository root, after building OpenCV as described in the main README:

```sh
cmake --preset benchmarks
cmake --build --preset benchmarks --target parhelion_bench_pipeline --parallel
./build/benchmarks/benchmarks/parhelion_bench_pipeline
```

Use `--quick` for five samples at 17x9 and 256x256, or `--help` for usage.
The executable always includes Parhelion; OpenCV and vImage follow the existing
`PARHELION_BENCHMARK_OPENCV` and `PARHELION_BENCHMARK_VIMAGE` options. With both disabled,
it runs just Parhelion and still checks the scalar reference.

## what is timed

Every call starts with the same existing input and returns a newly allocated output.
All per-call allocations, parameter preparation and temporary buffers are included.
Parhelion uses its public API, including the input copy needed to preserve the source
while converting owning storage. Other backends split the source directly into newly
allocated planes. No backend reuses scratch buffers or precomputed parameters across calls.

| column | included work |
|---|---|
| total ms | complete call, without internal stage clocks |
| setup | initial buffers and wrappers; Parhelion's input copy; vImage weights and lookup table |
| split | interleaved to planar, including allocations inside public conversion calls |
| brighten | saturated color-channel adjustment |
| blur | convolution, including scratch query/allocation and any internally prepared weights |
| merge | planar to interleaved, including allocations inside public conversion calls |
| other | return overhead and intermediate cleanup after the last stage |
| time/ours | backend total divided by Parhelion total; above 1 means Parhelion was faster |

Final output destruction and correctness checks are outside timing. Stage values come
from separate instrumented calls; they and the total are independent medians and need
not add up. Each measurement has three warmups and 31 samples, with rotating backend
order and alternating instrumented/uninstrumented call order. This is a warm repeated-input
benchmark, not a cold-memory or streaming-video measurement. Run it without competing workloads.

OpenCV parallel dispatch is disabled with `cv::setNumThreads(0)`; every vImage
operation uses `kvImageDoNotTile`.
All backends use planar intermediates to measure the same layout pipeline. These results
do not establish the fastest possible interleaved pipeline each library could implement.

## correctness and recorded results

Before timing, an independent scalar implementation checks constant, random and impulse
inputs, tiny images, vector tails, radii 0/1/4, negative brightness and RGBA alpha behavior.
Every measured size is also checked against that reference before timing and after the
last sample; source bytes must remain unchanged. Allowed byte error is 1 for Parhelion
and vImage, 2 for OpenCV, and 0 for radius-zero cases because blur rounding differs.

[results.txt](results.txt) records one full run with its CPU, OS, compiler, flags and command.
Results describe that machine and build, not every SME2 implementation.
