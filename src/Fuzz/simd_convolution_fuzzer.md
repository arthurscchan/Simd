# FP32 Convolution Fuzzer

## Rationale

The requested coverage gap is the FP32 convolution implementation family. This
target exercises valid convolution geometry rather than mostly rejecting random
parameter structures. Each input initializes both
`Simd::Base::SynetConvolution32fInit` and `SimdSynetConvolution32fInit`, guaranteeing
an explicit Base path as well as a runtime-dispatched path without directly
calling unsupported ISA implementations. No other new fuzz target is required.

The contract follows `TestSynetConvolution32f.cpp`, `SimdSynetConvParam.h`, and the
FP32 public API. Source and destination types are FP32 with matching contiguous
NCHW or NHWC layouts. Weight storage has `kernelY*kernelX*srcC*dstC/group`
elements, arranged as in the test for the selected layout. There is no tensor
stride parameter. Channel counts are positive and at most 64, groups divide both
channel counts, batch is 1 or 2, axes are at most 32, kernels are at most 7,
strides are 1 through 3, dilations are 1 or 2, and each padding edge is at most 3.

## Input Schema

Inputs shorter than 15 bytes are ignored. Byte 0 selects the family using
`(byte + 256 - 'A') % 8`. Thus printable `A` through `H` map directly to the eight
families, and every possible byte still selects a family. Header bytes 1 through
14 are normalized as `uint8_t(byte - '0')`; printable `0` means zero and `1`
means one. All 256 values remain useful mutation inputs.

| Byte | Meaning |
| --- | --- |
| 0 | Family A through H |
| 1 | Batch: 1 plus value modulo 2 |
| 2, 3 | Independent source height and width variations |
| 4, 5 | Source and destination channel variations; depthwise shares channels |
| 6, 7 | Family-specific stride, dilation, kernel, or group variations |
| 8, 9, 10, 11 | Padding controls; some families use byte 8 for symmetric padding |
| 12 | Identity or Relu, modulo 2 |
| 13 | Which context uses caller scratch: Base for even, dispatcher for odd |
| 14 | Bias present for even, absent for odd |
| 15 onward | Tensor-value bytes |

Payload bytes map to finite values in `[-1,1]` by `byte/127.5f - 1`. A single
cyclic payload cursor fills source, weights, initial destination, bias, and
activation parameters in that order. An empty payload fills zeros. Identity
and Relu need no activation parameters, but a live array of at least two floats
is provided as in the tests. Output values are not constrained after forward.

`max_len=266383` permits a 15-byte header plus one byte per value for two
131072-element tensors (batch 2, 64 channels, 32x32), 4096 pointwise weights,
64 bias values, and 64 activation values. Short inputs are useful because the
payload repeats; this is a useful value-data ceiling, not a required input size
or a claim that every theoretical maximal shape fits the MAC budget. There is
no dictionary, binary corpus, or corpus generator.

## Families And Text Seeds

The table describes the checked-in seed headers, not immutable shapes. Mutations
vary axes, channels, batch, activations, bias, scratch ownership, and selected
geometry within each family's constraints.

| Seed | Family and initial geometry |
| --- | --- |
| `direct_nchw.txt` | A: NCHW, 3 to 32 channels, 7x7, 3x3 stride 2, pad 0, destination 3x3 |
| `depthwise_3x3.txt` | B: NHWC, group=srcC=dstC=32, 7x8, 3x3 stride 1, pad 1 |
| `depthwise_7x7.txt` | C: NHWC, group=srcC=dstC=32, 7x7, 7x7 stride 1, pad 3 |
| `pointwise_nhwc.txt` | D: NHWC, 32 to 32 channels, 8x9, 1x1, stride 1, pad 0 |
| `grouped_dilated.txt` | E: NCHW, 16 to 16 channels, group 2, 13x13, 3x3, dilation 2x1, pad 0 |
| `winograd_2x2.txt` | F: NHWC, 32 to 32 channels, 6x7, 3x3, pad 1, Base 2x2 tile |
| `winograd_3x3.txt` | G: NHWC, 32 to 32 channels, 9x9, 3x3, pad 1, Base 3x3 tile |
| `winograd_4x4.txt` | H: NHWC, 32 to 32 channels, 12x12, 3x3, pad 1, Base 4x4 tile |
| `winograd_unpadded_batch2.txt` | H: batch 2, NHWC, 32 to 32 channels, 12x12, 3x3, pad 0 |

A retains small input-channel counts and source axes at least 7 to avoid the
earlier NCHW GEMM-NT small-source selector. B and C select NHWC depthwise. D
targets accelerated NHWC pointwise direct and Base GEMM (Base NHWC direct's
predicate is disabled). E always has groups greater than 1 and Y dilation 2,
bypassing direct and Winograd to exercise grouped GEMM-NN and image lowering;
groups vary over 2, 4, 8, with 3x3, 5x5, 7x7 kernels and independent X dilation
and strides.

F through H retain group 1, more than 16 input channels, unit stride/dilation,
3x3 kernels, and symmetric padding 0 or 1. The actual Base Winograd predicate
also requires NHWC axes at least 4 and batch*height*width at least 36. The Base
tile selector first takes 4x4 when both axes are at least 8 and batch area is at
least 144; otherwise it takes 3x3 when both axes are at least 6, batch area is at
least 81, and both destination axes are divisible by 3; otherwise it takes 2x2.
The three main seeds satisfy these predicates at batch 1. Axis, batch, channel,
and padding mutations explore nearby thresholds and tails. Optional 2x2,
1x3, and 1x5 kernels are deliberately not added.

`SimdSynetConvolution32fInfo` is queried for each context after parameter setup.
Its result is available for debugger observation; the harness does not print it
or assert a particular ISA, accelerated algorithm, or tile choice. Dispatcher
selection depends on the build and hardware, and can fall back to Base.

## Bounds And Lifetime

Effective kernels and destination axes are computed in signed 64-bit arithmetic
only after checking the small operand bounds. Negative numerators and outputs
outside 1 through 32 are rejected before calling initialization. Bounded positive
dimensions then make the unsigned tensor-count and 64-bit MAC products safe.

Each forward costs at most 2097152 explicit convolution MACs, including batch.
There are exactly two forwards for accepted inputs, at most 4194304 explicit
MACs per input. Algorithm transforms, packing, and runtime GEMM selection have
additional overhead not represented by this mathematical MAC bound.

The aggregate buffer ceiling is 8 MiB, expressed as float elements. Accounting
includes the shared caller source, weights, destination, bias, activation array,
both queried internal-buffer sizes, and both queried external-buffer sizes.
Before `SetParams`, a conservative per-context growth allowance of
`max(weightCount,36*srcC*dstC)+256` floats reserves the largest selected Winograd
filter transform or original weights plus parameter padding. Internal sizes are
queried again after setup and before the second forward. External reservations
cover the single caller scratch allocation and the other context's possible
lazy internal scratch. This caps accounted buffers, not opaque context objects,
allocator metadata, or unreported transient allocations inside the library.

Both init results are checked before any context query. Scratch sizes are float
element counts, never byte counts. One context receives caller scratch and the
other receives `NULL` to exercise lazy internal scratch; byte 13 swaps roles
without adding forwards. `SetParams` precedes `Forward` on both contexts. All
caller arrays, including weights, bias, and activation parameters, outlive both
RAII context owners regardless of the returned internal-weight flag. Both owners
use `SimdRelease`, including early returns. There are no library changes.

## Validation Limits

This is sanitizer-only fuzzing. The Base and dispatched outputs are not compared:
different GEMM reduction orders and Winograd transforms require calibrated
numerical tolerances, which have not been established here. The harness does not
assert output finiteness or numerical correctness.

Only the required C++11 syntax-only check and static integration/corpus checks
were performed. No build, link, fuzzer execution, runtime ISA observation, or
coverage measurement was performed. The intended family reachability above is
derived from source predicates, not a runtime-tested coverage claim.
