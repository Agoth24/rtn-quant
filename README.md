# RTN Quantization Experiment

## Objective

Investigate the effect of the rounding operation in Round-To-Nearest quantization.

## Method

- Develop a general RTN quantizer for floating points.
    - implement an alternate quantizer without the rounding operation.

- Run both quantizers on a set of randomly generated numbers; then dequantize.

- Compare errors of the two quantization methods, iterate from observation.