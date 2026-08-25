## Overview
Binary soft decoder of LDPC codes with python cTypes wrapper. Supports the following decoders:
* sum-product
* min-sum with scale array and offset array
* Layered (horizontally and vertically) version of both decoders

For the sum-product case, a message passing algorithm is implemented with logarithm of input log-likelihood ratios using. This prevents us from hyperbolic tangents multiplication.

## Implementation notes
This repository has GLDPC decder implememntation descriped in https://ieeexplore.ieee.org/document/11131149