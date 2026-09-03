## Overview
Decoder of LDPC codes. Supports the following decoders:
* Bit-flipping
* Min-sum
* Multi-bit gradient descent bit-flipping
* PGDBF with momentum
* EPMGDBF

## Implementation notes
This repository has 
* bit-flipping decoder implementation described in https://www.researchgate.net/publication/228977165_Introducing_Low-Density_Parity-Check_Codes
* min-sum decoder implementation described in https://www.researchgate.net/publication/3159896_Reduced_complexity_iterative_decoding_of_low-density_parity_check_codes_based_on_belief_propagation
* multi-bit gradient descent bit-flipping decoder implementation described in http://arxiv.org/abs/0711.0261v2
* PGDBF with momentum decoder implementation described in https://arxiv.org/pdf/2204.02359