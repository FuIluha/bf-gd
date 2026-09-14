## Overview
Decoder of LDPC codes. Supports the following decoders:
* Bit-flipping
* Min-sum
* Multi-bit gradient descent bit-flipping
* PGDBF with momentum
* EPMGDBF
* E-GDBF (edge-wise gradient descent bit-flipping), Python and C++
* GDMS (gradient-descent min-sum) with Python and C++ implementations

## Implementation notes
This repository has 
* bit-flipping decoder implementation described in https://www.researchgate.net/publication/228977165_Introducing_Low-Density_Parity-Check_Codes
* min-sum decoder implementation described in https://www.researchgate.net/publication/3159896_Reduced_complexity_iterative_decoding_of_low-density_parity_check_codes_based_on_belief_propagation
* multi-bit gradient descent bit-flipping decoder implementation described in http://arxiv.org/abs/0711.0261v2
* PGDBF with momentum decoder implementation described in https://arxiv.org/pdf/2204.02359

### E-GDBF

E-GDBF keeps the PMGDBF flip rule, threshold, probability, and momentum, but
expresses the check contribution as an extrinsic message on every Tanner-graph
edge.  For an edge between check `a` and variable `i`,

```
r[a->i] = product(x[j] for j in N(a) excluding i)
E[a->i] = x[i] * r[a->i] + alpha*x[i]*y[i] + rho[l[i]]
E_post[i] = sum(E[a->i] for a in N(i))
```

There is no min-sum magnitude operation.  Every edge energy contains the full
channel and momentum terms.  Consequently, posterior aggregation weights these
terms by `deg(i)`, so E-GDBF is not algebraically equivalent to PMGDBF.  It is
an edge-wise modification of the Savin decoder, not a claim of a different
proven scalar objective.
