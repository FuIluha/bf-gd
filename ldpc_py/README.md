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

E-GDBF stores one persistent hard variable-to-check opinion on every Tanner-
graph edge.  Its synchronous message updates are

```
q[i->a] = sign(y[i])                                      # initialization
r[a->i] = product(q[j->a] for j in N(a) excluding i)
g[i->a] = alpha*y[i] + sum(r[b->i] for b in N(i) excluding a)
          + rho[l[i->a]]*q[i->a]
q_new[i->a] = sign(g[i->a])
```

The hard a-posteriori word is computed separately:

```
g_post[i] = alpha*y[i] + sum(r[a->i] for a in N(i))
x[i] = sign(g_post[i])
```

There is no min-sum magnitude operation and no multiplication of a check
message by the current hard-word bit.  The hard word is not fed back to the
checks: they use the independently stored `q[i->a]` signs on the next
iteration.  Momentum acts as inertia in the direction of the current edge
sign.  This is an edge-state message-passing modification, not a claim of a
different proven scalar objective.
