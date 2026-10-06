## Overview
Decoder of LDPC codes. Supports the following decoders:
* Bit-flipping
* Min-sum
* Multi-bit gradient descent bit-flipping
* PGDBF with momentum
* EPMGDBF
* FTGDBF (fixed-threshold gradient descent bit-flipping)
* E-GDBF V1 and V2 (edge-wise gradient descent bit-flipping), Python and C++
* GDMS (gradient-descent min-sum) with Python and C++ implementations

## Implementation notes
This repository has 
* bit-flipping decoder implementation described in https://www.researchgate.net/publication/228977165_Introducing_Low-Density_Parity-Check_Codes
* min-sum decoder implementation described in https://www.researchgate.net/publication/3159896_Reduced_complexity_iterative_decoding_of_low-density_parity_check_codes_based_on_belief_propagation
* multi-bit gradient descent bit-flipping decoder implementation described in http://arxiv.org/abs/0711.0261v2
* PGDBF with momentum decoder implementation described in https://arxiv.org/pdf/2204.02359

### GDMS

GDMS keeps a real bit state `x[n]` and a real extrinsic edge state
`q[n->m]`.  The channel and bipolar part of the ascent direction for either
state `z` is

```
channel_weight * y
- 4 * bipolar_weight * z * (z*z - 1)
```

This is the gradient of
`channel_weight * dot(y,z) - bipolar_weight * sum((z*z-1)^2)`.
The bit update adds every incident min-sum check message; the edge update
adds the same sum except for the recipient check.  The former `alpha*y` and
`-l2*z` terms are no longer used.  Python GDMS/MGDMS, C++ GDMS/MGDMS, and
the sign-only GDMS core used by E-GDBF V4 share these parameter names and
this update rule.

### E-GDBF

E-GDBF stores one persistent hard variable-to-check opinion on every Tanner-
graph edge.  Its synchronous message updates are

```
q[i->a] = sign(y[i])                                      # initialization
r[a->i] = product(q[j->a] for j in N(a) excluding i)
g[i->a] = alpha*y[i] + sum(r[b->i] for b in N(i) excluding a)
E[i->a] = q[i->a]*g[i->a] + rho[l[i->a]]
E_th = min(E over all edges of the word) + delta
q_new[i->a] = -q[i->a] if E[i->a] <= E_th else q[i->a]
```

V2 changes only the threshold: compute the mean edge energy of each variable
node, then use the smallest of these means plus `delta`. Individual edges are
still flipped by comparing their own energy to that common threshold. V1 and
V2 are separate decoder types in `decoder_factory.py`; the existing `run.sh`
accepts `experiments/experiment_cpp_egdbf_v2.json` for V2.

The hard a-posteriori word is computed separately:

```
g_post[i] = alpha*y[i] + sum(r[a->i] for a in N(i))
x[i] = sign(g_post[i])
```

There is no min-sum magnitude operation and no multiplication of a check
message by the current hard-word bit.  The hard word is not fed back to the
checks: they use the independently stored `q[i->a]` signs on the next
iteration. The `rho` term changes the energy of recently flipped edges. The
threshold `delta` is non-negative and defaults to `0` in older configs.
This is an edge-state message-passing modification, not a claim of a
different proven scalar objective.

Set `L` to `0` and `rho` to `[]` to disable momentum in both the Python and
C++ implementations. With `L > 0`, `rho` must contain exactly `L` values.
