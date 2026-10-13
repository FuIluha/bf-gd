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
* MPGD (alternating CN/VN gradient steps), Python and C++

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

### MPGD

Algorithm names: `mpgd` (Python, also registered in the visualizer) and
`cpp mpgd` (native loop for simulations and tuning). Both implement the same
deterministic algorithm. One iteration is a complete CN phase followed by a
VN phase. The decoder parameters are `alpha`, `eta_r`, `eta_v`, `tau_r`, and
`tau_v`; defaults are `1.0, 0.1, 0.1, 0.1, 0.1` respectively.

States `s[i->a]` and `u[a->i]` are soft signs, not LLRs. Define binary entropy
`h(z) = -(1+z)/2*log((1+z)/2) - (1-z)/2*log((1-z)/2)`. The local objectives are

```
R(s) = sum_a product(s[i->a] for i in N(a)) + tau_r*sum_edges h(s)
V(u) = alpha*sum_i y[i]*sum(u[a->i] for a in N(i))
       + sum_i sum(u[a->i]*u[b->i] for unordered pairs a,b in N(i))
       + tau_v*sum_edges h(u)
```

Initialization and synchronous updates:

```
s[i->a] = clip(tanh(alpha*y[i]/2))
u[a->i] = clip(s[i->a] + eta_r*(product(s[j->a] for j in N(a) excluding i)
                               - tau_r*atanh(s[i->a])))
s_new[i->a] = clip(u[a->i] + eta_v*(alpha*y[i]
                    + sum(u[b->i] for b in N(i) excluding a)
                    - tau_v*atanh(u[a->i])))
x[i] = sign(sum(s_new[i->a] for a in N(i)))
```

Each phase reads a fixed snapshot of its inputs. Sending the updated state
copies the same edge opinion to the other endpoint; there is no additional
product-message or coordinate-maximum assignment between gradient phases.
Products exclude the recipient without dividing by its value, so zeros are
supported. `clip` uses `[-1+1e-6, 1-1e-6]`; `tau_r=0` or `tau_v=0` disables
that phase's entropy term. Both step sizes must be positive. No momentum,
step decay, or direct-assignment mode is used.

A zero final vote follows the channel sign (`y=0` means `+1`). The initial
decision uses the channel sign. Parity is checked before the first cycle and
after every complete cycle; the hard decision is not fed back to checks.
The Python decoder's production `step_state` also drives visualization.

This is a proposed local message-passing algorithm with edge memory. Fixed
projected steps are not guaranteed to increase either objective or a common
objective for a whole cycle. It is not standard sum-product and no decoding
performance claim is implied by the objectives.
