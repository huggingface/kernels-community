"""Layer-level entry points, for `kernels`' layer mapping.

`transformers` marks the layers a kernel may replace with `@use_kernel_forward_from_hub("<name>")`, and a
kernel repository supplies a class of that name whose `forward` is grafted onto the real module. The module
keeps its own parameters and constructor -- the class here is stateless, which `kernels` enforces: no
`__init__`, no members besides `forward`, and a signature matching the layer being replaced.

Named `SoftmaxTopKRouter` rather than `TopKRouter` because the rewrite below is only valid for routers
that score with a softmax. One that scores with a sigmoid returns the same shapes and would graft
cleanly onto this, then be quietly wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import ops


class SoftmaxTopKRouter(nn.Module):
    """A softmax router's top-k, in two dispatches instead of four.

    The model softmaxes over every expert, takes the top `k`, then renormalizes those `k`. Softmax is
    monotonic, so the top `k` probabilities belong to the top `k` logits, and renormalizing them is a
    softmax over just those logits -- `exp(x_i) / sum_{j in topk} exp(x_j)` either way. So the whole
    tail is skipped: select first, then softmax the `k` that survived, which the kernel does in the
    same pass.

    Worth it because a MoE decode step is dispatch-bound rather than arithmetic-bound: this is 40
    layers x 2 fewer operations, and the top-k itself stops being a full sort.

    `indices` comes back int32, which is what an expert-routed matmul wants.
    """

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        scores, indices = ops.top_k(router_logits, self.top_k, True)
        return router_logits, scores.to(router_logits.dtype), indices
