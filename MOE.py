import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mlp import GatedMLP


class GatedMoE(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        num_experts=4,
        top_k=2,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        expert_capacity_ratio=1.2, 
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_ratio = expert_capacity_ratio

        self.experts = nn.ModuleList([
            GatedMLP(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                activation=activation,
                bias=bias,
                multiple_of=multiple_of,
                **factory_kwargs
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(in_features, num_experts, bias=False, **factory_kwargs)

    def forward(self, x):
        """
        x: [B, L, D]
        Return: [B, L, D]
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [N, D]
        N = x_flat.size(0)
        capacity = int(self.expert_capacity_ratio * N / self.num_experts)

        # Step 1: Gating + top-k
        gate_logits = self.gate(x_flat)                   # [N, E]
        topk_scores, topk_indices = gate_logits.topk(self.top_k, dim=-1)  # [N, K]
        topk_weights = F.softmax(topk_scores, dim=-1)     # [N, K]

        # Step 2: Expert forward with capacity mask
        output = torch.zeros_like(x_flat)                 # [N, D]

        for k in range(self.top_k):
            idx_k = topk_indices[:, k]  # [N]
            for expert_id in range(self.num_experts):
                mask = (idx_k == expert_id)               # [N]
                indices = mask.nonzero(as_tuple=False).squeeze(1)  # 选中 token 的位置
                if indices.numel() == 0:
                    continue
                if indices.numel() > capacity:
                    indices = indices[:capacity]          # truncate to capacity

                x_sel = x_flat[indices]                   # [M, D]
                y_sel = self.experts[expert_id](x_sel)    # [M, D]
                w_sel = topk_weights[indices, k].unsqueeze(-1)  # [M, 1]
                output[indices] += w_sel * y_sel         

        return output.view(B, L, D)

