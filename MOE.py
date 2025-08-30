import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mlp import GatedMLP

# class GatedMoE(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         hidden_features=None,
#         out_features=None,
#         num_experts=4,
#         top_k=2,
#         activation=F.silu,
#         bias=False,
#         multiple_of=128,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()
#         factory_kwargs = {"device": device, "dtype": dtype}
#         self.num_experts = num_experts
#         self.top_k = top_k

#         self.experts = nn.ModuleList([
#             GatedMLP(
#                 in_features=in_features,
#                 hidden_features=hidden_features,
#                 out_features=out_features,
#                 activation=activation,
#                 bias=bias,
#                 multiple_of=multiple_of,
#                 **factory_kwargs
#             ) for _ in range(num_experts)
#         ])

#         self.gate = nn.Linear(in_features, num_experts, bias=False, **factory_kwargs)

#     def forward(self, x):
#         """
#         x: [B, L, D]
#         Return: [B, L, D]
#         """
#         B, L, D = x.shape
#         x_flat = x.view(-1, D)  # [B*L, D]

#         # === Step 1: Get softmax scores for all experts ===
#         gate_logits = self.gate(x_flat)  # [B*L, E]
#         gate_scores = F.softmax(gate_logits, dim=-1)  # soft routing

#         # === Step 2: Select top-k expert indices per token ===
#         topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # [B*L, K]

#         # === Step 3: Accumulate weighted expert outputs (only top-k) ===
#         out_flat = torch.zeros_like(x_flat)
#         for k in range(self.top_k):
#             expert_idx = topk_indices[:, k]            # [B*L]
#             mask = (expert_idx.unsqueeze(1) == torch.arange(self.num_experts, device=x.device).unsqueeze(0))  # [B*L, E]
#             for i in range(self.num_experts):
#                 token_mask = mask[:, i]                # [B*L]
#                 if token_mask.any():
#                     x_selected = x_flat[token_mask]
#                     y_selected = self.experts[i](x_selected)  # expert_i forward
#                     weight = topk_scores[token_mask, k].unsqueeze(-1)
#                     out_flat[token_mask] += weight * y_selected

#         return out_flat.view(B, L, -1)



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
        expert_capacity_ratio=1.2,  # 新增：每个专家容量比例
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
                output[indices] += w_sel * y_sel          # 聚合

        return output.view(B, L, D)

