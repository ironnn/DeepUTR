from functools import partial
import torch
from mamba_ssm.modules.block import Block
from torch import nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mlp import GatedMLP
from MOE import GatedMoE
from Caduceus_Bimamba import BiMambaWrapper
from performer_pytorch import PerformerLM


def create_block(
    d_model,
    d_intermediate=0,
    mixer_type="mamba",
    ssm_cfg=None,
    norm_epsilon=1e-5,
    residual_in_fp32=False,
    fused_add_norm=True,
    layer_idx=0,
    bidirectional=False,
    bidirectional_strategy="add",
    bidirectional_weight_tie=True,
    device=None,
    dtype=torch.float32,
    mlp_type="moe", 
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # === Mixer ===
    if mixer_type == "mamba":
        bidirectional_kwargs = {
            "bidirectional": bidirectional,
            "bidirectional_strategy": bidirectional_strategy,
            "bidirectional_weight_tie": bidirectional_weight_tie,
        }
        mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)


    # === MLP or MoE ===
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    elif mlp_type == "moe":
        mlp_cls = partial(GatedMoE, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    elif mlp_type == "mlp":
        mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    else:
        raise ValueError(f"Unsupported mlp_type: {mlp_type}")

    # === Norm ===
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)

    # === Block ===
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class jambaregression(nn.Module):
    def __init__(
        self,
        num_mamba_blocks=3,
        d_model=64,
        d_intermediate=512,
        vocab_size=16,
        max_seq_len=4096,
        fused_add_norm=True,
        residual_in_fp32=True,
        transformer_depth=2,
    ):
        super().__init__()

        self.num_mamba_blocks = num_mamba_blocks
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.transformer_depth = transformer_depth

        self.performer = PerformerLM(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            dim=d_model,
            depth=transformer_depth,        
            heads=4,
            causal=False,
            nb_features=64,
            feature_redraw_interval=1000,
            generalized_attention=False,
            kernel_fn=torch.nn.ReLU(),
            reversible=False,
            ff_chunks=1,
            use_scalenorm=False,
            use_rezero=False,
            ff_glu=True,
            emb_dropout=0.0,
            ff_dropout=0.0,
            attn_dropout=0.0,
            local_attn_heads=2,
            local_window_size=24,
            rotary_position_emb=True,
            shift_tokens=True,
        )

        self.mamba_blocks = nn.ModuleList([
            create_block(
                d_model=d_model,
                d_intermediate=d_intermediate,
                mixer_type="mamba",
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                layer_idx=i,
                mlp_type="mlp" if i % 2 == 0 else "moe",
            )
            for i in range(num_mamba_blocks)
        ])
        
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1)

        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(24, 1)
        
        self.celltype_emb = nn.Embedding(num_embeddings=27, embedding_dim=self.conv3.out_channels*2)
        self.fc1 = nn.Linear(self.conv3.out_channels*2 + self.conv3.out_channels*2 + 1, 24)
        

    def forward(self, x, attention_mask=None, tpm=None, cell_type=None, **kwargs):
        # x: [B, L]  token ids

        hidden_states = self.performer(x, mask=attention_mask, return_encodings=True)  # [B, L, D]
        residual = None


        for block in self.mamba_blocks:
            hidden_states, residual = block(hidden_states, residual)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
            hidden_states = hidden_states * mask  # zero out padding tokens

        x = hidden_states.permute(0, 2, 1)  # (B, D, L)
        x = self.conv1(x)                   # (B, C1, L1)
        x = self.conv2(x)                   # (B, C2, L2)
        x = self.conv3(x)                   # (B, C3, L3)
        x = x.permute(0, 2, 1)              # (B, L3, C3)

        mean_pool = x.mean(dim=1)           # (B, C3)
        max_pool, _ = x.max(dim=1)          # (B, C3)
        agg = torch.cat([mean_pool, max_pool], dim=1)  # (B, C3*2)
        
        cell_emb = self.celltype_emb(cell_type.squeeze(-1))     # (B, C3*2)

        tpm = tpm.unsqueeze(1)
        all_feat = torch.cat([agg, tpm, cell_emb], dim=1)  # (B, C3*2+C3//2+1)

        x = self.fc1(all_feat)
        x = self.silu(x)
        output = self.fc2(x)                # (B, 1)



        return output