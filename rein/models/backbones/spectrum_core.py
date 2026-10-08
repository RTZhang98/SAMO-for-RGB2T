import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

class LoRA_Disentanglement(nn.Module):
    def __init__(self, dim, rank=16, alpha=1.0):
        super().__init__()

        self.semantic_lora = LoRA_Efficient(dim, rank, alpha)
        self.domain_lora = LoRA_Efficient(dim, rank, alpha)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):

        sem = self.semantic_lora(x)
        dom = self.domain_lora(x)

        concat_feat = torch.cat([sem, dom], dim=1)

        concat_feat = concat_feat.transpose(1, 2)

        fused = self.fusion(concat_feat)

        fused = fused.transpose(1, 2)
        
        return sem, dom, fused

class LoRA_Efficient(nn.Module):
    def __init__(self, dim, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        self.scaling = alpha / rank
        
    def forward(self, x):

        x_t = x.transpose(1, 2)

        lora_out = self.lora_down(x_t)
        lora_out = self.lora_up(lora_out)
        lora_out = lora_out * self.scaling

        lora_out = lora_out.transpose(1, 2)

        return x + lora_out

class LowComponentProcessor(nn.Module):
    def __init__(self, dim, rank=16, alpha=1.0):
        super().__init__()

        self.disentangle = LoRA_Disentanglement(dim, rank, alpha)

        self.cross_fusion = nn.Sequential(

            nn.Linear(dim * 2, dim, bias=False),

        )
        
    def forward(self, rgb_low, ir_low_ref):

        sem_rgb, dom_rgb, fused_rgb = self.disentangle(rgb_low)

        sem_ir, dom_ir, fused_ir = self.disentangle(ir_low_ref)

        cross_concat = torch.cat([sem_rgb, dom_ir], dim=1)

        cross_concat = cross_concat.transpose(1, 2)

        adapted_low = self.cross_fusion(cross_concat)

        adapted_low = adapted_low.transpose(1, 2)

        mid_output = {'sem_rgb': sem_rgb, 'dom_rgb': dom_rgb, 'fused_rgb': fused_rgb, 'ori_rgb': rgb_low,
                        'sem_ir': sem_ir, 'dom_ir': dom_ir, 'fused_ir': fused_ir, 'ori_ir': ir_low_ref, 'adapted_low': adapted_low}

        return adapted_low, mid_output

class MidComponentProcessor(nn.Module):
    def __init__(self, noise_strength=0.1, learnable_strength=True, use_channel_correlation=True):
        super().__init__()
        self.use_channel_correlation = use_channel_correlation
        
        if learnable_strength:

            self.noise_strength = nn.Parameter(
                torch.ones(1) * noise_strength
            )
        else:
            self.register_buffer('noise_strength', 
                               torch.ones(1) * noise_strength)
    
    def compute_batch_statistics(self, x):
        B, C, L = x.shape

        x_flat = x.permute(0, 2, 1).reshape(-1, C)

        mean = x_flat.mean(dim=0, keepdim=True).view(1, C, 1)
        std = x_flat.std(dim=0, keepdim=True).view(1, C, 1) + 1e-8

        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
        
        return mean, std, cov
    
    def generate_correlated_noise(self, ref_band, target_shape):
        B, C, L = target_shape

        base_noise = torch.randn(B, C, L, device=ref_band.device)
        
        if not self.use_channel_correlation:
            return base_noise

        _, _, ref_cov = self.compute_batch_statistics(ref_band)
        
        try:

            reg_cov = ref_cov + torch.eye(C, device=ref_cov.device) * 1e-4

            L_chol = torch.linalg.cholesky(reg_cov)

            noise_flat = base_noise.permute(0, 2, 1).reshape(-1, C)

            correlated_noise_flat = noise_flat @ L_chol.T

            correlated_noise = correlated_noise_flat.view(B, L, C).permute(0, 2, 1)
            
        except RuntimeError as e:

            print(f"Cholesky分解失败，使用特征值分解。错误: {e}")
            
            eigenvalues, eigenvectors = torch.linalg.eigh(reg_cov)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
            L_chol = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
            
            noise_flat = base_noise.permute(0, 2, 1).reshape(-1, C)
            correlated_noise_flat = noise_flat @ L_chol.T
            correlated_noise = correlated_noise_flat.view(B, L, C).permute(0, 2, 1)
        
        return correlated_noise
    
    def forward(self, feats_band, ref_band):
        assert feats_band.dim() == 3, f"特征维度应为3，当前为 {feats_band.dim()}"
        assert ref_band.dim() == 3, f"参考维度应为3，当前为 {ref_band.dim()}"
        
        B, C, L = feats_band.shape

        ref_mean, ref_std, _ = self.compute_batch_statistics(ref_band)

        correlated_noise = self.generate_correlated_noise(ref_band, feats_band.shape)

        noise_mean = correlated_noise.mean(dim=2, keepdim=True)
        noise_std = correlated_noise.std(dim=2, keepdim=True) + 1e-8
        normalized_noise = (correlated_noise - noise_mean) / noise_std

        final_noise = normalized_noise * ref_std * self.noise_strength

        feats_band_processed = feats_band + final_noise
        
        return feats_band_processed

class HighComponentFusion(nn.Module):
    def __init__(self, channels, init_alpha=0.0, use_gate=True):
        super().__init__()
        self.channels = channels
        self.use_gate = use_gate
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * init_alpha)

        if use_gate:
            self.gate = GateModule(channels)
    
    def forward(self, feats, ref):
        B, C, L = feats.shape
        assert ref.shape == feats.shape, \
            f"特征和参考的形状不匹配: {feats.shape} vs {ref.shape}"

        alpha = torch.sigmoid(self.alpha)
        fused = (1 - alpha) * feats + alpha * ref

        if self.use_gate:
            fused = self.gate(fused, feats)
        
        return fused

class GateModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused, original):
        concat = torch.cat([fused, original], dim=1)
        gate = self.gate_conv(concat)
        gated = gate * fused + (1 - gate) * original
        return gated

class SpectrumCore(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        Q: int = 8,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.Q = Q
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

        self.mlp_token2feat_low = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_token2feat_mid = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_token2feat_high = nn.Linear(self.embed_dims, self.embed_dims)

        self.mlp_delta_f_low = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f_mid = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f_high = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)

        nn.init.kaiming_uniform_(self.mlp_delta_f_low.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_delta_f_mid.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_delta_f_high.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.mlp_token2feat_low.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat_mid.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat_high.weight, a=math.sqrt(5))

        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

        self.low_component_func = LowComponentProcessor(dim=self.embed_dims, rank=16)
        self.mid_component_func = MidComponentProcessor(noise_strength=0.1,learnable_strength=True, use_channel_correlation=True)
        self.high_component_func = HighComponentFusion(channels=self.embed_dims, init_alpha=0.0)

        self.fusion_linear = nn.Conv1d(3 * self.embed_dims, self.embed_dims, kernel_size=1)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:

            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def create_frequency_masks(self, H, W, device):
        cy, cx = H // 2, W // 2

        y = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x = torch.arange(W, device=device).view(1, -1).expand(H, W)
        dist = ((y - cy) ** 2 + (x - cx) ** 2).sqrt()

        n = min(H, W)
        max_radius = n / 2

        r1 = max_radius / 8
        r2 = 7 * max_radius / 8

        masks = []

        mask_low = (dist < r1).float()
        masks.append(mask_low)

        mask_mid = ((dist >= r1) & (dist < r2)).float()
        masks.append(mask_mid)

        mask_high = (dist >= r2).float()
        masks.append(mask_high)
        
        return masks

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int, mlp_token2feat: nn.Module, mlp_delta_f: nn.Module,) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            mlp_token2feat(tokens[1:, :]),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f

class LoRASpectrumCore(SpectrumCore):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()

        del self.learnable_tokens

        self.learnable_tokens_low_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_low_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        self.learnable_tokens_mid_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_mid_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        self.learnable_tokens_high_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_high_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )

        nn.init.uniform_(self.learnable_tokens_low_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_low_b.data, -val, val)

        nn.init.uniform_(self.learnable_tokens_mid_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_mid_b.data, -val, val)

        nn.init.uniform_(self.learnable_tokens_high_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_high_b.data, -val, val)

    def get_tokens_low(self, layer):
        if layer == -1:
            return self.learnable_tokens_low_a @ self.learnable_tokens_low_b
        else:
            return self.learnable_tokens_low_a[layer] @ self.learnable_tokens_low_b[layer]
    
    def get_tokens_mid(self, layer):
        if layer == -1:
            return self.learnable_tokens_mid_a @ self.learnable_tokens_mid_b
        else:
            return self.learnable_tokens_mid_a[layer] @ self.learnable_tokens_mid_b[layer]
    
    def get_tokens_high(self, layer):
        if layer == -1:
            return self.learnable_tokens_high_a @ self.learnable_tokens_high_b
        else:
            return self.learnable_tokens_high_a[layer] @ self.learnable_tokens_high_b[layer]
    
    def get_tokens(self, layer, component='low'):
        if component == 'low':
            return self.get_tokens_low(layer)
        elif component == 'mid':
            return self.get_tokens_mid(layer)
        elif component == 'high':
            return self.get_tokens_high(layer)
        else:
            raise ValueError(f"未知的频段类型: {component}，应为 'low', 'mid', 或 'high'")
