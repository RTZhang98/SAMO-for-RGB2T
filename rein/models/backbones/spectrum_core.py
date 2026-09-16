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
        # 两个独立的LoRA分支
        self.semantic_lora = LoRA_Efficient(dim, rank, alpha)
        self.domain_lora = LoRA_Efficient(dim, rank, alpha)
        
        # 融合模块 - 适配序列输入
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, L] - 低频特征序列
        Returns:
            sem: [B, C, L] - 语义特征
            dom: [B, C, L] - 域特征
            fused: [B, C, L] - 融合特征
        """
        # 解耦
        sem = self.semantic_lora(x)  # [B, C, L]
        dom = self.domain_lora(x)    # [B, C, L]
        
        # 融合：在channel维度concat
        # [B, C, L] + [B, C, L] -> [B, 2C, L]
        concat_feat = torch.cat([sem, dom], dim=1)  # [B, 2C, L]
        
        # Transpose for Linear: [B, 2C, L] -> [B, L, 2C]
        concat_feat = concat_feat.transpose(1, 2)
        
        # Fusion
        fused = self.fusion(concat_feat)  # [B, L, C]
        
        # Transpose back: [B, L, C] -> [B, C, L]
        fused = fused.transpose(1, 2)  # [B, C, L]
        
        return sem, dom, fused

class LoRA_Efficient(nn.Module):
    """更高效的LoRA实现 - 直接在序列维度操作"""
    def __init__(self, dim, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank分解
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        """
        Args:
            x: [B, C, L]
        Returns:
            out: [B, C, L]
        """
        # 直接在最后一个维度操作
        # x: [B, C, L] -> transpose -> [B, L, C]
        x_t = x.transpose(1, 2)  # [B, L, C]
        
        # LoRA transformation
        lora_out = self.lora_down(x_t)  # [B, L, rank]
        lora_out = self.lora_up(lora_out)  # [B, L, C]
        lora_out = lora_out * self.scaling
        
        # Transpose back: [B, L, C] -> [B, C, L]
        lora_out = lora_out.transpose(1, 2)  # [B, C, L]
        
        # 残差连接
        return x + lora_out

class LowComponentProcessor(nn.Module):
    def __init__(self, dim, rank=16, alpha=1.0):
        super().__init__()
        # 解耦模块
        self.disentangle = LoRA_Disentanglement(dim, rank, alpha)
        
        # 交叉融合模块：专门用于融合 semantic_RGB + domain_IR
        self.cross_fusion = nn.Sequential(
            # nn.Linear(dim * 2, dim * 2),
            # nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(dim * 2, dim, bias=False),
            #nn.LayerNorm(dim)
        )
        
    def forward(self, rgb_low, ir_low_ref):
        """
        Args:
            rgb_low: [B, C, L] - RGB低频成分
            ir_low_ref: [B, C, L] - IR参考低频成分
        
        Returns:
            adapted_low: [B, C, L] - 融合后的低频成分 (semantic_RGB + domain_IR)
        """
        # 1. 解耦RGB：提取语义和域
        sem_rgb, dom_rgb, fused_rgb = self.disentangle(rgb_low)  # 各为 [B, C, L]
        
        # 2. 解耦IR reference：提取语义和域
        sem_ir, dom_ir, fused_ir = self.disentangle(ir_low_ref)  # 各为 [B, C, L]
        
        # 3. 交叉融合：RGB的语义 + IR的域
        # Concat: [B, C, L] + [B, C, L] -> [B, 2C, L]
        cross_concat = torch.cat([sem_rgb, dom_ir], dim=1)  # [B, 2C, L]
        
        # Transpose for Linear: [B, 2C, L] -> [B, L, 2C]
        cross_concat = cross_concat.transpose(1, 2)  # [B, L, 2C]
        
        # 交叉融合
        adapted_low = self.cross_fusion(cross_concat)  # [B, L, C]
        
        # Transpose back: [B, L, C] -> [B, C, L]
        adapted_low = adapted_low.transpose(1, 2)  # [B, C, L]

        mid_output = {'sem_rgb': sem_rgb, 'dom_rgb': dom_rgb, 'fused_rgb': fused_rgb, 'ori_rgb': rgb_low,
                        'sem_ir': sem_ir, 'dom_ir': dom_ir, 'fused_ir': fused_ir, 'ori_ir': ir_low_ref, 'adapted_low': adapted_low}

        return adapted_low, mid_output

class MidComponentProcessor(nn.Module):
    """
    高级中频段处理模块：适用于 (B, C, L) 形状的特征
    其中 L = H * W
    """
    def __init__(self, noise_strength=0.1, learnable_strength=True, use_channel_correlation=True):
        """
        Args:
            num_bands: 频带数量
            noise_strength: 初始噪声强度
            learnable_strength: 是否使用可学习的噪声强度
            use_channel_correlation: 是否使用通道间相关性
        """
        super().__init__()
        self.use_channel_correlation = use_channel_correlation
        
        if learnable_strength:
            # 为每个频带学习独立的噪声强度
            self.noise_strength = nn.Parameter(
                torch.ones(1) * noise_strength
            )
        else:
            self.register_buffer('noise_strength', 
                               torch.ones(1) * noise_strength)
    
    def compute_batch_statistics(self, x):
        """
        计算整个batch的统计信息
        Args:
            x: (B, C, L) 的特征图，其中 L = H * W
        Returns:
            mean: (1, C, 1) 批次均值
            std: (1, C, 1) 批次标准差
            cov: (C, C) 协方差矩阵
        """
        B, C, L = x.shape
        
        # 重塑为 (B*L, C) 来计算统计量
        x_flat = x.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)
        
        # 计算均值和标准差（跨batch和空间维度）
        mean = x_flat.mean(dim=0, keepdim=True).view(1, C, 1)  # (1, C, 1)
        std = x_flat.std(dim=0, keepdim=True).view(1, C, 1) + 1e-8  # (1, C, 1)
        
        # 计算协方差矩阵（用于通道间相关性）
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)  # (B*L, C)
        cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)  # (C, C)
        
        return mean, std, cov
    
    def generate_correlated_noise(self, ref_band, target_shape):
        """
        生成具有通道相关性的噪声
        Args:
            ref_band: (B, C, L) 参考频带
            target_shape: (B, C, L) 目标形状
            band_idx: 当前频带索引
        Returns:
            noise: (B, C, L) 相关噪声
        """
        B, C, L = target_shape
        
        # 生成基础高斯噪声
        base_noise = torch.randn(B, C, L, device=ref_band.device)  # (B, C, L)
        
        if not self.use_channel_correlation:
            return base_noise
        
        # 计算参考的协方差矩阵
        _, _, ref_cov = self.compute_batch_statistics(ref_band)
        
        try:
            # 正则化协方差矩阵以确保正定性
            reg_cov = ref_cov + torch.eye(C, device=ref_cov.device) * 1e-4
            
            # Cholesky分解
            L_chol = torch.linalg.cholesky(reg_cov)  # (C, C)
            
            # 重塑噪声: (B, C, L) -> (B*L, C)
            noise_flat = base_noise.permute(0, 2, 1).reshape(-1, C)  # (B*L, C)
            
            # 应用相关性变换: (B*L, C) @ (C, C) -> (B*L, C)
            correlated_noise_flat = noise_flat @ L_chol.T  # (B*L, C)
            
            # 重塑回原始形状: (B*L, C) -> (B, L, C) -> (B, C, L)
            correlated_noise = correlated_noise_flat.view(B, L, C).permute(0, 2, 1)
            
        except RuntimeError as e:
            # 如果Cholesky分解失败，使用特征值分解
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
        
        # 计算参考频带的批次统计信息
        ref_mean, ref_std, _ = self.compute_batch_statistics(ref_band)
        
        # 生成具有通道间相关性的噪声
        correlated_noise = self.generate_correlated_noise(ref_band, feats_band.shape)
        
        # 归一化噪声（在空间维度L上计算统计量）
        noise_mean = correlated_noise.mean(dim=2, keepdim=True)  # (B, C, 1)
        noise_std = correlated_noise.std(dim=2, keepdim=True) + 1e-8  # (B, C, 1)
        normalized_noise = (correlated_noise - noise_mean) / noise_std
        
        # 应用参考的统计特性和可学习的强度
        # ref_std shape: (1, C, 1)
        # normalized_noise shape: (B, C, L)
        # 广播后得到 (B, C, L)
        final_noise = normalized_noise * ref_std * self.noise_strength
        
        # 添加噪声到特征
        feats_band_processed = feats_band + final_noise
        
        return feats_band_processed

class HighComponentFusion(nn.Module):
    """
    高频段融合模块：通过可学习强度因子将参考特征注入到目标特征中
    输入形状: (B, C, L)，其中 L = H * W
    """
    def __init__(self, channels, init_alpha=0.0, use_gate=True):
        """
        Args:
            channels: 通道数 C
            fusion_mode: 融合模式
                - 'simple': 简单线性融合
                - 'adaptive': 自适应通道级融合
                - 'spatial': 空间自适应融合
                - 'full': 完全自适应融合（通道+空间）
            init_alpha: 初始融合强度
            use_attention: 是否使用注意力机制
            use_gate: 是否使用门控机制
        """
        super().__init__()
        self.channels = channels
        self.use_gate = use_gate
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * init_alpha)
        
        # 可选的门控机制
        if use_gate:
            self.gate = GateModule(channels)
    
    def forward(self, feats, ref):
        """
        Args:
            feats: 目标特征 (B, C, L)
            ref: 参考特征 (B, C, L)
        Returns:
            fused: 融合后的特征 (B, C, L)
        """
        B, C, L = feats.shape
        assert ref.shape == feats.shape, \
            f"特征和参考的形状不匹配: {feats.shape} vs {ref.shape}"
        
        # alpha: (1, C, 1) -> 广播到 (B, C, L)
        alpha = torch.sigmoid(self.alpha)  # 限制在 [0, 1]
        fused = (1 - alpha) * feats + alpha * ref
        
        # 应用门控机制（可选）
        if self.use_gate:
            fused = self.gate(fused, feats)
        
        return fused

class GateModule(nn.Module):
    """门控模块：动态控制融合特征的保留程度"""
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused, original):
        """
        Args:
            fused: 融合后的特征 (B, C, L)
            original: 原始特征 (B, C, L)
        Returns:
            gated: 门控后的特征 (B, C, L)
        """
        concat = torch.cat([fused, original], dim=1)  # (B, 2*C, L)
        gate = self.gate_conv(concat)  # (B, C, L)
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
        # self.freq_weights = nn.Parameter(torch.ones(3) / 3)
        # self.freq_weights = nn.Parameter(torch.ones(3, 1024) / 3)

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
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def create_frequency_masks(self, H, W, device):
        """
        创建3个频率带的mask
        - 第1个频带(i=0): 低频圆形区域
        - 第2个频带(i=1): 中频圆环 (合并原来的2-7频带)
        - 第3个频带(i=2): 高频外围区域
        
        参数:
            H, W: 特征图的高度和宽度
            device: 设备
        
        返回:
            masks: 列表，包含3个mask，每个mask对应一个频率带
        """
        cy, cx = H // 2, W // 2  # 中心点
        
        # 计算每个点到中心的距离
        y = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x = torch.arange(W, device=device).view(1, -1).expand(H, W)
        dist = ((y - cy) ** 2 + (x - cx) ** 2).sqrt()
        
        # 计算最大半径
        n = min(H, W)
        max_radius = n / 2
        
        # 定义三个频段的分界半径
        # 低频: [0, r1)
        # 中频: [r1, r2)
        # 高频: [r2, max_radius]
        r1 = max_radius / 8  # 原来第1个频带的上界 (1/8)
        r2 = 7 * max_radius / 8  # 原来第7个频带的上界 (7/8)
        
        # 创建3个频率带的mask
        masks = []
        
        # 第1个频带：低频圆形区域 (dist < r1)
        mask_low = (dist < r1).float()
        masks.append(mask_low)
        
        # 第2个频带：中频圆环 (r1 <= dist < r2)
        mask_mid = ((dist >= r1) & (dist < r2)).float()
        masks.append(mask_mid)
        
        # 第3个频带：高频外围区域 (dist >= r2)
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
        
        # 删除原始的 learnable_tokens
        del self.learnable_tokens
        
        # 创建三组 LoRA tokens: low, mid, high
        # Low frequency components
        self.learnable_tokens_low_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_low_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        
        # Mid frequency components
        self.learnable_tokens_mid_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_mid_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        
        # High frequency components
        self.learnable_tokens_high_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_high_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        
        # 计算初始化范围
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        
        # 初始化所有 LoRA 参数
        # Low frequency
        nn.init.uniform_(self.learnable_tokens_low_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_low_b.data, -val, val)
        
        # Mid frequency
        nn.init.uniform_(self.learnable_tokens_mid_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_mid_b.data, -val, val)
        
        # High frequency
        nn.init.uniform_(self.learnable_tokens_high_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_high_b.data, -val, val)

    def get_tokens_low(self, layer):
        """
        获取低频 tokens
        
        Args:
            layer: 层索引，-1 表示所有层
        
        Returns:
            低频 tokens，形状为 [num_layers, token_length, embed_dims] 或 [token_length, embed_dims]
        """
        if layer == -1:
            return self.learnable_tokens_low_a @ self.learnable_tokens_low_b
        else:
            return self.learnable_tokens_low_a[layer] @ self.learnable_tokens_low_b[layer]
    
    def get_tokens_mid(self, layer):
        """
        获取中频 tokens
        
        Args:
            layer: 层索引，-1 表示所有层
        
        Returns:
            中频 tokens，形状为 [num_layers, token_length, embed_dims] 或 [token_length, embed_dims]
        """
        if layer == -1:
            return self.learnable_tokens_mid_a @ self.learnable_tokens_mid_b
        else:
            return self.learnable_tokens_mid_a[layer] @ self.learnable_tokens_mid_b[layer]
    
    def get_tokens_high(self, layer):
        """
        获取高频 tokens
        
        Args:
            layer: 层索引，-1 表示所有层
        
        Returns:
            高频 tokens，形状为 [num_layers, token_length, embed_dims] 或 [token_length, embed_dims]
        """
        if layer == -1:
            return self.learnable_tokens_high_a @ self.learnable_tokens_high_b
        else:
            return self.learnable_tokens_high_a[layer] @ self.learnable_tokens_high_b[layer]
    
    def get_tokens(self, layer, component='low'):
        """
        通用接口获取指定频段的 tokens
        
        Args:
            layer: 层索引，-1 表示所有层
            component: 频段类型，可选 'low', 'mid', 'high'
        
        Returns:
            指定频段的 tokens
        """
        if component == 'low':
            return self.get_tokens_low(layer)
        elif component == 'mid':
            return self.get_tokens_mid(layer)
        elif component == 'high':
            return self.get_tokens_high(layer)
        else:
            raise ValueError(f"未知的频段类型: {component}，应为 'low', 'mid', 或 'high'")
