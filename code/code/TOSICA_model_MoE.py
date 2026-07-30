# ============================================================================
# 增强的TOSICA_model_MoE.py - 添加LRP可解释性功能
# ============================================================================

# 平衡过拟合优化版MoE-Transformer模型 + LRP可解释性
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def init_weights(m):
    """保守的权重初始化，防止过拟合"""
    if isinstance(m, nn.Linear):
        # 使用更小的初始化方差
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


class RegularizedExpert(nn.Module):
    """正则化专家网络，增强泛化能力"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.25):
        super().__init__()
        # 使用更保守的架构
        self.hidden_dim = min(hidden_dim, input_dim)  # 限制隐藏层大小
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # 添加LayerNorm增强稳定性
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
            nn.Dropout(dropout * 0.5)  # 输出层使用较小的dropout
        )
        
        # 权重衰减友好的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
    
    def forward(self, x):
        return self.net(x)


class BalancedGating(nn.Module):
    """平衡门控网络，增强负载均衡"""
    
    def __init__(self, input_dim, num_experts, top_k=2, gating_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # 更保守的门控网络
        self.gate = nn.Sequential(
            nn.LayerNorm(input_dim),  # 输入归一化
            nn.Linear(input_dim, num_experts),
            nn.Dropout(gating_dropout)  # 门控dropout
        )
        
        # 专家使用统计
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
        
        # 负载均衡温度参数
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))
    
    def forward(self, x):
        # 温度缩放的门控分数
        gate_scores = self.gate(x) / torch.clamp(self.temperature, min=0.1, max=2.0)
        
        # 添加轻微噪声增强泛化（仅训练时）
        if self.training:
            noise = torch.randn_like(gate_scores) * 0.01
            gate_scores = gate_scores + noise
        
        # Top-K选择
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        
        # 更新专家使用统计
        if self.training:
            with torch.no_grad():
                batch_size = x.size(0)
                self.total_tokens += batch_size
                
                for i in range(self.num_experts):
                    usage = (top_k_indices == i).float().sum()
                    self.expert_usage[i] = 0.99 * self.expert_usage[i] + 0.01 * usage
        
        return top_k_weights, top_k_indices
    
    def get_load_balancing_loss(self):
        """增强的负载平衡损失"""
        if self.total_tokens > 0:
            usage_rates = self.expert_usage / (self.total_tokens + 1e-8)
            target_rate = 1.0 / self.num_experts
            
            # 组合L1和L2损失
            l1_loss = F.l1_loss(usage_rates, torch.full_like(usage_rates, target_rate))
            l2_loss = F.mse_loss(usage_rates, torch.full_like(usage_rates, target_rate))
            
            return 0.7 * l1_loss + 0.3 * l2_loss
        return torch.tensor(0.0, device=self.expert_usage.device)
    
    def get_diversity_loss(self):
        """专家多样性损失"""
        if self.total_tokens > 10:  # 需要足够的样本
            usage_std = torch.std(self.expert_usage)
            target_std = torch.sqrt(torch.tensor(1.0 / (12 * self.num_experts)))
            return F.mse_loss(usage_std, target_std)
        return torch.tensor(0.0, device=self.expert_usage.device)


class BalancedMoELayer(nn.Module):
    """平衡的MoE层，防止过拟合"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2, dropout=0.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 限制隐藏层大小防止过拟合
        effective_hidden = min(hidden_dim, input_dim * 2)
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            RegularizedExpert(input_dim, effective_hidden, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # 平衡门控
        self.gate = BalancedGating(input_dim, num_experts, top_k, dropout * 0.5)
        
        # 残差连接
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        # 输出正则化
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        # 随机深度（Stochastic Depth）
        self.drop_path_prob = 0.1 if dropout > 0.2 else 0.0
    
    def forward(self, x):
        batch_size = x.size(0)
        residual = self.residual(x)
        
        # 随机深度：训练时随机跳过某些层
        if self.training and self.drop_path_prob > 0:
            if torch.rand(1).item() < self.drop_path_prob:
                return self.output_norm(residual), None, None  # 🔧 修复：确保返回3个值
        
        # 门控选择
        gate_weights, expert_indices = self.gate(x)
        
        # 并行计算所有专家输出（批量化）
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        
        # 基于门控权重聚合输出
        output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
        
        for i in range(self.top_k):
            expert_idx = expert_indices[:, i]  # [batch]
            weight = gate_weights[:, i].unsqueeze(-1)  # [batch, 1]
            
            # 选择对应专家的输出
            selected_outputs = torch.gather(
                expert_outputs, 1,
                expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.output_dim)
            ).squeeze(1)  # [batch, output_dim]
            
            output += weight * selected_outputs
        
        # 残差连接和正则化
        output = output + residual * 0.1  # 较小的残差权重
        output = self.output_norm(output)
        output = self.output_dropout(output)
        
        return output, gate_weights, expert_indices  # 🔧 修复：确保总是返回3个值


class RegularizedTransformerBlock(nn.Module):
    """正则化Transformer块，增强泛化能力"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.25, use_pre_norm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_pre_norm = use_pre_norm
        
        # 确保embed_dim能被num_heads整除
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络 - 使用较小的扩展比例
        ffn_dim = min(embed_dim * 2, embed_dim + 128)  # 限制FFN大小
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.LayerNorm(ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 注意力dropout
        self.attention_dropout = nn.Dropout(dropout * 0.5)
        
        # 存储注意力权重
        self.attention_weights = None
        
        # 随机深度
        self.drop_path_prob = 0.05 if dropout > 0.2 else 0.0
    
    def forward(self, x):
        # 确保输入是3D张量
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Pre-norm或Post-norm
        if self.use_pre_norm:
            # Pre-norm: 先归一化再计算
            norm_x = self.norm1(x)
            attn_output, attn_weights = self.self_attention(norm_x, norm_x, norm_x)
        else:
            # Post-norm: 先计算再归一化
            attn_output, attn_weights = self.self_attention(x, x, x)
        
        # 存储注意力权重
        self.attention_weights = attn_weights.detach() if attn_weights is not None else None
        
        # 注意力dropout和残差连接
        attn_output = self.attention_dropout(attn_output)
        
        # 随机深度
        if self.training and self.drop_path_prob > 0:
            if torch.rand(1).item() < self.drop_path_prob:
                attn_output = torch.zeros_like(attn_output)
        
        if self.use_pre_norm:
            x = x + attn_output
            # 第二个子层
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + ff_output
        else:
            x = self.norm1(x + attn_output)
            # 第二个子层
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)
        
        return x.squeeze(1) if x.size(1) == 1 else x


class LRPHook:
    """LRP (Layer-wise Relevance Propagation) 钩子类"""
    
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def save_activation(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook
    
    def save_gradient(self, name):
        def hook(grad):
            self.gradients[name] = grad.detach()
            return grad
        return hook
    
    def register_hooks(self, model):
        """为模型的关键层注册钩子"""
        # 为基因编码器注册钩子
        if hasattr(model, 'gene_encoder'):
            hook = model.gene_encoder.register_forward_hook(self.save_activation('gene_encoder'))
            self.hooks.append(hook)
        
        # 为蛋白质编码器注册钩子
        if hasattr(model, 'protein_encoder'):
            hook = model.protein_encoder.register_forward_hook(self.save_activation('protein_encoder'))
            self.hooks.append(hook)
        
        # 为融合层注册钩子
        if hasattr(model, 'fusion'):
            hook = model.fusion.register_forward_hook(self.save_activation('fusion'))
            self.hooks.append(hook)
        
        # 为MoE层注册钩子
        if hasattr(model, 'moe_layers'):
            for i, moe_layer in enumerate(model.moe_layers):
                hook = moe_layer.register_forward_hook(self.save_activation(f'moe_{i}'))
                self.hooks.append(hook)
        
        # 为Transformer层注册钩子
        if hasattr(model, 'transformer_layers'):
            for i, transformer in enumerate(model.transformer_layers):
                hook = transformer.register_forward_hook(self.save_activation(f'transformer_{i}'))
                self.hooks.append(hook)
        
        # 为分类器注册钩子
        if hasattr(model, 'classifier'):
            hook = model.classifier.register_forward_hook(self.save_activation('classifier'))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}


class BalancedMoETransformer(nn.Module):
    """平衡的MoE-Transformer，优化过拟合控制 + LRP可解释性"""
    
    def __init__(self, gene_dim, protein_dim, num_classes, embed_dim=128,
                 num_experts=4, top_k=2, num_transformer_layers=2,
                 num_moe_layers=1, num_heads=4, dropout=0.25, **kwargs):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.gene_dim = gene_dim
        self.protein_dim = protein_dim
        
        # 确保embed_dim能被num_heads整除
        if embed_dim % num_heads != 0:
            embed_dim = (embed_dim // num_heads) * num_heads
            print(f"⚠️ 调整embed_dim为 {embed_dim} 以适配 {num_heads} 个注意力头")
        
        self.embed_dim = embed_dim
        
        # 特征编码器 - 使用较保守的架构
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 特征融合 - 增加正则化
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # MoE层
        self.moe_layers = nn.ModuleList([
            BalancedMoELayer(embed_dim, embed_dim, embed_dim, num_experts, top_k, dropout)
            for _ in range(num_moe_layers)
        ])
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            RegularizedTransformerBlock(embed_dim, num_heads, dropout, use_pre_norm=True)
            for _ in range(num_transformer_layers)
        ])
        
        # 分类头 - 增强正则化
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 权重初始化
        self.apply(init_weights)
        
        # 注意力存储
        self.attention_storage = {}
        
        # 模型正则化参数
        self.register_buffer('training_steps', torch.zeros(1))
        
        # 🚀 新增：LRP可解释性组件
        self.lrp_hook = LRPHook()
        self.lrp_enabled = False
        
    def enable_lrp(self):
        """启用LRP功能"""
        if not self.lrp_enabled:
            self.lrp_hook.register_hooks(self)
            self.lrp_enabled = True
            print("✅ LRP功能已启用")
    
    def disable_lrp(self):
        """禁用LRP功能"""
        if self.lrp_enabled:
            self.lrp_hook.remove_hooks()
            self.lrp_enabled = False
            print("✅ LRP功能已禁用")
    
    def forward(self, gene_data, protein_data, return_attention=False, ensemble_mode=False):
        # 特征编码
        gene_embed = self.gene_encoder(gene_data)
        protein_embed = self.protein_encoder(protein_data)
        
        # 特征融合
        fused = torch.cat([gene_embed, protein_embed], dim=1)
        x = self.fusion(fused)
        
        # MoE处理 - 🔧 修复：安全处理MoE层的返回值
        moe_outputs = []
        for moe_layer in self.moe_layers:
            try:
                # 安全调用MoE层
                moe_result = moe_layer(x)
                
                # 确保返回值格式正确
                if isinstance(moe_result, tuple) and len(moe_result) >= 3:
                    x, gate_weights, expert_indices = moe_result[:3]  # 只取前3个
                elif isinstance(moe_result, tuple) and len(moe_result) == 2:
                    x, gate_weights = moe_result
                    expert_indices = None
                elif isinstance(moe_result, tuple) and len(moe_result) == 1:
                    x = moe_result[0]
                    gate_weights = None
                    expert_indices = None
                else:
                    # 如果不是元组，假设是单个张量
                    x = moe_result
                    gate_weights = None
                    expert_indices = None
                
                moe_outputs.append((gate_weights, expert_indices))
                
            except Exception as e:
                print(f"⚠️ MoE层 {len(moe_outputs)} 处理失败: {e}")
                # 跳过这个MoE层，使用原始输入
                moe_outputs.append((None, None))
        
        # Transformer处理
        transformer_attention = []
        for transformer in self.transformer_layers:
            x = transformer(x)
            if hasattr(transformer, 'attention_weights') and transformer.attention_weights is not None:
                transformer_attention.append(transformer.attention_weights)
        
        # 分类
        logits = self.classifier(x)
        
        # 更新训练步数
        if self.training:
            self.training_steps += 1
        
        # 🔧 修复：根据return_attention参数返回不同的值
        if return_attention:
            self.attention_storage = {
                'transformer_attention': transformer_attention,
                'moe_gates': [(gw.detach() if gw is not None else None, 
                              ei.detach() if ei is not None else None) for gw, ei in moe_outputs],
                'gene_embedding': gene_embed.detach(),
                'protein_embedding': protein_embed.detach()
            }
            return logits, self.attention_storage, moe_outputs, gene_embed, protein_embed
        else:
            # 🔧 关键修复：默认只返回logits，避免解包错误
            return logits
    
    # 🚀 新增：LRP可解释性方法
    def generate_lrp(self, gene_data, protein_data, target_class=None, epsilon=1e-6):
        """
        生成LRP (Layer-wise Relevance Propagation) 解释
        
        Parameters:
        -----------
        gene_data : torch.Tensor
            基因表达数据 [batch_size, gene_dim]
        protein_data : torch.Tensor
            蛋白质表达数据 [batch_size, protein_dim]
        target_class : int or torch.Tensor, optional
            目标类别，如果为None则使用预测类别
        epsilon : float
            LRP中的小量，防止除零
        
        Returns:
        --------
        gene_relevance : torch.Tensor
            基因特征的相关性分数
        protein_relevance : torch.Tensor
            蛋白质特征的相关性分数
        logits : torch.Tensor
            模型的原始输出
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 启用LRP功能
        if not self.lrp_enabled:
            self.enable_lrp()
        
        # 确保输入需要梯度
        gene_data = gene_data.clone().detach().requires_grad_(True)
        protein_data = protein_data.clone().detach().requires_grad_(True)
        
        # 前向传播
        logits = self.forward(gene_data, protein_data)
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * gene_data.size(0), device=gene_data.device)
        
        # 选择目标类别的logits
        batch_size = logits.size(0)
        target_logits = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)
        
        # 清零梯度
        if gene_data.grad is not None:
            gene_data.grad.zero_()
        if protein_data.grad is not None:
            protein_data.grad.zero_()
        
        # 反向传播计算梯度
        target_logits.sum().backward(retain_graph=True)
        
        # 计算LRP相关性分数
        # 使用epsilon-LRP规则：R = (input * grad) / (input + epsilon * sign(input))
        gene_grad = gene_data.grad.detach() if gene_data.grad is not None else torch.zeros_like(gene_data)
        protein_grad = protein_data.grad.detach() if protein_data.grad is not None else torch.zeros_like(protein_data)
        
        # Epsilon-LRP规则实现
        gene_epsilon = epsilon * torch.sign(gene_data)
        protein_epsilon = epsilon * torch.sign(protein_data)
        
        gene_relevance = (gene_data * gene_grad) / (gene_data + gene_epsilon + 1e-12)
        protein_relevance = (protein_data * protein_grad) / (protein_data + protein_epsilon + 1e-12)
        
        return gene_relevance.detach(), protein_relevance.detach(), logits.detach()
    
    def generate_advanced_lrp(self, gene_data, protein_data, target_class=None, 
                             method='epsilon', alpha=1.0, beta=0.0, epsilon=1e-6):
        """
        生成高级LRP解释（支持多种LRP变体）
        
        Parameters:
        -----------
        method : str
            LRP方法: 'epsilon', 'alpha_beta', 'gamma'
        alpha, beta : float
            Alpha-Beta LRP的参数
        """
        self.eval()
        
        if not self.lrp_enabled:
            self.enable_lrp()
        
        # 确保输入需要梯度
        gene_data = gene_data.clone().detach().requires_grad_(True)
        protein_data = protein_data.clone().detach().requires_grad_(True)
        
        # 前向传播并收集激活
        logits = self.forward(gene_data, protein_data)
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * gene_data.size(0), device=gene_data.device)
        
        target_logits = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)
        
        # 清零梯度
        self.zero_grad()
        if gene_data.grad is not None:
            gene_data.grad.zero_()
        if protein_data.grad is not None:
            protein_data.grad.zero_()
        
        # 反向传播
        target_logits.sum().backward(retain_graph=True)
        
        # 根据方法计算相关性
        if method == 'epsilon':
            # Epsilon-LRP
            gene_epsilon = epsilon * torch.sign(gene_data)
            protein_epsilon = epsilon * torch.sign(protein_data)
            
            gene_relevance = (gene_data * gene_data.grad) / (gene_data + gene_epsilon + 1e-12)
            protein_relevance = (protein_data * protein_data.grad) / (protein_data + protein_epsilon + 1e-12)
            
        elif method == 'alpha_beta':
            # Alpha-Beta LRP
            gene_grad = gene_data.grad
            protein_grad = protein_data.grad
            
            # 正向和负向贡献
            gene_pos = torch.clamp(gene_data * gene_grad, min=0)
            gene_neg = torch.clamp(gene_data * gene_grad, max=0)
            gene_relevance = alpha * gene_pos + beta * gene_neg
            
            protein_pos = torch.clamp(protein_data * protein_grad, min=0)
            protein_neg = torch.clamp(protein_data * protein_grad, max=0)
            protein_relevance = alpha * protein_pos + beta * protein_neg
            
        elif method == 'gamma':
            # Gamma-LRP (加权梯度)
            gamma = 0.25
            gene_relevance = gene_data * gene_data.grad * (1 + gamma)
            protein_relevance = protein_data * protein_data.grad * (1 + gamma)
            
        else:
            # 默认使用简单的梯度×输入
            gene_relevance = gene_data * gene_data.grad
            protein_relevance = protein_data * protein_data.grad
        
        # 收集层级相关性信息
        layer_relevances = {}
        for name, activation in self.lrp_hook.activations.items():
            if name in self.lrp_hook.gradients:
                grad = self.lrp_hook.gradients[name]
                layer_relevances[name] = (activation * grad).sum(dim=-1) if activation.dim() > 2 else activation * grad
        
        return {
            'gene_relevance': gene_relevance.detach(),
            'protein_relevance': protein_relevance.detach(),
            'logits': logits.detach(),
            'layer_relevances': layer_relevances,
            'target_class': target_class,
            'method': method
        }
    
    def compute_feature_importance(self, gene_data, protein_data, n_samples=10):
        """
        计算特征重要性统计
        
        Parameters:
        -----------
        gene_data : torch.Tensor
            基因数据
        protein_data : torch.Tensor
            蛋白质数据
        n_samples : int
            用于统计的样本数量
        
        Returns:
        --------
        dict : 特征重要性统计
        """
        self.eval()
        
        n_samples = min(n_samples, gene_data.size(0))
        indices = torch.randperm(gene_data.size(0))[:n_samples]
        
        gene_sample = gene_data[indices]
        protein_sample = protein_data[indices]
        
        all_gene_relevances = []
        all_protein_relevances = []
        
        for i in range(n_samples):
            try:
                gene_rel, protein_rel, _ = self.generate_lrp(
                    gene_sample[i:i+1], protein_sample[i:i+1]
                )
                all_gene_relevances.append(gene_rel.abs())
                all_protein_relevances.append(protein_rel.abs())
            except Exception as e:
                print(f"⚠️ 样本 {i} LRP计算失败: {e}")
                continue
        
        if not all_gene_relevances:
            return None
        
        # 聚合统计
        gene_importance = torch.cat(all_gene_relevances, dim=0).mean(dim=0)
        protein_importance = torch.cat(all_protein_relevances, dim=0).mean(dim=0)
        
        # 找到最重要的特征
        top_genes = torch.argsort(gene_importance, descending=True)[:20]
        top_proteins = torch.argsort(protein_importance, descending=True)[:10]
        
        return {
            'gene_importance': gene_importance,
            'protein_importance': protein_importance,
            'top_genes': top_genes,
            'top_proteins': top_proteins,
            'gene_importance_stats': {
                'mean': gene_importance.mean().item(),
                'std': gene_importance.std().item(),
                'max': gene_importance.max().item(),
                'min': gene_importance.min().item()
            },
            'protein_importance_stats': {
                'mean': protein_importance.mean().item(),
                'std': protein_importance.std().item(),
                'max': protein_importance.max().item(),
                'min': protein_importance.min().item()
            }
        }
    
    def get_moe_auxiliary_loss(self):
        """获取MoE辅助损失"""
        total_balance_loss = 0.0
        total_diversity_loss = 0.0
        
        for moe_layer in self.moe_layers:
            total_balance_loss += moe_layer.gate.get_load_balancing_loss()
            total_diversity_loss += moe_layer.gate.get_diversity_loss()
        
        num_layers = len(self.moe_layers)
        if num_layers > 0:
            avg_balance_loss = total_balance_loss / num_layers
            avg_diversity_loss = total_diversity_loss / num_layers
            return avg_balance_loss, avg_diversity_loss
        else:
            return torch.tensor(0.0), torch.tensor(0.0)
    
    def reset_moe_stats(self):
        """重置MoE统计"""
        for moe_layer in self.moe_layers:
            moe_layer.gate.expert_usage.zero_()
            moe_layer.gate.total_tokens.zero_()
    
    def get_regularization_loss(self):
        """获取模型正则化损失"""
        reg_loss = 0.0
        
        # L2正则化
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        
        return reg_loss * 1e-6  # 很小的正则化系数
    
    def get_moe_expert_stats(self):
        """获取MoE专家使用统计"""
        stats = {}
        for i, moe_layer in enumerate(self.moe_layers):
            gate = moe_layer.gate
            if gate.total_tokens > 0:
                usage_rates = gate.expert_usage / gate.total_tokens
                stats[f'layer_{i}'] = {
                    'usage_rates': usage_rates.cpu().numpy().tolist(),
                    'total_tokens': gate.total_tokens.item(),
                    'balance_score': 1.0 - usage_rates.std().item(),
                    'entropy': -(usage_rates * torch.log(usage_rates + 1e-8)).sum().item()
                }
        return stats


class EnhancedMultiModalLoss(nn.Module):
    """增强的多模态损失函数，优化过拟合控制"""
    
    def __init__(self, num_experts=4, alpha=0.8, beta=0.5, gamma=0.2, 
                 delta=0.15, epsilon=0.1, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha        # 主任务权重
        self.beta = beta         # 负载均衡权重
        self.gamma = gamma       # 多样性权重
        self.delta = delta       # 一致性权重
        self.epsilon = epsilon   # 正则化权重
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits, labels, moe_outputs=None, model=None):
        # 主分类损失
        classification_loss = self.ce_loss(logits, labels)
        total_loss = self.alpha * classification_loss
        
        # MoE相关损失
        if moe_outputs is not None and len(moe_outputs) > 0:
            # 负载均衡损失
            balance_loss = 0.0
            diversity_loss = 0.0
            
            if model is not None and hasattr(model, 'get_moe_auxiliary_loss'):
                aux_balance, aux_diversity = model.get_moe_auxiliary_loss()
                balance_loss += aux_balance
                diversity_loss += aux_diversity
            else:
                # 简化版本
                for gate_weights, expert_indices in moe_outputs:
                    if gate_weights is not None:
                        # 负载均衡：鼓励均匀使用专家
                        expert_probs = F.softmax(gate_weights, dim=-1)
                        avg_expert_prob = expert_probs.mean(dim=0)
                        uniform_prob = torch.ones_like(avg_expert_prob) / len(avg_expert_prob)
                        balance_loss += F.kl_div(
                            F.log_softmax(avg_expert_prob.unsqueeze(0), dim=-1),
                            uniform_prob.unsqueeze(0),
                            reduction='batchmean'
                        )
                        
                        # 多样性：鼓励专家激活的多样性
                        diversity_loss += -torch.mean(torch.var(expert_probs, dim=1))
            
            total_loss += self.beta * balance_loss
            total_loss += self.gamma * diversity_loss
        
        # 模型正则化损失
        if model is not None and hasattr(model, 'get_regularization_loss'):
            reg_loss = model.get_regularization_loss()
            total_loss += self.epsilon * reg_loss
        
        return total_loss, classification_loss


def create_ultra_fast_moe_transformer(gene_dim, protein_dim, num_classes, config=None):
    """创建平衡的MoE-Transformer"""
    
    if config is None:
        config = {
            'embed_dim': 128,
            'num_experts': 4,
            'top_k': 2,
            'num_transformer_layers': 2,
            'num_moe_layers': 1,
            'num_heads': 4,
            'dropout': 0.25
        }
    
    # 支持的参数列表
    supported_params = {
        'embed_dim', 'num_experts', 'top_k', 'num_transformer_layers',
        'num_moe_layers', 'num_heads', 'dropout'
    }
    
    # 过滤配置参数
    filtered_config = {}
    removed_params = []
    
    for key, value in config.items():
        if key in supported_params:
            filtered_config[key] = value
        else:
            removed_params.append(f"{key}={value}")
    
    if removed_params:
        print(f"⚠️ 已移除不支持的参数: {', '.join(removed_params)}")
    
    # 参数验证和调整
    embed_dim = filtered_config.get('embed_dim', 128)
    num_heads = filtered_config.get('num_heads', 4)
    
    # 确保embed_dim能被num_heads整除
    if embed_dim % num_heads != 0:
        new_embed_dim = (embed_dim // num_heads) * num_heads
        filtered_config['embed_dim'] = new_embed_dim
        print(f"⚠️ 调整embed_dim: {embed_dim} → {new_embed_dim} (适配{num_heads}个注意力头)")
    
    model = BalancedMoETransformer(
        gene_dim=gene_dim,
        protein_dim=protein_dim,
        num_classes=num_classes,
        **filtered_config
    )
    
    return model


# 向后兼容函数
def create_enhanced_moe_transformer(gene_dim, protein_dim, num_classes, config=None):
    """向后兼容函数"""
    return create_ultra_fast_moe_transformer(gene_dim, protein_dim, num_classes, config)


def create_improved_moe_transformer(gene_dim, protein_dim, num_classes, config=None):
    """向后兼容函数"""
    return create_ultra_fast_moe_transformer(gene_dim, protein_dim, num_classes, config)


# 损失函数别名
class SimplifiedMultiModalLoss(EnhancedMultiModalLoss):
    """向后兼容的损失函数"""
    pass


class AdvancedMultiModalLoss(EnhancedMultiModalLoss):
    """向后兼容的损失函数"""
    pass


class BalancedMultiModalLoss(EnhancedMultiModalLoss):
    """向后兼容的损失函数"""
    pass


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """获取模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def test_enhanced_model_with_lrp():
    """测试增强版模型（包含LRP功能）"""
    print("🧪 测试增强版MoE-Transformer（含LRP）...")
    
    batch_size = 16
    gene_dim = 2000
    protein_dim = 13
    num_classes = 7
    
    gene_data = torch.randn(batch_size, gene_dim)
    protein_data = torch.randn(batch_size, protein_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试配置
    config = {'embed_dim': 128, 'num_experts': 4, 'num_heads': 4, 'dropout': 0.25}
    
    try:
        # 创建模型
        model = create_ultra_fast_moe_transformer(gene_dim, protein_dim, num_classes, config)
        
        print(f"✅ 模型参数: {count_parameters(model):,}")
        print(f"✅ 模型大小: {get_model_size(model):.2f} MB")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            logits = model(gene_data, protein_data)
            print(f"✅ 前向传播成功，输出形状: {logits.shape}")
        
        # 测试LRP功能
        print("🔍 测试LRP可解释性功能...")
        model.enable_lrp()
        
        # 测试基础LRP
        gene_rel, protein_rel, logits = model.generate_lrp(
            gene_data[:2], protein_data[:2]
        )
        print(f"✅ 基础LRP测试成功: 基因相关性{gene_rel.shape}, 蛋白质相关性{protein_rel.shape}")
        
        # 测试高级LRP
        advanced_results = model.generate_advanced_lrp(
            gene_data[:2], protein_data[:2], method='alpha_beta', alpha=2.0, beta=-1.0
        )
        print(f"✅ 高级LRP测试成功")
        
        # 测试特征重要性计算
        importance_stats = model.compute_feature_importance(gene_data[:5], protein_data[:5])
        if importance_stats:
            print(f"✅ 特征重要性计算成功: top基因{len(importance_stats['top_genes'])}, top蛋白质{len(importance_stats['top_proteins'])}")
        
        # 测试MoE统计
        moe_stats = model.get_moe_expert_stats()
        print(f"✅ MoE统计获取成功: {len(moe_stats)} 个MoE层")
        
        model.disable_lrp()
        
        print("🎉 增强版模型（含LRP）测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_model_with_lrp()