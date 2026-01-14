import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import  InputEmbedding
# from .ours_memory_module import MemoryModule

# ours
# 原先的
from .ours_memory_module import MemoryModule


class PatchLocalBranch(nn.Module):
    """
    封装 PatchLocalTransformer，使其在双分支模型中作为 Patch 分支出错。
    注意：由于是单尺度，patch_scales 参数不再相关，我们使用 patch_len。
    """

    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 patch_len, n_memory, shrink_thres, device, phase_type, dataset_name):
        super(PatchLocalBranch, self).__init__()

        self.patch_len = patch_len
        self.patch_scales = [1]
        self.d_model = d_model
        # PatchLocalTransformer 包含了 Embedding, Encoder, Memory, Decoder
        self.plt_model = PatchLocalTransformer(
            win_size=win_size, enc_in=enc_in, c_out=c_out, patch_len=patch_len,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout, activation=activation,
            device=device, n_memory=n_memory, shrink_thres=shrink_thres,
            # Phase type/Dataset name 用于 Memory Module 初始化
            phase_type=phase_type, dataset_name=f"{dataset_name}_patch"
        )

    def forward(self, x):
        # 调用 PatchLocalTransformer 的 forward
        out_dict = self.plt_model(x)

        # 适配 GCN_PAD_Model 所需的输出格式
        return {
            # 重构输出
            'output': out_dict['out'],

            # Patch Queries (单尺度，使用 1 作为尺度键)
            # PatchLocalTransformer 的 queries 是 Token-level (B, N_patch, D)
            'patch_queries': {1: out_dict['queries']},

            # Patch Mems (单尺度)
            'patch_mems': {1: out_dict['mem']},

            # Patch Attns (单尺度)
            'patch_attns': {1: out_dict['attn']},

            # 新增的点级 Queries (用于 test 时的精确距离计算)
            'point_queries': out_dict['point_queries']
        }

    # 用于 K-Means 的查询提取
    def get_all_queries_for_kmeans(self, x):
        # 使用 PatchLocalTransformer 内部的 queries 提取方法
        out_dict = self.plt_model.get_all_queries_for_kmeans(x)

        # 适配多尺度接口的字典结构
        return {'patch_queries': {1: out_dict['queries']}}


class PatchLocalTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, patch_len,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', device=None,
                 n_memory=10, shrink_thres=0.0025, memory_initial=False,
                 memory_init_embeddings=None, phase_type=None, dataset_name=None
                 ):
        super(PatchLocalTransformer, self).__init__()

        self.patch_len = patch_len  # P_len
        self.d_model = d_model
        self.enc_in = enc_in  # C_in
        self.memory_initial = memory_initial
        # === 新增：非线性激活 ===
        self.patch_activation = nn.GELU()  # 或 nn.ReLU()
        self.num_patches = win_size // patch_len
        if win_size % patch_len != 0:
            print("Warning: Window size must be divisible by patch_len for non-overlapping patches.")

        # --- 移除原来的 self.embedding (C_in -> D) ---
        # self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # 1. 核心 Patch Embedding (Flatten Projection)
        # Token 原始输入维度：P_len * C_in
        # self.patch_input_dim = self.patch_len * self.enc_in
        self.patch_input_dim =  self.enc_in

        # # 定义 Tokenization 核心层：将展平后的 Patch (P_len * C_in) 映射到 D 维度
        self.patch_embedding = nn.Linear(self.enc_in, d_model)
        # self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        #隐藏维度 H (通常与 d_model 相同或更大)
        #//todo patchlen 1
        # H_dim = d_model * 8
        #
        # self.patch_embedding = nn.Sequential(
        #     # 第一层：展平输入 -> 隐藏层 H_dim
        #     nn.Linear(self.patch_input_dim, H_dim),
        #     nn.GELU(),  # 非线性激活
        #     nn.Dropout(dropout),
        #
        #     # 第二层：隐藏层 H_dim -> 最终输出 D
        #     nn.Linear(H_dim, d_model),
        #     nn.LayerNorm(d_model)  # 最终归一化，有助于稳定 Encoder 输入
        # )
        # 可选：如果需要位置编码，可以在这里定义
        #//todo patch-len 1
        if not memory_initial and memory_init_embeddings is not None:
            # Note: 这里的 initialize_memories 需要确保它初始化的是 self.mem_module.mem
            self.initialize_memories(memory_init_embeddings)

            # 2. Encoder (序列长度是 num_patches)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(self.num_patches, d_model, n_heads, dropout=dropout),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # 3. Memory Module
        self.mem_module = MemoryModule(
            n_memory=n_memory,
            fea_dim=d_model,
            shrink_thres=shrink_thres,
            device=device,
            memory_init_embedding=memory_init_embeddings if memory_initial else None,  # 传递给 MemoryModule
            phase_type=phase_type,
            dataset_name=dataset_name)

        # 4. Decoder (将 2D 维度映射回 C_out)
        self.weak_decoder = Decoder(d_model * 2, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    # 辅助函数：Patching & Flattening (取代原来的 patchify)
    def patch_and_flatten(self, x):
        B, L, C = x.shape

        # 1. Patching: (B, L, C) -> (B, N_patch, P_len, C)
        x_patched = x.reshape(B, self.num_patches, self.patch_len, C)

        # 2. Flatten P_len 和 C 维度 -> (B * N_patch, P_len * C)
        x_flattened = x_patched.reshape(B * self.num_patches, self.patch_input_dim)

        return x_flattened  # (B*N_patch, P_len * C_in)

    def forward(self, x):
        B, L, C = x.shape

        # 1. Patching & Flattening: (B, L, C) -> (B * N_patch, P_len * C)
        x_flattened = self.patch_and_flatten(x)
        queries_token_flat = self.patch_embedding(x_flattened)


        # 3. Reshape back to Sequence: (B, N_patch, D)
        queries_token = queries_token_flat.view(B, self.num_patches, self.d_model)

        # --- 点级 Queries 的处理 (需要重新定义) ---
        # 此时的 Q_points (用于异常分数计算) 应该基于 Token queries 上采样或投影得到
        # 为了兼容您的后续异常分数计算，我们暂时使用 Token Queries 的平铺作为 Q_points 的基础
        Q_points = queries_token_flat.view(B, self.num_patches, self.d_model)
        # Note: 除非你的异常分数计算依赖于 L长度，否则保留 N_patch 长度的 queries_encoded 即可
        # 如果需要 L长度的点级 Query，需要在 encoder 之后进行上采样 (复杂度增加)
        # 鉴于您的目标是 Patch Transformer，我们应该使用 Queries_encoded (N_patch长度)
        Q_points = None  # 暂时设为 None，避免误导，应该使用 queries_encoded 来计算异常分数

        # 4. Encoder (处理 Token Queries)
        queries_encoded = self.encoder(queries_token)  # Q_encoded: (B, N_patch, D)

        # 5. Memory Module Readout
        outputs_mem = self.mem_module(queries_encoded)
        out_mem = outputs_mem['output']  # O: (B, N_patch, 2D)
        attn = outputs_mem['attn']
        mem = self.mem_module.mem

        # 6. Decoder -> Reconstruction X_hat
        X_hat_token = self.weak_decoder(out_mem)  # (B, N_patch, C_out)

        # 上采样/逆Patching: (B, N_patch, C) -> (B, N_patch, P_len, C) -> (B, L, C)
        # 注意：这里假设 Decoder 输出的 C_out 维度，与原始 C_in 维度相同
        X_hat = X_hat_token.unsqueeze(2).repeat(1, 1, self.patch_len, 1)
        X_hat = X_hat.reshape(B, L, C)

        # 返回结果：
        return {
            "out": X_hat,
            "queries": queries_encoded,  # Token Queries (用于 Memory Loss)
            "point_queries": queries_encoded,  # 使用编码后的 Token Queries 作为点级 queries 的替代 (N_patch 长度)
            "mem": mem,
            "attn": attn
        }

    # === 新增方法：用于 K-Means 聚类 ===
    def get_all_queries_for_kmeans(self, x):
        B, L, C = x.shape

        # 1. Patching & Flattening
        x_flattened = self.patch_and_flatten(x)

        # 2. Patch Embedding (Tokenization)
        queries_token_flat = self.patch_embedding(x_flattened)

        # 3. Reshape back to Sequence: (B, N_patch, D)
        queries_seq = queries_token_flat.view(B, self.num_patches, self.d_model)

        # 4. Encoder
        queries = self.encoder(queries_seq)  # (B, N_patch, D)

        return {'queries': queries}
class GlobalSlidingTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, global_context_N, global_context_L, a_init=0.1,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', device=None,
                 n_memory=50, shrink_thres=0.0025):
        super(GlobalSlidingTransformer, self).__init__()

        self.win_size = win_size
        self.global_context_N = global_context_N
        self.global_context_L = global_context_L
        self.d_model = d_model

        # --- 全局上下文注入参数 ---
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.register_buffer('X0_context', torch.zeros(win_size, d_model))

        # 1. Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # 2. Encoder (Transformer)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # 3. Memory Module
        # 假设 MemoryModule 接收 (B, L, D) 的 queries，输出 (B, L, D) 的 readout
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device)

        # 4. Decoder
        # 假设 MemoryModule 输出 O 的维度是 d_model，因此 Decoder 输入是 d_model
        self.weak_decoder = Decoder(2*d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def set_X0(self, X0_tensor):
        """
        用于在训练前设置全局上下文 X0。
        X0_tensor 形状应为 (L, D)
        """
        if X0_tensor.shape[0] != self.win_size or X0_tensor.shape[1] != self.d_model:
            raise ValueError(f"X0 shape mismatch. Expected ({self.win_size}, {self.d_model}), got {X0_tensor.shape}")

        # 使用 copy_ 或 buffer 赋值
        self.X0_context.data.copy_(X0_tensor)

    def forward(self, x):
        '''
        x (input time window Xs) : B x L x enc_in
        L = win_size
        '''
        # 1. 局部窗口嵌入 (B, L, enc_in) -> (B, L, D)
        # 1. 嵌入与注入 (Xs' = E(Xs) + a * E(X0))
        Xs = self.embedding(x)
        aX0 = self.a * self.X0_context.to(Xs.device)
        Xs_prime = Xs + aX0

        # 2. Transformer Encoder -> Queries Q
        queries = self.encoder(Xs_prime)  # Q: (B, L, D)

        # 3. Memory Module Readout
        outputs_mem = self.mem_module(queries)

        out_mem = outputs_mem['output']  # O: (B, L, D)
        attn = outputs_mem['attn']  # Attn: (B*L, N_mem)
        mem = self.mem_module.mem  # M: (N_mem, D)

        # 4. Decoder -> Reconstruction X_hat
        X_hat = self.weak_decoder(out_mem)  # X_hat: (B, L, C_out)

        # 返回所有需要的组件用于损失计算和测试
        return {"out": X_hat, "queries": queries, "mem": mem, "attn": attn}
        # === 新增方法：用于 K-Means 聚类 ===

    def get_all_queries_for_kmeans(self, x):
        '''
        计算 queries (Encoder output) 用于 K-Means 初始化。
        x (input time window Xs) : B x L x enc_in
        Returns: Queries (B, L, D)
        '''

        # 1. 嵌入与注入 (Xs' = E(Xs) + a * E(X0))
        Xs = self.embedding(x)

        # 注意：在训练 K-Means 时，我们通常使用固定的 X0，但 X0_context 已经在 self.register_buffer 中
        # 我们需要确保 X0_context 已经设置，并且在正确的设备上。
        aX0 = self.a * self.X0_context.to(Xs.device)
        Xs_prime = Xs + aX0

        # 2. Transformer Encoder -> Queries Q
        queries = self.encoder(Xs_prime)  # Q: (B, L, D)

        # 为了兼容 Solver 中对字典的期望，我们返回一个字典（即使只有一个键）
        return {'queries': queries}


class GlobalContextBranch(nn.Module):
    def __init__(self, win_size, enc_in, c_out, N_sampling, L_context, a_init=1.0,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', device=None,
                 n_memory=10, shrink_thres=0.0025, memory_init_embeddings=None, phase_type=None, dataset_name=None):
        super(GlobalContextBranch, self).__init__()

        self.win_size = win_size  # L
        self.d_model = d_model
        self.N_sampling = N_sampling  # N
        self.L_context = L_context  # 确保 L_context == win_size

        # --- 核心：可学习的全局上下文缩放因子 a ---
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.register_buffer('X0_context', torch.zeros(self.win_size, d_model))  # X0 buffer (L, D)

        # 1. Embedding (C_in -> D)
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # 2. Encoder (Transformer, 序列长度 L)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # 3. Memory Module (接收 B*L, D)
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                                       memory_init_embedding=memory_init_embeddings, phase_type=phase_type,
                                       dataset_name=f"{dataset_name}_gclt")

        # 4. Decoder (输入 2D)
        self.weak_decoder = Decoder(d_model * 2, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def set_X0(self, X0_tensor):
        """用于在训练前设置全局上下文 X0。"""
        if X0_tensor.shape[0] != self.win_size or X0_tensor.shape[1] != self.d_model:
            raise ValueError(f"X0 shape mismatch. Expected ({self.win_size}, {self.d_model}), got {X0_tensor.shape}")
        self.X0_context.data.copy_(X0_tensor.to(self.X0_context.device))

    def forward(self, x):
        B, L, C = x.shape

        # 1. 局部窗口嵌入 Xs: (B, L, C) -> (B, L, D)
        Xs = self.embedding(x)

        # 2. 全局上下文注入 Xs' = Xs + a * X0
        aX0 = self.a * self.X0_context.to(Xs.device)
        Xs_prime = Xs + aX0

        # 3. Encoder (Transformer) -> Queries Q (点级)
        queries = self.encoder(Xs_prime)  # Q: (B, L, D)

        # 4. Memory Module Readout
        outputs_mem = self.mem_module(queries)

        out_mem = outputs_mem['output']
        attn = outputs_mem['attn']
        mem = self.mem_module.mem

        # 5. Decoder -> Reconstruction X_hat
        X_hat = self.weak_decoder(out_mem)

        # 返回 Queries (用于 K-Means) 和重构结果
        return {"out": X_hat, "queries": queries, "mem": mem, "attn": attn}

    def get_all_queries_for_kmeans(self, x):
        B, L, C = x.shape
        Xs = self.embedding(x)
        aX0 = self.a * self.X0_context.to(Xs.device)
        Xs_prime = Xs + aX0
        queries = self.encoder(Xs_prime)
        return {'queries': queries}
def deterministic_interpolate(tensor, target_len):
    """
    一个完全确定性的、基于 GPU 的一维线性插值函数。
    tensor: 输入张量，形状 [N, C, L_in]
    target_len: 目标长度 L_out
    """
    N, C, L_in = tensor.shape
    device = tensor.device

    # 如果目标长度和输入长度相同，直接返回
    if L_in == target_len:
        return tensor

    # 1. 计算每个输出点在原始输入尺度下的“虚拟”位置
    #    例如，上采样 2 倍，输出点 0,1,2,3 对应输入点 0, 0.5, 1, 1.5
    ratio = (L_in - 1) / (target_len - 1)
    output_indices_float = torch.arange(target_len, device=device) * ratio

    # 2. 找到每个输出点左右两边的原始输入点的索引
    left_indices = output_indices_float.floor().long()
    right_indices = (left_indices + 1).clamp(max=L_in - 1)  # clamp 防止越界

    # 3. 计算每个输出点距离左边点的权重
    #    例如，虚拟位置 1.5 距离左边点 1 的权重是 0.5
    right_weights = output_indices_float - left_indices.float()
    left_weights = 1.0 - right_weights

    # 4. 从原始张量中 gather 出左右两边的值
    #    我们需要将索引扩展到 [N, C, target_len]
    left_values = tensor.gather(-1, left_indices.expand(N, C, -1))
    right_values = tensor.gather(-1, right_indices.expand(N, C, -1))

    # 5. 进行加权平均
    #    需要将权重也扩展到 [N, C, target_len]
    interpolated_values = left_weights * left_values + right_weights * right_values

    return interpolated_values


class PatchBranch(nn.Module):
    def __init__(self, win_size, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 patch_scales, n_memory_per_scale, shrink_thres, device, phase_type, dataset_name):
        super(PatchBranch, self).__init__()
        self.patch_scales = patch_scales
        self.d_model = d_model
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
        self.encoders = nn.ModuleList()
        self.memories = nn.ModuleList()
        for z in patch_scales:
            effective_win_size = win_size // z
            self.encoders.append(
                Encoder([EncoderLayer(AttentionLayer(effective_win_size, d_model, n_heads, dropout=dropout),
                                      d_model, d_ff, dropout=dropout, activation=activation) for _ in range(e_layers)],
                        norm_layer=nn.LayerNorm(d_model)))
            mem_dataset_name_suffix = f"_patch_z{z}"
            self.memories.append(
                MemoryModule(n_memory=n_memory_per_scale, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
                             phase_type=phase_type, dataset_name=f"{dataset_name}{mem_dataset_name_suffix}"))
        fusion_input_dim = d_model * len(patch_scales)
        self.fusion_net = nn.Sequential(nn.Linear(fusion_input_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.decoder = Decoder(d_model * 2, enc_in, d_ff, activation, dropout)
####这是原先可以运行的
    def forward(self, x, memory_init_embeddings=None):
        N, L, C_in = x.shape
        queries_from_scales, attns_from_scales, mems_from_scales, recons_from_scales = {}, {}, {}, {}
        for i, z in enumerate(self.patch_scales):
            x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=z, stride=z).permute(0, 2, 1) if z > 1 else x
            pooled_embed = self.embedding(x_pooled)
            patch_queries = self.encoders[i](pooled_embed)
            if memory_init_embeddings and f"patch_z{z}" in memory_init_embeddings:
                self.memories[i].mem = nn.Parameter(memory_init_embeddings[f"patch_z{z}"])
            mem_out = self.memories[i](patch_queries)
            recons_queries_z = mem_out['output'].view(N, L // z, 2 * self.d_model)[:, :, self.d_model:]
            queries_from_scales[z] = patch_queries
            attns_from_scales[z] = mem_out['attn']
            mems_from_scales[z] = self.memories[i].mem
            # recons_from_scales[z] = F.interpolate(recons_queries_z.transpose(1, 2), size=L, mode='linear',
            #                                       align_corners=False).transpose(1, 2)
            # ==============================================================
            # ===        使用我们自己的确定性 interpolate 函数         ===
            # ==============================================================
            # 旧代码:
            # recons_from_scales[z] = F.interpolate(recons_queries_z.transpose(1, 2), size=L, mode='linear',
            #                                       align_corners=False).transpose(1, 2)

            # 新代码:
            # 1. 确保输入是 [N, C, L] 格式
            tensor_to_interpolate = recons_queries_z.transpose(1, 2)
            # 2. 调用确定性函数
            interpolated_tensor = deterministic_interpolate(tensor_to_interpolate, target_len=L)
            # 3. 恢复形状
            recons_from_scales[z] = interpolated_tensor.transpose(1, 2)
            # ==============================================================

        fusion_input_list = [recons_from_scales[z] for z in self.patch_scales]  # Fuse recons as per H-PAD figure
        fusion_input = torch.cat(fusion_input_list, dim=-1)
        final_queries = self.fusion_net(fusion_input)
        final_mem_out = self.memories[0](final_queries)  # Use z=1 memory for final enhancement
        final_recons_output = self.decoder(final_mem_out['output'])
        return {'output': final_recons_output, 'queries': queries_from_scales, 'mems': mems_from_scales,
                'attns': attns_from_scales}






class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)    # N x L x C(=d_model)
    

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # self.decoder_layer = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=2,
        #                              batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        # self.decoder_layer_add = nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        # out = self.decoder_layer1(x.transpose(-1, 1))
        # out = self.dropout(self.activation(self.batchnorm(out)))

        # decoder ablation
        # for _ in range(10):
        #     out = self.dropout(self.activation(self.decoder_layer_add(out)))

        # out = self.decoder_layer2(out).transpose(-1, 1)     
        '''
        out : reconstructed output
        '''
        out = self.out_linear(x)
        return out      # N x L x c_out


class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu',\
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

       

       
        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)   # N x L x C(=d_model)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        
        # 原本的
        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device, memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name)



        # ours
        self.weak_decoder = Decoder(2* d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

        # baselines
        # self.weak_decoder = Decoder(d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)


    def forward(self, x):
        '''
        x (input time window) : N x L x enc_in
        '''
       

        x = self.embedding(x)   # embeddin : N x L x C(=d_model)



        queries = out = self.encoder(x)   # encoder out : N x L x C(=d_model)

       

        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
       

        if self.memory_initial:
            return {"out":out, "memory_item_embedding":None, "queries":queries, "mem":mem}
        else:

            out = self.weak_decoder(out)

            '''
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            '''
            return {"out":out, "memory_item_embedding":memory_item_embedding, "queries":queries, "mem":mem, "attn":attn}
