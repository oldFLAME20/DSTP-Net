import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

from model.Transformer import  PatchBranch,PatchLocalBranch,GlobalContextBranch,EncoderLayer, Encoder, Decoder, AttentionLayer, InputEmbedding, MemoryModule
from model.embedding import InputEmbedding
from model.ours_memory_module import MemoryModule


class TransformerVar(nn.Module):
    # 这是 T Branch 的核心结构：Encoder(L) -> Memory -> Decoder(2D -> C)
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu',
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial
        self.d_model = d_model

        # 1. Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # 2. Encoder
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
        self.mem_module = MemoryModule(
            n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, device=device,
            memory_init_embedding=memory_init_embedding, phase_type=phase_type, dataset_name=dataset_name
        )

        # 4. Decoder (输入 2D)
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def forward(self, x):
        x_embedded = self.embedding(x)
        queries = self.encoder(x_embedded)

        outputs = self.mem_module(queries)
        readout_2D = outputs['output']  # Q || O
        attn = outputs['attn']
        memory_item_embedding = outputs['memory_init_embedding']
        mem = self.mem_module.mem

        if self.memory_initial:
            # 第一阶段只返回 Queries 和 Memory 原型
            return {"out": readout_2D, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            # 第二阶段返回最终重构和所有特征
            X_hat = self.weak_decoder(readout_2D)

            return {
                "out": X_hat,
                "out_mem_2D": readout_2D,  # Memory Readout (Q || O)
                "memory_item_embedding": memory_item_embedding,
                "queries": queries,
                "mem": mem,
                "attn": attn
            }

    def get_all_queries_for_kmeans(self, x):
        x_embedded = self.embedding(x)
        queries = self.encoder(x_embedded)
        return {"queries": queries}

#gcn+transformer
class GCN_T_Model(nn.Module):
    """
    消融实验模型：融合 GCN 结构专家和标准 Transformer (T) 专家。
    T 专家使用 TransformerVar 的结构 (T Encoder -> T Memory -> T Decoder)。
    """

    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, n_memory_t, shrink_thres, gamma,
                 device, phase_type, dataset_name, memory_initial, memory_init_embeddings):
        super(GCN_T_Model, self).__init__()

        # 融合参数 gamma
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.d_model = d_model

        # === 1. Standard Transformer Branch (T Expert) ===
        # 使用 TransformerVar (T Encoder -> T Memory -> T Decoder)
        # 注意：这里需要确保 TransformerVar 有相应的 memory_init_embeddings 参数
        self.t_branch = TransformerVar(
            win_size=win_size, enc_in=enc_in, c_out=c_out,
            n_memory=n_memory_t, shrink_thres=shrink_thres,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
            activation=activation, device=device,
            memory_init_embedding=memory_init_embeddings[
                't'] if memory_init_embeddings and 't' in memory_init_embeddings else None,
            memory_initial=memory_initial, phase_type=phase_type, dataset_name=f"{dataset_name}_t"
        )

        # === 2. GCN Branch (结构专家) ===
        # GCN 需要自己的 embedding，以避免依赖 t_branch 内部的实现
        self.gcn_embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type,
                                              dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)

        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        # 初始化 GCN Memory
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])

        # 初始化 T Branch Memory (假设键名为 't')
        if 't' in mem_embeds:
            self.t_branch.mem_module.mem = nn.Parameter(mem_embeds['t'])

    def forward(self, x):
        # --- 1. T Branch Calculation (Baseline T+M) ---
        # T_branch (TransformerVar) 内部处理 embedding 和 reconstruction
        t_out_dict = self.t_branch(x)
        Xt_hat = t_out_dict['out']
        # print(f"DEBUG: Xt_hat shape: {Xt_hat.shape}")  # 必须是 (B, L, 55)

        # --- 2. GCN Branch Calculation ---
        x_embed_gcn = self.gcn_embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        Xgcn_hat = self.gcn_branch_decoder(gcn_mem_out['output'])
        # print(f"DEBUG: Xgcn_hat shape: {Xgcn_hat.shape}")  # 必须是 (B, L, 55)

        # --- 3. Fusion ---
        gamma_val = torch.sigmoid(self.gamma)
        final_reconstruction = gamma_val * Xt_hat + (1 - gamma_val) * Xgcn_hat

        # --- Return Combined Output ---
        return {
            'final_output': final_reconstruction,
            # T Branch Info
            't_queries': t_out_dict['queries'],
            't_mems': self.t_branch.mem_module.mem,  # 直接返回 Memory Module 的原型
            't_attns': t_out_dict['attn'],
            # GCN Info
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn']
        }

    def get_all_queries_for_kmeans(self, x):
        # GCN Queries
        x_embed_gcn = self.gcn_embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)

        # T Queries
        # TransformerVar.forward 包含 embedding，所以我们调用它的辅助函数
        t_queries = self.t_branch.get_all_queries_for_kmeans(x)['queries']

        return {'gcn_queries': gcn_queries, 't_queries': t_queries}


####消融实验
# =====================================================================
# === 新增：用于消融实验的 "GCN-Only" 模型 ===
# =====================================================================
class GCN_Only_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, shrink_thres, device, phase_type, dataset_name,
                 memory_initial, memory_init_embeddings, **kwargs): # 使用**kwargs忽略不用的参数
        super(GCN_Only_Model, self).__init__()
        print("Initializing GCN_Only_Model for ablation study.")
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type, dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)
        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])

    def forward(self, x):
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        final_reconstruction = self.gcn_branch_decoder(gcn_mem_out['output'])
        return {
            'final_output': final_reconstruction,
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn']
        }

    def get_all_queries_for_kmeans(self, x):
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        return {'gcn_queries': gcn_queries}


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


# 假设导入了 GlobalContextBranch, InputEmbedding, MemoryModule, Decoder, GcnEncoderBranch

# 假设 GcnEncoderBranch, InputEmbedding, MemoryModule, Decoder, GCNConv 已定义
#这是gcn+global
class GCN_GCLT_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, N_sampling, L_context, a_init, n_memory_gclt, shrink_thres, gamma,
                 device, phase_type, dataset_name, memory_initial, memory_init_embeddings):
        super(GCN_GCLT_Model, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.d_model = d_model

        # GCN 和 GCLT 共享 Embedding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # === GCN 分支 ===
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type,
                                              dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)

        # === GCLT 分支 (Global Context Injected Transformer) ===
        self.gclt_branch = GlobalContextBranch(
            win_size=win_size, enc_in=enc_in, c_out=c_out, d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout, activation=activation,
            N_sampling=N_sampling, L_context=L_context, a_init=a_init,
            n_memory=n_memory_gclt, shrink_thres=shrink_thres, device=device, phase_type=phase_type,
            dataset_name=dataset_name  # GCLT 内部会添加 _gclt 后缀
        )

        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        # 初始化 GCN 的记忆
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])

        # 初始化 GCLT 的记忆
        if 'gclt' in mem_embeds:
            self.gclt_branch.mem_module.mem = nn.Parameter(mem_embeds['gclt'])

    def forward(self, x):
        # 共享嵌入 (GCN 只能处理嵌入后的特征)
        x_embed = self.embedding(x)

        # --- 1. GCN 分支计算 ---
        gcn_queries = self.gcn_branch_encoder(x_embed)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        Xgcn_hat = self.gcn_branch_decoder(gcn_mem_out['output'])

        # --- 2. GCLT 分支计算 ---
        # GCLT 分支需要原始输入 x，但它会在内部重新嵌入 (这里需要修正)
        # 为了避免双重嵌入，我们让 GCLT 接收嵌入后的特征 (需要修改 GCLT.forward)
        # **为了兼容性，我们让 GCLT 内部自己处理嵌入，并确保 set_X0 逻辑正确**
        gclt_out_dict = self.gclt_branch(x)
        Xgclt_hat = gclt_out_dict['out']

        # --- 3. 融合 ---
        gamma_val = torch.sigmoid(self.gamma)
        final_reconstruction = gamma_val * Xgclt_hat + (1 - gamma_val) * Xgcn_hat

        # --- 返回所有必要信息 ---
        return {
            'final_output': final_reconstruction,
            # GCLT 信息
            'gclt_queries': gclt_out_dict['queries'],  # (B, L, D)
            'gclt_mem': self.gclt_branch.mem_module.mem,
            'gclt_attn': gclt_out_dict['attn'],
            # GCN 信息
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn']
        }

    def get_all_queries_for_kmeans(self, x):
        # GCN Queries (使用共享嵌入 x_embed)
        x_embed = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed)

        # GCLT Queries
        gclt_queries = self.gclt_branch.get_all_queries_for_kmeans(x)['queries']

        return {'gcn_queries': gcn_queries, 'gclt_queries': gclt_queries}




# 假设这是 model/GNN_Module.py 或 model/Transformer.py 附近
#这是gcn+patch
class GCN_PLT_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, patch_len, n_memory_patch, shrink_thres, gamma,
                 device, phase_type, dataset_name, memory_initial, memory_init_embeddings):
        super(GCN_PLT_Model, self).__init__()
        self.patch_len = patch_len
        self.num_patches = win_size//patch_len
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # === GCN 分支 (保持不变) ===
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type,
                                              dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)

        # === Patch 分支 (替换为 PatchLocalBranch) ===
        self.plt_branch = PatchLocalBranch(
            win_size=win_size, enc_in=enc_in, c_out=c_out, d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_ff, dropout=dropout, activation=activation,
            patch_len=patch_len,  # 使用新的 patch_len 参数
            n_memory=n_memory_patch, shrink_thres=shrink_thres, device=device, phase_type=phase_type,
            dataset_name=dataset_name
        )

        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        # 初始化 GCN 的记忆
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])

        # 初始化 Patch Local Branch 的记忆
        if 'patch' in mem_embeds and 1 in mem_embeds['patch']:  # 假设 K-Means 结果存储在 'patch' 键下的尺度 1
            self.plt_branch.plt_model.mem_module.mem = nn.Parameter(mem_embeds['patch'][1])  # 直接赋值给 Memory Module

    def forward(self, x):
        # --- GCN 分支计算 ---
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        Xgcn_hat = self.gcn_branch_decoder(gcn_mem_out['output'])

        # --- Patch Local Transformer 分支计算 ---
        plt_out_dict = self.plt_branch(x)
        Xplt_hat = plt_out_dict['output']

        # --- 融合 ---
        gamma_val = torch.sigmoid(self.gamma)
        final_reconstruction = gamma_val * Xplt_hat + (1 - gamma_val) * Xgcn_hat

        # --- 返回所有必要信息 ---
        return {
            'final_output': final_reconstruction,
            'patch_queries': plt_out_dict['patch_queries'],
            'patch_mems': plt_out_dict['patch_mems'],
            'patch_attns': plt_out_dict['patch_attns'],
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn'],
            # === 关键：新增点级 Queries ===
            'point_queries': plt_out_dict['point_queries']
        }

    def get_all_queries_for_kmeans(self, x):
        # ... (GCN 分支 Queries 提取保持不变) ...
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)

        # Patch Queries 提取
        patch_queries_dict = self.plt_branch.get_all_queries_for_kmeans(x)['patch_queries']

        return {'gcn_queries': gcn_queries, 'patch_queries': patch_queries_dict}
# =====================================================================
# === 新增：用于消融实验的 "Patch-Only" 模型 ===
# =====================================================================
class PatchOnly_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 patch_scales, n_memory_patch, shrink_thres, device, phase_type, dataset_name,
                 memory_initial, memory_init_embeddings, **kwargs): # 使用**kwargs忽略不用的参数
        super(PatchOnly_Model, self).__init__()
        print("Initializing PatchOnly_Model for ablation study.")
        # 注意：PatchOnly_Model 不需要自己的 embedding, 因为 PatchBranch 内部自带了
        self.patch_branch = PatchBranch(win_size, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
                                        patch_scales, n_memory_patch, shrink_thres, device, phase_type, dataset_name)
        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        if 'patch' in mem_embeds:
            for i, z in enumerate(self.patch_branch.patch_scales):
                key = f"patch_z{z}"
                if key in mem_embeds['patch']:
                    self.patch_branch.memories[i].mem = nn.Parameter(mem_embeds['patch'][key])

    def forward(self, x):
        patch_out_dict = self.patch_branch(x)
        final_reconstruction = patch_out_dict['output']
        return {
            'final_output': final_reconstruction,
            'patch_queries': patch_out_dict['queries'],
            'patch_mems': patch_out_dict['mems'],
            'patch_attns': patch_out_dict['attns']
        }

    def get_all_queries_for_kmeans(self, x):
        patch_queries_dict = {}
        for i, z in enumerate(self.patch_branch.patch_scales):
            x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=z, stride=z).permute(0, 2, 1) if z > 1 else x
            # 使用 PatchBranch 内部的 embedding
            pooled_embed = self.patch_branch.embedding(x_pooled)
            patch_queries = self.patch_branch.encoders[i](pooled_embed)
            patch_queries_dict[z] = patch_queries
        return {'patch_queries': patch_queries_dict}

# class GcnEncoderBranch(nn.Module):
#     def __init__(self, win_size, d_model, n_layers=1, dropout=0.1, top_k=15, d_graph_embed=64):
#         super(GcnEncoderBranch, self).__init__()
#         self.node_feature_dim = win_size
#         self.num_nodes = d_model
#         self.n_layers = n_layers
#         self.top_k = top_k
#         self.d_graph_embed = d_graph_embed
#
#         # --- 图构建部分与 GAT 分支完全相同 ---
#         self.query_proj = nn.Linear(self.node_feature_dim, self.d_graph_embed)
#         self.key_proj = nn.Linear(self.node_feature_dim, self.d_graph_embed)
# ####这部分使用pytorch的聚合方式，下面是我自己手动实现的聚合
#         # --- 核心修改：使用 GCNConv 替换 GATConv ---
#         self.gcn_layers = nn.ModuleList()
#         for _ in range(n_layers):
#             self.gcn_layers.append(
#                 GCNConv(self.node_feature_dim, self.node_feature_dim)
#             )
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
# #这部分是可以运行的，我的可行代码
#     def forward(self, x):
#         batch_size, win_size, d_model = x.shape
#         x_transposed = x.transpose(1, 2)
#
#         # --- 图构建的前向传播与 GAT 分支完全相同 ---
#         data_list = []
#         for i in range(batch_size):
#             node_features = x_transposed[i]
#             query, key = self.query_proj(node_features), self.key_proj(node_features)
#             adj_matrix = torch.matmul(query, key.t()) / (self.d_graph_embed ** 0.5)
#             # ==========================================================
#             # ===                     核心修改                     ===
#             # ==========================================================
#             # --- 错误的代码 (in-place) ---
#             # adj_matrix.fill_diagonal_(-float('inf'))
#
#             # --- 正确的代码 (non-in-place) ---
#             # 创建一个对角线为 True 的布尔掩码
#             diag_mask = torch.eye(d_model, device=x.device, dtype=torch.bool)
#             # 使用掩码填充，这会返回一个新的张量，而不是修改原始张量
#             adj_matrix = adj_matrix.masked_fill(diag_mask, -float('inf'))
#             # ==========================================================
#             actual_top_k = min(self.top_k, d_model - 1)
#             if actual_top_k <= 0: actual_top_k = 1
#             _, top_k_indices = torch.topk (adj_matrix, k=actual_top_k, dim=-1)
#             source_nodes = torch.arange(d_model, device=x.device).repeat_interleave(actual_top_k)
#             target_nodes = top_k_indices.view(-1)
#             edge_index = torch.stack([source_nodes, target_nodes])
#             self_loops = torch.arange(d_model, dtype=torch.long, device=x.device).unsqueeze(0).repeat(2, 1)
#             edge_index = torch.cat([edge_index, self_loops], dim=1)
#             data_list.append(Data(x=node_features, edge_index=edge_index))
#
#         batched_data = Batch.from_data_list(data_list)
#         h = batched_data.x
#
#         # --- 核心修改：使用 GCN 层进行前向传播 ---
#         for l_idx in range(self.n_layers):
#             h = self.gcn_layers[l_idx](h, batched_data.edge_index)
#             if l_idx < self.n_layers - 1:
#                 h = self.relu(h)
#                 h = self.dropout(h)
#
#         out_transposed = h.view(batch_size, d_model, win_size)
#         return out_transposed.transpose(1, 2)
# #这部分是可以运行的，我的可行代码




# =====================================================================
# === 新增：整合 GCN 和 Patch 两个并行分支的顶层模型 ===
# =====================================================================
class GCN_PAD_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, patch_scales, n_memory_patch, shrink_thres, gamma,
                 device, phase_type, dataset_name, memory_initial, memory_init_embeddings):
        super(GCN_PAD_Model, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # --- 核心修改：使用 GcnEncoderBranch ---
        # 注意：GCN 不需要 n_heads 参数，但为了保持签名一致，我们接收它但不使用
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)

        # 重命名 GAT 相关的组件为 GCN
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type,
                                              dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)

        # Patch 分支保持完全不变
        self.patch_branch = PatchBranch(win_size, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
                                        patch_scales, n_memory_patch, shrink_thres, device, phase_type, dataset_name)

        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        # 仅初始化 GCN 和 Patch 的记忆
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])
        if 'patch' in mem_embeds:
            for i, z in enumerate(self.patch_branch.patch_scales):
                key = f"patch_z{z}"
                if key in mem_embeds['patch']:
                    self.patch_branch.memories[i].mem = nn.Parameter(mem_embeds['patch'][key])

    def forward(self, x):
        # --- GCN 分支计算 ---
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        Xgcn_hat = self.gcn_branch_decoder(gcn_mem_out['output'])

        # --- Patch 分支计算 (保持不变) ---
        patch_out_dict = self.patch_branch(x)
        Xz_hat = patch_out_dict['output']

        # --- 融合 (保持不变) ---
        gamma_val = torch.sigmoid(self.gamma)
        final_reconstruction = gamma_val * Xz_hat + (1 - gamma_val) * Xgcn_hat

        # --- 返回包含 GCN 信息的字典 ---
        return {
            'final_output': final_reconstruction,
            'patch_queries': patch_out_dict['queries'],
            'patch_mems': patch_out_dict['mems'],
            'patch_attns': patch_out_dict['attns'],
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn']
        }

    def get_all_queries_for_kmeans(self, x):
        # --- GCN 分支 ---
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)

        # --- Patch 分支 (保持不变) ---
        patch_queries_dict = {}
        for i, z in enumerate(self.patch_branch.patch_scales):
            x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=z, stride=z).permute(0, 2, 1) if z > 1 else x
            pooled_embed = self.patch_branch.embedding(x_pooled)
            patch_queries = self.patch_branch.encoders[i](pooled_embed)
            patch_queries_dict[z] = patch_queries

        return {'gcn_queries': gcn_queries, 'patch_queries': patch_queries_dict}




class GcnEncoderBranch(nn.Module):
    def __init__(self, win_size, d_model, n_layers=1, dropout=0.1, top_k=15, d_graph_embed=64):
        super(GcnEncoderBranch, self).__init__()
        self.node_feature_dim = win_size
        self.num_nodes = d_model
        self.n_layers = n_layers
        self.top_k = top_k
        self.d_graph_embed = d_graph_embed

        # --- 图构建部分与 GAT 分支完全相同 ---
        self.query_proj = nn.Linear(self.node_feature_dim, self.d_graph_embed)
        self.key_proj = nn.Linear(self.node_feature_dim, self.d_graph_embed)
####这部分使用pytorch的聚合方式，下面是我自己手动实现的聚合
        # --- 核心修改：使用 GCNConv 替换 GATConv ---
        self.gcn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gcn_layers.append(
                GCNConv(self.node_feature_dim, self.node_feature_dim)
            )
####这部分使用pytorch的聚合方式，下面是我自己手动实现的聚合
        # --- 核心修改 1: GCNConv 现在只作为线性变换层使用 ---
        # 我们将手动实现聚合，所以 GCNConv 只用来乘以权重 W
        # self.gcn_lins = nn.ModuleList()
        # for _ in range(n_layers):
        #     # GCNConv 内部有一个 nn.Linear，我们将直接使用它
        #     self.gcn_lins.append(nn.Linear(self.node_feature_dim, self.node_feature_dim))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
#这部分是可以运行的，我的可行代码
    def forward(self, x):
        batch_size, win_size, d_model = x.shape
        x_transposed = x.transpose(1, 2)

        # --- 图构建的前向传播与 GAT 分支完全相同 ---
        data_list = []
        for i in range(batch_size):
            node_features = x_transposed[i]
            query, key = self.query_proj(node_features), self.key_proj(node_features)
            adj_matrix = torch.matmul(query, key.t()) / (self.d_graph_embed ** 0.5)
            adj_matrix.fill_diagonal_(-float('inf'))
            actual_top_k = min(self.top_k, d_model - 1)
            if actual_top_k <= 0: actual_top_k = 1
            _, top_k_indices = torch.topk (adj_matrix, k=actual_top_k, dim=-1)
            source_nodes = torch.arange(d_model, device=x.device).repeat_interleave(actual_top_k)
            target_nodes = top_k_indices.view(-1)
            edge_index = torch.stack([source_nodes, target_nodes])
            self_loops = torch.arange(d_model, dtype=torch.long, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            data_list.append(Data(x=node_features, edge_index=edge_index))

        batched_data = Batch.from_data_list(data_list)
        h = batched_data.x

        # --- 核心修改：使用 GCN 层进行前向传播 ---
        for l_idx in range(self.n_layers):
            h = self.gcn_layers[l_idx](h, batched_data.edge_index)
            if l_idx < self.n_layers - 1:
                h = self.relu(h)
                h = self.dropout(h)

        out_transposed = h.view(batch_size, d_model, win_size)
        return out_transposed.transpose(1, 2)
#这部分是可以运行的，我的可行代码
#     def forward(self, x):
#         batch_size, win_size, d_model = x.shape
#         x_transposed = x.transpose(1, 2)  # Shape: [batch_size, d_model, win_size]
#
#         data_list = []
#         for i in range(batch_size):
#             node_features = x_transposed[i]  # Shape: [d_model, win_size]
#
#             # --- 核心修改: 使用余弦相似度构建邻接矩阵 ---
#             # 1. 扩展维度以进行广播计算 (N, 1, D) vs (1, N, D) -> (N, N)
#             node_features_unsqueezed1 = node_features.unsqueeze(1)
#             node_features_unsqueezed0 = node_features.unsqueeze(0)
#
#             # 2. 计算所有节点对之间的余弦相似度
#             adj_matrix = F.cosine_similarity(node_features_unsqueezed1, node_features_unsqueezed0, dim=-1)
#             # adj_matrix shape: [d_model, d_model]
#             # ---------------------------------------------------
#
#             # 后续的图稀疏化和边索引构建逻辑完全保持不变
#             adj_matrix.fill_diagonal_(-float('inf'))  # 屏蔽自环，top_k 不会选到自己
#
#             actual_top_k = min(self.top_k, d_model - 1)
#             if actual_top_k <= 0: actual_top_k = 1
#
#             _, top_k_indices = torch.topk(adj_matrix, k=actual_top_k, dim=-1)
#
#             source_nodes = torch.arange(d_model, device=x.device).repeat_interleave(actual_top_k)
#             target_nodes = top_k_indices.view(-1)
#             edge_index = torch.stack([source_nodes, target_nodes])
#
#             # 添加自环
#             self_loops = torch.arange(d_model, dtype=torch.long, device=x.device).unsqueeze(0).repeat(2, 1)
#             edge_index = torch.cat([edge_index, self_loops], dim=1)
#
#             data_list.append(Data(x=node_features, edge_index=edge_index))
#
#         # 后续的批处理和GCN传播逻辑完全保持不变
#         batched_data = Batch.from_data_list(data_list)
#         h = batched_data.x
#
#         for l_idx in range(self.n_layers):
#             h = self.gcn_layers[l_idx](h, batched_data.edge_index)
#             if l_idx < self.n_layers - 1:
#                 h = self.relu(h)
#                 h = self.dropout(h)
#
#         out_transposed = h.view(batch_size, d_model, win_size)
#         return out_transposed.transpose(1, 2)




# =====================================================================
# === 新增：整合 GCN 和 Patch 两个并行分支的顶层模型 ===
# =====================================================================
class GCN_PAD_Model(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model, n_heads, e_layers, d_ff, dropout, activation,
                 n_memory_gcn, top_k, d_graph_embed, patch_scales, n_memory_patch, shrink_thres, gamma,
                 device, phase_type, dataset_name, memory_initial, memory_init_embeddings):
        super(GCN_PAD_Model, self).__init__()
        #### //todo 将gamma修改成门控,现在又是自主学习的
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        ####

        # # 新的方式 (动态门控):
        # # 1. 定义一个特征提取器，用于将输入序列压缩成一个代表性向量。
        # #    AdaptiveAvgPool1d 是一个简单而有效的方法。
        # self.gate_feature_extractor = nn.AdaptiveAvgPool1d(1)
        #
        # # 2. 定义一个小型多层感知机 (MLP) 作为门控网络。
        # #    它的输入是特征数(enc_in)，输出是一个标量。
        # gate_hidden_dim = max(16, enc_in // 4)  # 设置一个合理的隐藏层维度
        # self.gamma_gate_mlp = nn.Sequential(
        #     nn.Linear(enc_in, gate_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(gate_hidden_dim, 1),
        #     nn.Sigmoid()  # 使用 Sigmoid 函数将输出值 gamma 压缩到 (0, 1) 区间
        # )
        # print("Initialized GCN_PAD_Model with a dynamic gamma gating mechanism.")

        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)

        # --- 核心修改：使用 GcnEncoderBranch ---
        # 注意：GCN 不需要 n_heads 参数，但为了保持签名一致，我们接收它但不使用
        self.gcn_branch_encoder = GcnEncoderBranch(win_size, d_model, 1, dropout, top_k, d_graph_embed)

        # 重命名 GAT 相关的组件为 GCN
        self.gcn_branch_memory = MemoryModule(n_memory_gcn, d_model, shrink_thres, device, phase_type=phase_type,
                                              dataset_name=f"{dataset_name}_gcn")
        self.gcn_branch_decoder = Decoder(d_model * 2, c_out, d_ff, activation, dropout)

        # Patch 分支保持完全不变
        self.patch_branch = PatchBranch(win_size, enc_in, d_model, n_heads, e_layers, d_ff, dropout, activation,
                                        patch_scales, n_memory_patch, shrink_thres, device, phase_type, dataset_name)

        if not memory_initial and memory_init_embeddings is not None:
            self.initialize_memories(memory_init_embeddings)

    def initialize_memories(self, mem_embeds):
        # 仅初始化 GCN 和 Patch 的记忆
        if 'gcn' in mem_embeds:
            self.gcn_branch_memory.mem = nn.Parameter(mem_embeds['gcn'])
        if 'patch' in mem_embeds:
            for i, z in enumerate(self.patch_branch.patch_scales):
                key = f"patch_z{z}"
                if key in mem_embeds['patch']:
                    self.patch_branch.memories[i].mem = nn.Parameter(mem_embeds['patch'][key])

    def forward(self, x):
        # #### //todo 将gamma修改为门控
        # # ======================================================================
        # # === 核心修改 2: 计算动态 gamma 值 ===
        # # ======================================================================
        # # 1. 提取全局特征用于门控网络
        # #    [B, L, C] -> [B, C, L]
        # x_permuted = x.permute(0, 2, 1)
        # #    [B, C, L] -> [B, C, 1] -> [B, C]
        # global_features = self.gate_feature_extractor(x_permuted).squeeze(-1)
        #
        # # 2. 通过门控网络计算 gamma_val
        # #    gamma_val 的形状将是 [Batch, 1]
        # gamma_val = self.gamma_gate_mlp(global_features)
        # # ======================================================================


        ## GCN不变
        # --- GCN 分支计算 ---
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)
        gcn_mem_out = self.gcn_branch_memory(gcn_queries)
        Xgcn_hat = self.gcn_branch_decoder(gcn_mem_out['output'])

        # --- Patch 分支计算 (保持不变) ---
        patch_out_dict = self.patch_branch(x)
        Xz_hat = patch_out_dict['output']
        #超参数
        # --- 融合 (保持不变) ---
        gamma_val = torch.sigmoid(self.gamma)
        final_reconstruction = gamma_val * Xz_hat + (1 - gamma_val) * Xgcn_hat

        # # 新的融合方式:
        # # gamma_val 的形状是 [B, 1]，需要扩展以匹配 Xz_hat [B, L, C]
        # # [B, 1] -> [B, 1, 1] 以便进行广播
        # gamma_val_expanded = gamma_val.unsqueeze(-1)
        #
        # final_reconstruction = gamma_val_expanded * Xz_hat + (1 - gamma_val_expanded) * Xgcn_hat
        # --- 返回包含 GCN 信息的字典 ---
        return {
            'final_output': final_reconstruction,
            'patch_queries': patch_out_dict['queries'],
            'patch_mems': patch_out_dict['mems'],
            'patch_attns': patch_out_dict['attns'],
            'gcn_queries': gcn_queries,
            'gcn_mem': self.gcn_branch_memory.mem,
            'gcn_attn': gcn_mem_out['attn']
        }

    def get_all_queries_for_kmeans(self, x):
        # --- GCN 分支 ---
        x_embed_gcn = self.embedding(x)
        gcn_queries = self.gcn_branch_encoder(x_embed_gcn)

        # --- Patch 分支 (保持不变) ---
        patch_queries_dict = {}
        for i, z in enumerate(self.patch_branch.patch_scales):
            x_pooled = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=z, stride=z).permute(0, 2, 1) if z > 1 else x
            pooled_embed = self.patch_branch.embedding(x_pooled)
            patch_queries = self.patch_branch.encoders[i](pooled_embed)
            patch_queries_dict[z] = patch_queries

        return {'gcn_queries': gcn_queries, 'patch_queries': patch_queries_dict}

