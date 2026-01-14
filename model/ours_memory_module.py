from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
import  os



class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None, dataset_name=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        # self.memory_init_embedding = memory_init_embedding

        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        # =======================================================
        # ===            最终的、正确的初始化修正             ===
        # =======================================================

        # 创建一个临时的张量 initial_mem
        initial_mem = None

        if memory_init_embedding is None:
            if phase_type == 'test':
                # --- 核心修正加载路径 ---

                # 假设 dataset_name 已经是 "MSL_gcn" 或 "MSL_patch"

                if '_gcn' in dataset_name:
                    file_suffix = '_gcn_memory_item.pth'
                elif '_patch' in dataset_name:
                    # 假设 Patch 分支是单尺度的，且 K-Means 保存时使用了 z1 后缀
                    # 如果 PatchLocalBranch 是单尺度，其 Memory 应该与 'patch_z1' 绑定
                    file_suffix = '_patch_z1_memory_item.pth'
                else:
                    # 如果都不是，回退到默认
                    file_suffix = '_memory_item.pth'

                # 构造加载路径
                load_path = f'./memory_item/{dataset_name.replace("_gcn", "").replace("_patch", "")}{file_suffix}'

                # 为了简化和确保与保存一致性，如果 dataset_name 是 MSL_patch，我们希望加载 MSL_patch_z1_memory_item.pth
                # 如果我们在 GCN_PLT_Model 中传入的是 MSL_patch (即 dataset_name='MSL_patch')

                # 重新简化加载逻辑: 假设保存时文件名是 {dataset}_gcn_... 或 {dataset}_patch_z1_...
                # 我们需要知道这个 Memory Module 属于哪个分支
                if 'patch' in dataset_name:
                    base_name = dataset_name.replace('_patch', '')
                    final_file_name = f"{base_name}_patch_z1_memory_item.pth"  # 匹配 K-Means 保存的单尺度名称
                # 修正后的逻辑：使用原始 dataset 名称和正确的后缀
                elif 'gcn' in dataset_name:
                    base_name = dataset_name.replace('_gcn', '')
                    final_file_name = f"{base_name}_gcn_memory_item.pth"
                else:
                    final_file_name = f"{dataset_name}_memory_item.pth"

                load_path = f'./memory_item/{final_file_name}'

                initial_mem = torch.load(load_path, map_location=device)
                print(f"Loaded memory from {load_path} for test phase.")
                # # === 修正加载路径 ===
                # load_path = f'./memory_item/{dataset_name}_patch_local_memory_item.pth'
                #
                # # 检查文件是否存在，如果不存在则回退到默认文件名（以防万一）
                # if not os.path.exists(load_path):
                #     # 如果您的旧代码保存的是 MSL_memory_item.pth，这里需要适配
                #     load_path = f'./memory_item/{dataset_name}_memory_item.pth'
                #
                # # load_path = f'./memory_item/{dataset_name}_memory_item.pth'
                # initial_mem = torch.load(load_path)
                # print(f"Loaded memory from {load_path} for test phase.")
            else:
                # 第一阶段训练
                print('Initializing memory with random values for first train phase.')
                initial_mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)
        else:
            # 第二阶段训练
            if phase_type == 'second_train':
                print('Initializing memory from K-Means for second train phase.')
                initial_mem = memory_init_embedding  # 这是一个来自 k-means 的普通张量

        if initial_mem is None:
            raise ValueError("Memory was not initialized. Check phase_type and memory_init_embedding.")

        # =======================================================
        # ===             高性能的双缓冲方案                 ===
        # =======================================================
        # 创建两个可训练的 memory 参数
        self.mem_a = nn.Parameter(initial_mem.clone(), requires_grad=True)
        self.mem_b = nn.Parameter(initial_mem.clone(), requires_grad=True)

        # 一个指针，决定当前 forward 使用哪个 memory
        # 0 表示使用 mem_a, 1 表示使用 mem_b
        self.current_mem_ptr = 0
        # =======================================================

        # 无论来源如何，最后都必须用 nn.Parameter 包装
        self.mem = nn.Parameter(initial_mem, requires_grad=True)

        # =======================================================
    # relu based hard shrinkage function, only works for positive values
    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)

        return output

        # 在 get_attn_score 中移除 .cuda()，让设备管理更集中

    def get_attn_score(self, query, key):
        # 这个方法是OK的，保持不变
        attn = torch.matmul(query, torch.t(key))
        attn = F.softmax(attn, dim=-1)
        if (self.shrink_thres > 0):
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=1)
        return attn

    # =======================================================
    # ===            修改 update 和 read 的签名            ===
    # =======================================================
    # 让它们接收 mem 作为参数，而不是从 self 中读取

    def update(self, query, mem_for_grad):
        # mem_for_grad 是从 forward 函数传入的 self.mem.clone()
        attn = self.get_attn_score(mem_for_grad, query.detach())
        add_mem = torch.matmul(attn, query.detach())
        update_gate = torch.sigmoid(self.U(mem_for_grad) + self.W(add_mem))
        new_mem_val = (1 - update_gate) * mem_for_grad + update_gate * add_mem

        # 在 no_grad 上下文中，用新值原地更新真正的 self.mem 参数
        with torch.no_grad():
            self.mem.copy_(new_mem_val)

    def read(self, query, mem_for_read):
        # mem_for_read 也是从 forward 函数传入的 self.mem.clone()
        attn = self.get_attn_score(query, mem_for_read.detach())
        add_memory = torch.matmul(attn, mem_for_read.detach())
        read_query = torch.cat((query, add_memory), dim=1)
        return {'output': read_query, 'attn': attn}






    # =======================================================
    # ===             修改 forward 的核心逻辑             ===
    # =======================================================
    def forward(self, query):
        s = query.data.shape
        l = len(s)

        self.U.to(query.device)
        self.W.to(query.device)

        query = query.contiguous()
        query = query.view(-1, s[-1])

        # 在所有操作之前，克隆一份 memory，并放到正确的设备上
        # 这个克隆的 mem 将用于本次 forward 的所有计算
        mem_for_this_forward = self.mem.clone().to(query.device)

        if self.phase_type != 'test':
            # 将克隆的 mem 传入 update
            self.update(query, mem_for_this_forward)

        # 同样，将克隆的 mem 传入 read
        outs = self.read(query, mem_for_this_forward)

        read_query, attn = outs['output'], outs['attn']

        if l == 3:
            read_query = read_query.view(s[0], s[1], 2 * s[2])
            attn = attn.view(s[0], s[1], self.n_memory)

        # 返回时，返回的是更新后的 self.mem
        return {'output': read_query, 'attn': attn, 'memory_init_embedding': self.mem.to(query.device)}


    ###以上部分是完全可以运行的版本，但是速度较慢


    # def get_attn_score(self, query, key):
    #     '''
    #     Calculating attention score with sparsity regularization
    #     query (initial features) : (NxL) x C or N x C -> T x C
    #     key (memory items): M x C
    #     '''
    #     attn = torch.matmul(query, torch.t(key.cuda()))    # (TxC) x (CxM) -> TxM
    #     attn = F.softmax(attn, dim=-1)
    #
    #     if (self.shrink_thres > 0):
    #         attn = self.hard_shrink_relu(attn, self.shrink_thres)
    #         # re-normalize
    #         attn = F.normalize(attn, p=1, dim=1)
    #
    #     return attn
    #
    # def read(self, query):
    #     '''
    #     query (initial features) : (NxL) x C or N x C -> T x C
    #     read memory items and get new robust features,
    #     while memory items(cluster centers) being fixed
    #     '''
    #     self.mem = self.mem.cuda()
    #     attn = self.get_attn_score(query, self.mem.detach())  # T x M
    #     add_memory = torch.matmul(attn, self.mem.detach())    # T x C
    #
    #     # add_memory = F.normalize(add_memory, dim=1)
    #     read_query = torch.cat((query, add_memory), dim=1)  # T x 2C
    #
    #     return {'output': read_query, 'attn': attn}
    #
    #
    #
    # def update(self, query):
    #     '''
    #     Update memory items(cluster centers)
    #     Fix Encoder parameters (detach)
    #     query (encoder output features) : (NxL) x C or N x C -> T x C
    #     '''
    #     self.mem = self.mem.cuda()
    #     attn = self.get_attn_score(self.mem, query.detach())  # M x T
    #     add_mem = torch.matmul(attn, query.detach())   # M x C
    #
    #     # update gate : M x C
    #     update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem)) # M x C
    #     self.mem = (1 - update_gate)*self.mem + update_gate*add_mem
    #     #
    #     # self.mem = F.noramlize(self.mem + add_mem, dim=1)   # M x C

    # def forward(self, query):
    #     '''
    #     query (encoder output features) : N x L x C or N x C
    #     '''
    #     s = query.data.shape
    #     l = len(s)
    #
    #     query = query.contiguous()
    #     query = query.view(-1, s[-1])  # N x L x C or N x C -> T x C
    #
    #     # Normalized encoder output features
    #     # query = F.normalize(query, dim=1)
    #
    #     # update memory items(cluster centers), while encoder parameters being fixed
    #     if self.phase_type != 'test':
    #         self.update(query)
    #
    #     # get new robust features, while memory items(cluster centers) being fixed
    #     outs = self.read(query)
    #
    #     read_query, attn = outs['output'], outs['attn']
    #
    #     if l == 2:
    #         pass
    #     elif l == 3:
    #         read_query = read_query.view(s[0], s[1], 2*s[2])
    #         attn = attn.view(s[0], s[1], self.n_memory)
    #     else:
    #         raise TypeError('Wrong input dimension')
    #     '''
    #     output : N x L x 2C or N x 2C
    #     attn : N x L x M or N x M
    #     '''
    #     return {'output': read_query, 'attn': attn, 'memory_init_embedding':self.mem}

