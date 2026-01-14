from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F


def fourier_adaptive_patching(y_true, delta):
    """
    忠实实现论文 3.3 节的傅里ệt自适应补丁（FAP）。
    Args:
        y_true (torch.Tensor): 真实值张量，形状: [Batch, Seq_Len, Features]
        delta (int): 预定义的、允许的最大补丁长度阈值。
    Returns:
        int: 通过自适应计算得出的补丁长度 P。
    """
    # 为了稳定性，我们对整个 batch 取平均，并以第一个特征通道为代表进行分析
    y_true_mono = y_true.mean(dim=0)[:, 0]  # 形状变为: [Seq_Len]

    T = y_true_mono.shape[0]

    # 1. 快速傅里ệt变换
    fft_result = torch.fft.rfft(y_true_mono)
    amplitudes = torch.abs(fft_result)

    # 2. 找到幅度最高的“主导频率” f (排除直流分量)
    dominant_freq_index = torch.argmax(amplitudes[1:]) + 1

    # 3. 计算主导周期 p = T / f
    period = T / dominant_freq_index.item()

    # 4. 根据论文公式 (3) 确定最终的补丁长度 P
    patch_len = int(min(period, delta))

    # 确保补丁长度至少为2，才有统计意义
    return max(patch_len, 2)


class PSLoss(nn.Module):
    """
    严格按照论文实现的补丁级结构损失 (Lps)。
    Lps = α * Lcorr + β * Lvar + γ * Lmean
    """

    def __init__(self, ps_alpha=1.0, ps_beta=1.0, ps_gamma=1.0):
        super(PSLoss, self).__init__()
        self.ps_alpha = ps_alpha
        self.ps_beta = ps_beta
        self.ps_gamma = ps_gamma
        self.eps = 1e-7  # 用于防止除以零的小常数
        print(f"PSLoss (Lps) 已初始化，权重(α,β,γ)=({ps_alpha},{ps_beta},{ps_gamma})")

    def _pcc(self, y_true_patch, y_pred_patch):
        """ 辅助函数：计算一批补丁的皮尔逊相关系数 (PCC) """
        mean_true = torch.mean(y_true_patch, dim=-1, keepdim=True)
        mean_pred = torch.mean(y_pred_patch, dim=-1, keepdim=True)
        vx = y_true_patch - mean_true
        vy = y_pred_patch - mean_pred
        corr = torch.sum(vx * vy, dim=-1) / (
                    torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + self.eps)
        return corr

    def _calculate_lcorr(self, true_patches, pred_patches):
        """ 对应论文公式 (4)，计算 Lcorr """
        true_patches_flat = true_patches.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
        pred_patches_flat = pred_patches.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
        pcc_values = self._pcc(true_patches_flat, pred_patches_flat)
        l_corr = 1.0 - torch.mean(pcc_values)
        return l_corr

    def _calculate_lvar(self, true_patches, pred_patches):
        """ 对应论文公式 (6)，计算 Lvar """
        dist_true = F.softmax(true_patches, dim=-1)
        dist_pred = F.softmax(pred_patches, dim=-1)
        kl_div = F.kl_div(dist_pred.log(), dist_true, reduction='batchmean')
        return kl_div

    def _calculate_lmean(self, true_patches, pred_patches):
        """ 对应论文公式 (7)，计算 Lmean """
        mean_true = torch.mean(true_patches, dim=-1)
        mean_pred = torch.mean(pred_patches, dim=-1)
        l_mean = F.l1_loss(mean_pred, mean_true)  # MAE
        return l_mean

    def forward(self, y_true, y_pred, patch_len):
        stride = max(patch_len // 2, 1)
        true_patches = y_true.permute(0, 2, 1).unfold(2, patch_len, stride).permute(0, 2, 1, 3)
        pred_patches = y_pred.permute(0, 2, 1).unfold(2, patch_len, stride).permute(0, 2, 1, 3)

        l_corr = self._calculate_lcorr(true_patches, pred_patches)
        l_var = self._calculate_lvar(true_patches, pred_patches)
        l_mean = self._calculate_lmean(true_patches, pred_patches)

        l_ps = self.ps_alpha * l_corr + self.ps_beta * l_var + self.ps_gamma * l_mean

        return l_ps, l_corr, l_var, l_mean


class GatheringLoss(nn.Module):
    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce
        # === 核心修正：明确使用 reduction='none' ===
        # 注意：这里我们使用 self.reduce 来决定是否求平均，但 loss_fn 必须是 reduction='none'
        self.loss_fn = torch.nn.MSELoss(reduction='none')

    def get_score(self, query, key):
        qs = query.size()
        ks = key.size()
        score = torch.matmul(query, torch.t(key))
        score = F.softmax(score, dim=1)
        return score

    def forward(self, queries, items):
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)  # T x C
        score = self.get_score(queries, items)  # T x M

        _, indices = torch.topk(score, 1, dim=1)

        # 此时 gathering_loss 是 (T, C) 形状 (逐元素平方误差)
        gathering_loss = self.loss_fn(queries, items[indices].squeeze(1))

        if self.reduce:
            # 如果要求 reduce，则返回标量 (对所有 T*C 元素求平均)
            return torch.mean(gathering_loss)

            # 否则，返回 (B, N_patch) 形状

        # 1. 对特征 C 求和，得到每个 Token 的损失 (T)
        gathering_loss = torch.sum(gathering_loss, dim=-1)  # (T, C) -> (T)

        # 2. 重塑为 (B, N_patch)
        T = gathering_loss.size(0)
        N_patch = T // batch_size

        # === 关键：重塑为 (B, N_patch) ===
        gathering_loss = gathering_loss.contiguous().view(batch_size, N_patch)

        return gathering_loss  # (B, N_patch)
class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score

    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        # margin from 1.0
        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = items[indices[:, 0]]  # TxC
        neg = items[indices[:, 1]]  # TxC
        anc = queries              # TxC

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss

        spread_loss = spread_loss.contiguous().view(batch_size, -1)       # N x L

        return spread_loss     # N x L
# #这是原文的
# class GatheringLoss(nn.Module):
#     def __init__(self, reduce=True):
#         super(GatheringLoss, self).__init__()
#         self.reduce = reduce
#
#
#     def get_score(self, query, key):
#         '''
#         query : (NxL) x C or N x C -> T x C  (initial latent features)
#         key : M x C     (memory items)
#         '''
#         qs = query.size()
#         ks = key.size()
#
#         score = torch.matmul(query, torch.t(key))   # Fea x Mem^T : (TXC) X (CXM) = TxM
#         score = F.softmax(score, dim=1) # TxM
#
#         return score
#
#     def forward(self, queries, items):
#         '''
#         queries : N x L x C
#         items : M x C
#         '''
#         batch_size = queries.size(0)
#         d_model = queries.size(-1)
#
#         loss_mse = torch.nn.MSELoss(reduce=self.reduce)
#
#         queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
#         score = self.get_score(queries, items)      # TxM
#
#         _, indices = torch.topk(score, 1, dim=1)
#
#         gathering_loss = loss_mse(queries, items[indices].squeeze(1))
#
#         if self.reduce:
#             return gathering_loss
#
#         gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
#         gathering_loss = gathering_loss.contiguous().view(batch_size, -1)   # N x L
#
#         return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score

    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  # TxC
        anc = F.normalize(queries, p=2, dim=-1)               # TxC

        similarity = -1 * torch.sum(pos * anc, dim=-1)         # T
        similarity = similarity.contiguous().view(batch_size, -1)   # N x L

        return similarity     # N x L