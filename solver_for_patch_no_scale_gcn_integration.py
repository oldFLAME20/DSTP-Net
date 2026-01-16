# Some code based on https://github.com/thuml/Anomaly-Transformer
## 这个跟solver_for_patch_gcn_backup.py区别是本文件加上了kmeans超时自动跳过
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import math
from tqdm import tqdm

from model.GNN_Module import GCN_PAD_Model,GCN_PLT_Model
from model.Transformer import TransformerVar,PatchLocalTransformer,PatchLocalBranch
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import auc
from utils.utils import *




def _get_segments(labels):
    """
    将二进制标签序列转换为 (start, end) 的段列表。
    """
    segments = []
    start_idx = -1
    for i, label in enumerate(labels):
        if label == 1 and start_idx == -1:
            start_idx = i
        elif label == 0 and start_idx != -1:
            segments.append((start_idx, i - 1))
            start_idx = -1
    if start_idx != -1:  # 处理最后一个段
        segments.append((start_idx, len(labels) - 1))
    return segments


def calculate_range_based_metrics(gt_labels, pred_labels):
    """
    计算 Range-based Precision, Recall, 和 F-score.
    """
    true_segments = _get_segments(gt_labels)
    pred_segments = _get_segments(pred_labels)

    if not true_segments:
        # 如果没有真实异常，当且仅当没有预测异常时，所有指标为1
        return (1.0, 1.0, 1.0) if not pred_segments else (0.0, 0.0, 0.0)

    if not pred_segments:
        # 如果没有预测异常，但有真实异常，则 precision 为1 (没有误报)，recall 为0
        return (0.0, 1.0, 0.0)

    # 计算 Range Recall
    detected_true_segments = 0
    for t_start, t_end in true_segments:
        is_detected = any(max(p_start, t_start) <= min(p_end, t_end) for p_start, p_end in pred_segments)
        if is_detected:
            detected_true_segments += 1
    r_a_r = detected_true_segments / len(true_segments)

    # 计算 Range Precision
    correct_pred_segments = 0
    for p_start, p_end in pred_segments:
        is_correct = any(max(p_start, t_start) <= min(p_end, t_end) for t_start, t_end in true_segments)
        if is_correct:
            correct_pred_segments += 1
    r_a_p = correct_pred_segments / len(pred_segments)

    # 计算 Range F-score
    if r_a_p + r_a_r == 0:
        r_a_fscore = 0.0
    else:
        r_a_fscore = 2 * r_a_p * r_a_r / (r_a_p + r_a_r)

    return r_a_r, r_a_p, r_a_fscore


def calculate_vus_metrics_fast(gt_labels, scores):
    """
    计算 Volume-based ROC (V_ROC) 和 PR (V_PR) 的高效向量化版本。
    该算法通过一次排序和累积和来避免代价高昂的循环和单调性错误。
    """
    # 确保输入是 numpy 数组
    gt_labels = np.asarray(gt_labels)
    scores = np.asarray(scores)

    # 1. 预计算基本统计量
    n_positives = np.sum(gt_labels == 1)
    n_negatives = len(gt_labels) - n_positives

    if n_positives == 0 or n_negatives == 0:
        return 0.5, (n_positives / len(gt_labels)) if len(gt_labels) > 0 else 0.0

    total_reward = np.sum(scores[gt_labels == 1])
    if total_reward == 0:
        return 0.5, (n_positives / len(gt_labels))

    # 2. 按分数降序排序
    indices = np.argsort(scores)[::-1]
    sorted_gt = gt_labels[indices]
    sorted_scores = scores[indices]

    # 3. 计算累积和
    tp_reward_cum = np.cumsum(sorted_scores * sorted_gt)
    fp_cum = np.cumsum(1 - sorted_gt)

    # 4. 计算 VUS-ROC 曲线的点
    tpr_vus = tp_reward_cum / total_reward
    fpr = fp_cum / n_negatives

    # 5. 计算 VUS-PR 曲线的点
    total_pred_score_mass_cum = np.cumsum(sorted_scores)
    precision_vus = np.divide(tp_reward_cum, total_pred_score_mass_cum,
                              out=np.zeros_like(tp_reward_cum, dtype=float),
                              where=total_pred_score_mass_cum != 0)
    recall_vus = tpr_vus

    # 6. 添加起点以确保曲线从(0,0)开始
    fpr = np.concatenate([[0], fpr])
    tpr_vus = np.concatenate([[0], tpr_vus])

    recall_vus = np.concatenate([[0], recall_vus])
    # PR 曲线的第一个点是 (recall=0, precision=p_at_first_recall)
    # 这种方式比简单地设为1更稳健
    first_precision = precision_vus[0] if len(precision_vus) > 0 and tp_reward_cum[0] > 0 else 1.0
    precision_vus = np.concatenate([[first_precision], precision_vus])

    # 7. 计算曲线下面积
    v_roc = auc(fpr, tpr_vus)
    v_pr = auc(recall_vus, precision_vus)

    return v_roc, v_pr
#这个可能会报错
def calculate_vus_metrics(gt_labels, scores):
    """
    计算 Volume-based ROC (V_ROC) 和 PR (V_PR).
    这是一个简化的、但能反映核心思想的实现。
    """
    true_segments = _get_segments(gt_labels)
    if not true_segments:
        return 1.0, 1.0  # 如果没有异常，VUS 指标没有意义，返回完美分数

    # 从 scores 中分离出正常和异常部分
    normal_scores = scores[gt_labels == 0]
    anomaly_scores = scores[gt_labels == 1]

    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        # 如果缺少任一类别，无法计算AUC，返回默认值
        return 0.5, 0.5

    # 为每个异常段计算一个权重，奖励早期检测
    # 这里的权重是一个简单的例子：段内分数的总和
    segment_rewards = []
    for start, end in true_segments:
        segment_score = np.sum(scores[start:end + 1])
        segment_rewards.append(segment_score)
    total_reward = np.sum(segment_rewards)

    # 生成 VUS-ROC 曲线点
    roc_x, roc_y = [0], [0]
    pr_x, pr_y = [0], [1]  # PR曲线从 (0,1) 开始

    # 合并所有分数并排序，以生成阈值
    all_scores = np.unique(np.sort(scores))

    for thresh in np.flip(all_scores):  # 从高到低遍历阈值
        # 计算 False Positive Rate
        fp = np.sum(normal_scores >= thresh)
        fpr = fp / len(normal_scores) if len(normal_scores) > 0 else 0

        # 计算 VUS-style True Positive Rate / Recall
        tp_reward = 0
        for start, end in true_segments:
            tp_reward += np.sum(scores[start:end + 1][scores[start:end + 1] >= thresh])
        tpr_vus = tp_reward / total_reward if total_reward > 0 else 0

        roc_x.append(fpr)
        roc_y.append(tpr_vus)

        # 计算 VUS-style Precision
        total_detected_score_mass = np.sum(scores[scores >= thresh])
        precision_vus = tp_reward / total_detected_score_mass if total_detected_score_mass > 0 else 0

        pr_x.append(tpr_vus)  # Recall
        pr_y.append(precision_vus)  # Precision

    roc_x.append(1);
    roc_y.append(1)
    pr_x.append(1);
    pr_y.append(pr_y[-1])

    # 计算曲线下面积
    v_roc = auc(roc_x, roc_y)
    v_pr = auc(pr_x, pr_y)

    return v_roc, v_pr


# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1,2, 3'

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_features(queries_tensor, method='pca', num_points=10000, save_path='feature_map.png'):
    """
    使用 PCA 或 t-SNE 将高维 queries 投影到 2D 并绘图。

    queries_tensor: 展平后的 Queries Tensor (N_total, D)
    method: 'pca' 或 'tsne'
    num_points: 用于绘图的最大样本数量
    save_path: 图像保存路径
    """

    # 确保数据在 CPU 上并转换为 NumPy
    queries_np = queries_tensor.detach().cpu().numpy()

    N, D = queries_np.shape

    if N > num_points:
        # 随机采样，避免 t-SNE 耗时过长
        indices = np.random.choice(N, num_points, replace=False)
        data_to_plot = queries_np[indices]
    else:
        data_to_plot = queries_np

    print(f"Visualizing {data_to_plot.shape[0]} points using {method}...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_plot)  # <--- 必须在这里计算
    if method == 'pca':
        if D < 2:
            print("Error: Feature dimension too low for PCA.")
            return
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(data_to_plot)
        plt.title(f'PCA of Encoded Queries (Variance Explained: {np.sum(reducer.explained_variance_ratio_):.2f})')

    elif method == 'tsne':
        # --- 修正点 1：高维降维到 50 维 (PCA Pre-reduction) ---
        D_original = scaled_data.shape[1]
        if D_original > 50:
            print(f"Applying PCA pre-reduction from {D_original}D to 50D...")
            pca_pre = PCA(n_components=50, random_state=42)
            data_for_tsne = pca_pre.fit_transform(scaled_data)
        else:
            data_for_tsne = scaled_data

        # --- 修正点 2：调用 t-SNE ---
        # 增加迭代次数以提高收敛稳定性
        reducer = TSNE(n_components=2,
                       random_state=42,
                       n_jobs=4,
                       learning_rate=200,  # 常用默认值或手动调整
                       init='pca',
                       n_iter=1000)  # 确保有足够迭代次数

        reduced_data = reducer.fit_transform(data_for_tsne)
        plt.title('t-SNE of Encoded Queries (Pre-reduced)')

    else:
        raise ValueError("Invalid visualization method. Use 'pca' or 'tsne'.")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        palette="viridis",
        legend=False,
        alpha=0.6,
        s=10
    )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path)
    plt.close()
    print(f"Feature map saved to {save_path}")
  # 根据数据集类型导入不同的工具模块

class Solver(object):

    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        print(f"Solver master device set to: {self.device}")

        self._get_data()

        self.memory_initial = str(self.memory_initial).lower() == 'true'
        self.memory_init_embeddings = None

        self.build_model()
###这部分
        self.criterion = nn.MSELoss()
        self.entropy_loss = EntropyLoss()
        self.match_loss_gcn = GatheringLoss()
        self.sse_criterion = nn.MSELoss(reduction='sum')



        self._setup_logging()


    def _get_data(self):
        if self.mode != 'test':
            print(f"Loading data for train mode, dataset: {self.dataset}")
            self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='train',
                dataset=self.dataset
            )
            self.thre_loader = self.vali_loader
        else:  # self.mode == 'test'
            print(f"Loading data for test mode, dataset: {self.dataset}")
            # 1. 加载训练、验证 loader for thresholding
            # 这一部分是正确的，因为 mode='train' 返回3个值
            self.train_loader, self.thre_loader, _ = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='train',
                dataset=self.dataset
            )

            # 2. 加载测试 loader
            # ==========================================================
            # === 核心修改：将解包变量从3个改为2个 ===
            # ==========================================================
            self.test_loader, _ = get_loader_segment(
                data_path=self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                mode='test',
                dataset=self.dataset
            )
            # ==========================================================
            # === 修改结束 ===
            # ==========================================================
    def _setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def build_model(self):

        self.model = GCN_PLT_Model(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            n_heads=self.n_heads if hasattr(self, 'n_heads') else 8,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation='gelu',

            # GCN 特定参数
            n_memory_gcn=self.n_memory,
            top_k=self.topk if hasattr(self, 'topk') else 15,
            d_graph_embed=self.d_graph_embed if hasattr(self, 'd_graph_embed') else 64,

            # Patch Local Transformer 参数
            patch_len=self.patch_len if hasattr(self, 'patch_len') else (self.win_size // 10),
            n_memory_patch=self.n_memory_patch,
            shrink_thres=self.shrink_thres if hasattr(self, 'shrink_thres') else 0.0025,

            # 融合参数
            gamma=self.gamma if hasattr(self, 'gamma') else 0.5,

            # 状态和设备
            device=self.device,
            phase_type=self.phase_type,
            dataset_name=self.dataset,
            memory_initial=self.memory_initial,
            memory_init_embeddings=self.memory_init_embeddings
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    # def _calculate_loss(self, output_dict, input):
    #     # 1. 重构损失 L_rec (使用 final_output)
    #     sse = self.sse_criterion(output_dict['final_output'], input)
    #     L_rec = torch.sqrt(sse)  # 弗罗贝尼乌斯范数
    #
    #     # 2. Patch 熵损失 L_ent (使用 patch_attns)
    #     L_ent = 0
    #     if 'patch_attns' in output_dict and output_dict['patch_attns']:
    #         total_entropy_sum = 0
    #         # PatchLocalBranch 返回单尺度，键为 1
    #         attn_z = output_dict['patch_attns'][1]  # <--- 适配单尺度 Patch 分支
    #         entropy_map = -attn_z * torch.log(attn_z + 1e-12)
    #         total_entropy_sum = torch.sum(entropy_map)
    #         L_ent = total_entropy_sum
    #
    #     # 3. GCN 匹配损失 L_gcn (使用 gcn_queries 和 gcn_attn/gcn_mem)
    #     L_gcn = 0
    #     if 'gcn_queries' in output_dict:
    #         # L_gcn 逻辑保持不变 (它基于 GCN queries 和 GCN mem)
    #         gcn_queries_flat = output_dict['gcn_queries'].contiguous().view(-1, self.d_model)
    #         gcn_mem = output_dict['gcn_mem']
    #
    #         # 使用 GCN Attn 找到最近原型
    #         gcn_attn_flat = output_dict['gcn_attn'].contiguous().view(-1, self.n_memory)
    #         _, top_indices = torch.topk(gcn_attn_flat, 1, dim=-1)
    #         gcn_mem_on_device = gcn_mem.to(top_indices.device)
    #         nearest_mems = gcn_mem_on_device[top_indices.squeeze(1)]
    #
    #         # L2 距离之和
    #         distances = torch.norm(gcn_queries_flat - nearest_mems, p=2, dim=1)
    #         L_gcn = torch.sum(distances)
    #
    #     # 4. 总损失 LOSS (使用 a1, a2, a3)
    #     total_loss = self.a1 * L_rec + self.a2 * L_ent + self.a3 * L_gcn
    #
    #     return total_loss, L_rec.item(), L_ent.item(), L_gcn.item()
    ####//这部分是可以实现的可以跑的 todo
    # def _calculate_loss(self, output_dict, input):
    #     # 1. 重构损失 L_rec (使用 final_output)
    #     sse = self.sse_criterion(output_dict['final_output'], input)
    #     L_rec = torch.sqrt(sse)  # 弗罗贝尼乌斯范数
    #
    #     # 2. Patch 熵损失 L_ent (使用 patch_attns)
    #     L_ent = 0
    #     if 'patch_attns' in output_dict and output_dict['patch_attns']:
    #         total_entropy_sum = 0
    #         # PatchLocalBranch 返回单尺度，键为 1
    #         attn_z = output_dict['patch_attns'][1]  # <--- 适配单尺度 Patch 分支
    #         entropy_map = -attn_z * torch.log(attn_z + 1e-12)
    #         total_entropy_sum = torch.sum(entropy_map)
    #         L_ent = total_entropy_sum
    #
    #     # 3. GCN 匹配损失 L_gcn (使用 gcn_queries 和 gcn_attn/gcn_mem)
    #     L_gcn = 0
    #     if 'gcn_queries' in output_dict:
    #         # L_gcn 逻辑保持不变 (它基于 GCN queries 和 GCN mem)
    #         gcn_queries_flat = output_dict['gcn_queries'].contiguous().view(-1, self.d_model)
    #         gcn_mem = output_dict['gcn_mem']
    #
    #         # 使用 GCN Attn 找到最近原型
    #         gcn_attn_flat = output_dict['gcn_attn'].contiguous().view(-1, self.n_memory)
    #         _, top_indices = torch.topk(gcn_attn_flat, 1, dim=-1)
    #         gcn_mem_on_device = gcn_mem.to(top_indices.device)
    #         nearest_mems = gcn_mem_on_device[top_indices.squeeze(1)]
    #
    #         # L2 距离之和
    #         distances = torch.norm(gcn_queries_flat - nearest_mems, p=2, dim=1)
    #         L_gcn = torch.sum(distances)
    #
    #     # 4. 总损失 LOSS (使用 a1, a2, a3)
    #     total_loss = self.a1 * L_rec + self.a2 * L_ent + self.a3 * L_gcn
    #
    #     return total_loss, L_rec.item(), L_ent.item(), L_gcn.item()

    def _calculate_loss(self, output_dict, input):

        # 1. 重构损失 L_rec (使用 final_output)
        sse = self.sse_criterion(output_dict['final_output'], input)
        L_rec = torch.sqrt(sse)  # 弗罗贝尼乌斯范数

        # 2. Patch 熵损失 L_ent (使用 patch_attns)
        L_ent = 0
        if 'patch_attns' in output_dict and output_dict['patch_attns']:
            total_entropy_sum = 0
            # PatchLocalBranch 返回单尺度，键为 1
            attn_z = output_dict['patch_attns'][1]  # <--- 适配单尺度 Patch 分支
            entropy_map = -attn_z * torch.log(attn_z + 1e-12)
            total_entropy_sum = torch.sum(entropy_map)
            L_ent = total_entropy_sum


        # 3. GCN 熵损失 L_gcn_ent
        L_gcn_ent = 0

        if 'gcn_attn' in output_dict:
            total_entropy_sum_gcn = 0
            # 假设 EntropyLoss 现在返回总和
            attn_gcn = output_dict['gcn_attn'] # <--- 适配单尺度 Patch 分支
            total_entropy_sum_gcn = -attn_gcn * torch.log(attn_gcn + 1e-12)
            total_entropy_sum= torch.sum(total_entropy_sum_gcn)
            L_gcn_ent = total_entropy_sum   # 批次平均

        # 4. 总损失 LOSS (PyTorch Tensor)
        total_loss = self.a1 * L_rec + self.a2 * L_ent + self.a3 * L_gcn_ent

        # 5. 返回 Tensor 总损失，以及 Float 类型的分项损失 (用于日志记录)
        return total_loss, L_rec.item(), L_ent.item(), L_gcn_ent.item()
    def vali(self, vali_loader):
        self.model.eval()
        losses, recs, ents, gcns = [], [], [], []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output_dict = self.model(input)
                loss, rec, ent, gcn = self._calculate_loss(output_dict, input)
                losses.append(loss.item())
                recs.append(rec)
                ents.append(ent)
                gcns.append(gcn)
        return np.average(losses), np.average(recs), np.average(ents), np.average(gcns),

    def train(self, training_type):


        print(f"====================== TRAIN MODE: {training_type} ======================")
        torch.autograd.set_detect_anomaly(True)
        path = self.model_save_path
        os.makedirs(path, exist_ok=True)
        early_stopping = OneEarlyStopping(patience=10, verbose=True, dataset_name=self.dataset, type=training_type)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses, epoch_recs, epoch_ents, epoch_gcns = [], [], [], []
            epoch_start_time = time.time()
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            for i, (input_data, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                self.optimizer.zero_grad()
                input = input_data.float().to(self.device)



                output_dict = self.model(input)
                loss, rec, ent, gcn = self._calculate_loss(output_dict, input)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                epoch_recs.append(rec)
                epoch_ents.append(ent)
                epoch_gcns.append(gcn)

            train_loss, train_rec,train_ent, train_gcn = np.average(epoch_losses), np.average(epoch_recs), np.average(
                epoch_ents), np.average(epoch_gcns)
            valid_loss, valid_rec, valid_ent, valid_gcn = self.vali(self.vali_loader)
            # <<< MODIFICATION START: 记录 epoch 结束时间并计算耗时 >>>
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"\nEpoch: {epoch + 1}/{self.num_epochs}")
            print(
                f"  Train | Loss: {train_loss:.7f} | Rec: {train_rec:.7f} | Ent: {train_ent:.7f} | gcn_Match: {train_gcn:.7f}")
            print(
                f"  Valid | Loss: {valid_loss:.7f} | Rec: {valid_rec:.7f} | Ent: {valid_ent:.7f} | gcn_Match: {valid_gcn:.7f}")
            print(
                f"   Time  | Epoch duration: {epoch_duration:.2f} seconds")
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model_to_get_mems = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # final_mems = {
        #     'gcn': model_to_get_mems.gcn_branch_memory.mem.detach().cpu(),
        #     'patch': {f"patch_z{z}": model_to_get_mems.patch_branch.memories[i].mem.detach().cpu()
        #               for i, z in enumerate(model_to_get_mems.patch_branch.patch_scales)}
        # }
        # return final_mems

        ## 核心修改
        final_mems = {}

        # final_mems 必须以 GCN/Patch 期望的结构返回
        final_mems = {'gcn': {}, 'patch': {}}

        # === 1. 收集 GCN Memory ===
        if hasattr(model_to_get_mems, 'gcn_branch_memory'):
            print("Saving gcn branch memory...")
            # 键名必须是 'gcn'
            final_mems['gcn'] = model_to_get_mems.gcn_branch_memory.mem.detach().cpu()

        # === 2. 收集 Patch Memory (单尺度 1) ===
        if hasattr(model_to_get_mems, 'plt_branch'):
            print("Saving Patch branch memories...")

            # final_mems['patch'] 必须是一个字典
            final_mems['patch'] = {}
            # 假设 PatchLocalBranch 有一个 patch_scales 属性
            patch_scales = model_to_get_mems.plt_branch.patch_scales

            # 假设 memories 列表 (或等效结构) 位于 plt_branch
            if hasattr(model_to_get_mems.plt_branch, 'memories'):
                memories_list = model_to_get_mems.plt_branch.memories
            elif hasattr(model_to_get_mems.plt_branch, 'mem_module'):
                # 如果是直接的 mem_module (单实例)
                memories_list = [model_to_get_mems.plt_branch.mem_module]
            else:
                # 尝试从 GCN_PLT_Model.initialize_memories 所示的结构中提取
                memories_list = [model_to_get_mems.plt_branch.plt_model.mem_module]  # 如果结构嵌套

            # 遍历 Patch 分支的 memories
            for i, z in enumerate(patch_scales):
                if i < len(memories_list):
                    # 使用数字键 z (即 1)
                    final_mems['patch'][z] = memories_list[i].mem.detach().cpu()
                else:
                    print("Warning: Missing memory instance for scale.")
        return final_mems

    def test(self):

        # --- 初始化和参数获取 ---
        criterion = nn.MSELoss(reduction='none')
        gathering_fn = self.match_loss_gcn  # 假设 GatheringLoss 存储在这里
        temperature = self.temperature if hasattr(self, 'temperature') else 1.0

        # =====================================================================
        # === 1. 加载模型和获取 Patch 参数 ===
        # =====================================================================
        model_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # 确保加载到正确设备
        self.model.eval()

        print("====================== TEST MODE ======================")

        # 从模型中获取 Patch 参数 (在加载 state dict 后)
        model_to_get_params = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        L = self.win_size
        P_len = model_to_get_params.patch_len
        N_patch = model_to_get_params.num_patches

        # -----------------------

        def calculate_anomaly_score(output_dict, input):
            N, L, C = input.shape

            # --- a. 重构误差 sr(t) ---
            sr = torch.mean(criterion(output_dict['final_output'], input), dim=-1)  # Sr: [N, L]

            # 提取 Patch Queries Tensor
            # 注意：这里我们假设单尺度键是 1
            Q_points_dict = output_dict['patch_queries']
            if 1 in Q_points_dict:
                Q_points = Q_points_dict[1]  # Q_points shape: [N, N_patch, D]
            else:
                # 如果键是别的，例如 'queries' 或 'point_queries'，则需要适应
                # 基于您提供的 GCN_PLT_Model.forward 结构，键是 1
                raise KeyError("Patch queries dictionary missing expected key 1.")

            mem_z = output_dict['patch_mems'][1]

            # Q_points_flat 形状: [N * N_patch, D]
            Q_points_flat = Q_points.reshape(-1, self.d_model)
            mem_z_on_device = mem_z.to(Q_points_flat.device)

            # 计算最小距离 min_dists 形状: [480]
            dists = torch.cdist(Q_points_flat, mem_z_on_device)
            min_dists, _ = torch.min(dists, dim=1)

            # 1. 重塑为 Token 级偏差: [N, N_patch]
            N_patch_actual = Q_points.shape[1]  # 15
            sz_token = min_dists.view(N, N_patch_actual)  # [32, 15]

            # 2. 上采样回点级 L=105
            P_len = L // N_patch_actual  # 105 // 15 = 7
            sz = sz_token.unsqueeze(-1).repeat(1, 1, P_len).reshape(N, L)  # Sz: [N, L]

            # --- c. GCN 分支偏差 sg(t) ---
            gcn_queries = output_dict['gcn_queries']  # [N, L, D]
            gcn_mem = output_dict['gcn_mem']  # [N_mem, D]

            gcn_queries_flat = gcn_queries.reshape(-1, self.d_model)
            gcn_mem_on_device = gcn_mem.to(gcn_queries_flat.device)

            # 计算每个点到最近 GCN 原型的 L2 距离
            dists_gcn = torch.cdist(gcn_queries_flat, gcn_mem_on_device)
            min_dists_gcn, _ = torch.min(dists_gcn, dim=1)
            sg = min_dists_gcn.view(N, L)  # Sg: [N, L]
            # print("sz的大小",sz)
            # print("sg的大小",sg)
            # --- d. 最终分数 (H-PAD 融合风格) --- beta_list=(0 0.1 0.5 0.01 0.05 0.001 0.005 0.0001 0.0005)
            # feature_score 融合了 Patch 偏差和 GCN 偏差
            feature_score = sz + self.beta * sg

            # 最终分数：使用特征偏差加权的重构误差
            # 注意：这里我们使用 softmax(feature_score) 作为权重，这遵循您原始代码的乘法结构。
            final_score = torch.softmax(feature_score, dim=-1) * sr

            return final_score
        # =====================================================================
        # === 3. 循环调用评分函数 ===
        # =====================================================================

        # --- A. 训练集评分 (用于阈值) ---
        train_attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)  # input_data 已在循环开始时转换为 input

            anomaly_score = calculate_anomaly_score(output_dict, input)
            cri = anomaly_score.detach().cpu().numpy()
            train_attens_energy.append(cri)

        train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
        train_energy = np.array(train_attens_energy)
        # =====================================================================
        # === 3. 在训练集和验证集上计算分数，以确定异常阈值 ===
        # =====================================================================
        print("Calculating threshold on training and validation data...")
        all_scores = []
        # 使用 self.train_loader 和 self.thre_loader (即 vali_loader)
        for loader in [self.train_loader, self.thre_loader]:
            for i, (input_data, labels) in enumerate(tqdm(loader, desc="Thresholding")):
                input = input_data.float().to(self.device)
                with torch.no_grad():
                    output_dict = self.model(input)
                    anomaly_score = calculate_anomaly_score(output_dict, input)
                    all_scores.append(anomaly_score.detach().cpu().numpy())

        all_scores = np.concatenate(all_scores, axis=0).flatten()
        # 根据设定的异常比例确定阈值
        thresh = np.percentile(all_scores, 100 - self.anormly_ratio)
        print(f"Threshold determined: {thresh:.7f}")

        # =====================================================================
        # === 4. 在测试集上计算最终分数并进行预测 ===
        # =====================================================================
        print("Evaluating on test data...")
        test_labels_list = []
        test_scores_list = []
        for i, (input_data, labels) in enumerate(tqdm(self.test_loader, desc="Testing")):
            input = input_data.float().to(self.device)
            with torch.no_grad():
                output_dict = self.model(input)
                anomaly_score = calculate_anomaly_score(output_dict, input)
                test_scores_list.append(anomaly_score.detach().cpu().numpy())
                test_labels_list.append(labels.numpy())

        test_scores = np.concatenate(test_scores_list, axis=0).flatten()
        test_labels = np.concatenate(test_labels_list, axis=0).flatten()

        # =====================================================================
        # === 5. 评估预测结果 (这部分与您之前的代码逻辑一致) ===
        # =====================================================================
        pred = (test_scores > thresh).astype(int)
        gt = test_labels.astype(int)

        # 针对WADI数据集的特殊处理
        if self.dataset == 'WADI':
            gt = np.zeros_like(test_labels, dtype=int)
            gt[test_labels == -1] = 1

        # PA (Point-Adjustment) 调整逻辑
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        print("pred shape: ", pred.shape)
        print("gt shape:   ", gt.shape)

        # 指标计算
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        # 注意：AUC 指标应该使用未经阈值处理的原始分数 test_scores
        auc_roc = roc_auc_score(gt, test_scores)
        auc_pr = average_precision_score(gt, test_scores)

        print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")
        print(f"AUC-ROC  : {auc_roc:.4f}, AUC-PR   : {auc_pr:.4f}")
        print('=' * 50)
        # =====================================================================
        # === 新增：计算 R_A_R, R_A_P, V_ROC, V_PR ===
        # =====================================================================
        print("\nCalculating advanced time-series metrics...")

        # Range-based metrics 使用阈值化后的 pred 和 gt
        r_a_r, r_a_p, r_a_fscore = calculate_range_based_metrics(gt, pred)

        # VUS-based metrics 使用原始 scores 和 gt
        v_roc, v_pr = calculate_vus_metrics_fast(gt, test_scores)
        # =====================================================================

        print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")
        print(f"AUC-ROC  : {auc_roc:.4f}, AUC-PR   : {auc_pr:.4f}")

        # 新增打印
        print(f"R_A_R    : {r_a_r:.4f}, R_A_P     : {r_a_p:.4f}, R_A_F-score: {r_a_fscore:.4f}")
        print(f"V_ROC    : {v_roc:.4f}, V_PR      : {v_pr:.4f}")
        print('=' * 50)

        # # === 修改：将新指标传递给日志记录函数 ===
        self.record_results(accuracy, precision, recall, f_score, auc_roc,
                            auc_pr, r_a_r, r_a_p, r_a_fscore, v_roc,v_pr)
        # 日志记录...
        # self.record_results(accuracy, precision, recall, f_score, auc_roc, auc_pr)

        return accuracy, precision, recall, f_score, auc_roc, auc_pr

        ## record_results 自己增加的用来将结果输出到res_"数据集名.md中"  //todo
    def record_results(self, accuracy, precision, recall, f_score, auc_roc=None, auc_pr=None,r_a_r=None,r_a_p=None,r_a_fscore=None,v_roc=None,v_pr=None):
            """
            将实验结果附加到Markdown文件 # Range-based metrics 使用阈值化后的 pred 和 gt
        r_a_r, r_a_p, r_a_fscore = calculate_range_based_metrics(gt, pred)

        # VUS-based metrics 使用原始 scores 和 gt
        v_roc, v_pr = calculate_vus_metrics_fast(gt, test_scores)
            """
            # 创建结果文件路径

            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_no_scale.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_no_scale_integration.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_no_scale_integration_patch_flat_seed{self.seed}_patchLen{self.patch_len}_MLP8_.md"
            result_file = f"./res/{self.dataset}/res_{self.dataset}_main_seed{self.seed}_patchLen{self.patch_len}_MLP8_memory_{self.n_memory}_{self.n_memory_patch}_topk{self.topk}.md"


            # result_file = f"./res/{self.dataset}/res_{self.dataset}_seed{self.seed}_patchLen{self.patch_len}_MLP8_dmodel{self.d_model}_topk{self.topk}_dff{self.d_ff}_.md"


            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_no_scale_integration_patch_flat_no_MLP_seed{self.seed}_patchLen{self.patch_len}_dmodel{self.d_model}_dff{self.d_ff}.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_gamma_gate.md"
            # result_file = f"./res/{self.dataset}/res_{self.dataset}_gcn_patch_ablation.md"

            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            # 构建结果内容
            result_content = f"""## {self.dataset}_gcn_patch_no_scale_integration_patch_flat_seed{self.seed}_patchLen{self.patch_len}_memory_{self.n_memory}_topk{self.topk}_MLP8_dmodel{self.d_model}_dff{self.d_ff}"
        -==参数如下
        ==seed :{self.seed}  lamda:{self.lambd},dmodel:{self.d_model},d_ff:{self.d_ff},topK{self.topk}
        a1= {self.a1}, a2= {self.a2}, a3= {self.a3},    beta = {self.beta} ,patch_len={self.patch_len}
        n_memory = {self.n_memory},n_memory_patch = {self.n_memory_patch}, patch_scales= {self.patch_scales}==-
        - =================================================
        - **Accuracy**: {accuracy:.4f}
        - **Precision**: {precision:.4f}
        - **Recall**: {recall:.4f}
        - **F-score**: {f_score:.4f}
        - **AUC-ROC**: {auc_roc:.4f}
        - **AUC-PR**: {auc_pr:.4f}
        - **R_A_R**: {r_a_r:.4f}
        - **R_A_P**: {r_a_p:.4f}
        - **R_A_F-score**: {r_a_fscore:.4f}
        - **V_ROC**: {v_roc:.4f}
        - **V_PR**: {v_pr:.4f}
        - **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}

        ---

        """

            # 附加结果到文件
            try:
                with open(result_file, 'a', encoding='utf-8') as f:
                    f.write(result_content)
                self.logger.info(f"实验结果已附加到 {result_file}")
            except Exception as e:
                self.logger.error(f"写入结果文件时出错: {e}")
                print(f"写入结果文件时出错: {e}")
    def get_memory_initial_embedding(self, training_type='second_train'):
        # if self.dataset == 'WADI':
        #     from utils.utils_bank import k_means_clustering

        try:
            self.model.load_state_dict(
                torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth'))
            )
            self.model.eval()



            print("Collecting queries for all branches...")
            gcn_queries_list, patch_queries_lists = [], {1: []}  # Patch 分支现在只有尺度 1

            temp_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

            for i, (input_data, labels) in enumerate(tqdm(self.k_loader, desc="Collecting Queries")):
                input = input_data.float().to(self.device)
                with torch.no_grad():
                    output_dict = temp_model.get_all_queries_for_kmeans(input)

                    # GCN Queries (B, L, D)
                    gcn_queries_list.append(output_dict['gcn_queries'].cpu())

                    # Patch Queries (单尺度 1)
                    patch_queries_lists[1].append(output_dict['patch_queries'][1].cpu())

            self.memory_init_embeddings = {'patch': {}}

            # =======================================================
            # === 修正：在保存文件之前，创建 features 目录 ===
            # =======================================================
            feature_dir = "./features"
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
                print(f"Created feature directory: {feature_dir}")
            # === GCN K-Means ===
            gcn_queries_all = torch.cat(gcn_queries_list, dim=0)
            # =========================================================
            # === 诊断步骤：可视化特征空间 ===
            # =========================================================

            # 1. GCN Queries 可视化 (通常使用 PCA，因为它更快)
            gcn_queries_flat_for_vis = gcn_queries_all.reshape(-1, self.d_model)
            visualize_features(
                gcn_queries_flat_for_vis,
                method='pca',
                num_points=10000,
                save_path=f"./features/{self.dataset}_GCN_Queries_PCA.png"
            )



            self.memory_init_embeddings['gcn'] = k_means_clustering(gcn_queries_all.reshape(-1, self.d_model),
                                                                    self.n_memory, self.d_model)
            print(f"GCN memory initialized with shape: {self.memory_init_embeddings['gcn'].shape}")

            # === Patch K-Means (单尺度 1) ===
            patch_queries_all = torch.cat(patch_queries_lists[1], dim=0)
            # 2. Patch Queries 可视化 (如果 N_total 允许，可以使用 t-SNE，但 PCA 应该足够诊断)
            patch_queries_all_flat_for_vis = patch_queries_all.reshape(-1, self.d_model)
            visualize_features(
                patch_queries_all_flat_for_vis,
                method='pca',  # 优先使用 PCA
                num_points=10000,
                save_path=f"./features/{self.dataset}_{self.patch_len}_Patch_Queries_PCA.png"
            )

            # =========================================================

            # 确保文件夹存在
            os.makedirs("./features", exist_ok=True)
            self.memory_init_embeddings['patch'] = {
                1: k_means_clustering(
                    patch_queries_all.reshape(-1, self.d_model),
                    self.n_memory_patch if hasattr(self, 'n_memory_patch') else self.n_memory,
                    self.d_model
                )
            }
            print(f"Patch (z=1) memory initialized with shape: {self.memory_init_embeddings['patch'][1].shape}")

            self.memory_initial = True  # 保持为 True
            self.phase_type = 'second_train'
            self.build_model()  # 重新构建模型，注入原型

            print("Starting second stage training...")
            final_mems = self.train(training_type='second_train')

            # ... (Memory 保存逻辑，适配 GCN 和 Patch 单尺度) ...
            item_folder_path = "memory_item"
            os.makedirs(item_folder_path, exist_ok=True)

            # GCN 保存
            torch.save(final_mems['gcn'], os.path.join(item_folder_path, f"{self.dataset}_gcn_memory_item.pth"))

            # === 修正 Patch (单尺度 1) 保存逻辑 ===

            # 1. 安全获取 Patch 尺度
            if not hasattr(temp_model, 'plt_branch'):
                raise AttributeError("Model is missing the 'patch_branch' attribute for saving.")

            # 假设 PatchBranch 结构中 patch_scales 是 [1]
            PATCH_SCALE_Z = temp_model.plt_branch.patch_scales[0]
            PATCH_MEM_KEY = f"patch_z{PATCH_SCALE_Z}"

            # 2. 保存 Patch Memory
            if 'patch' in final_mems and PATCH_SCALE_Z in final_mems['patch']:
                # final_mems['patch'] 使用了数字键 [1] (来自 Solver.train)
                memory_to_save = final_mems['patch'][PATCH_SCALE_Z]

                # 文件名使用字符串键
                torch.save(memory_to_save,
                           os.path.join(item_folder_path, f"{self.dataset}_{PATCH_MEM_KEY}_memory_item.pth"))
                print(f"Patch ({PATCH_MEM_KEY}) memory item saved.")
            else:
                # 如果 final_mems['patch'] 的键结构不匹配 (即缺少数字键 1)
                raise KeyError(
                    f"Failed to retrieve Patch memory. Expected key '{PATCH_SCALE_Z}' not found in final_mems['patch']. Keys found: {final_mems['patch'].keys()}"
                )

            return True

        except TimeoutException:
            print("Skipping the rest of this trial due to K-Means timeout.")
            return False