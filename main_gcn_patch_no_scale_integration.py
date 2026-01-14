import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # 可能有帮助

# =================================================================
# ===              针对 torch_geometric 的最终 HACK             ===
# =================================================================
# 尝试在导入前设置这个环境变量，可能会改变 torch_geometric 的后端选择
# os.environ['TORCH_SCATTER_REDUCE'] = 'sum'
# =================================================================
import argparse
import torch

from torch.backends import cudnn
from utils.utils import *

# ours
from solver_for_patch_no_scale_gcn_integration import Solver

import  random

import sys # 确保导入


# ==========================================================
# ===             在这里添加设置种子的函数             ===
# ==========================================================
def set_seed(seed):
    """
    设置随机种子以确保实验的可复现性。
    """


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果你使用了CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU情况
        # 为了完全的可复现性，需要牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # 增加以下设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
# ==========================================================

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = False
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train(training_type='first_train')
    elif config.mode == 'test':

        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')  ###//todo 动态memory

    return solver


if __name__ == '__main__':
    cudnn.benchmark = False

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--win_size', type=int, default=105)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temp_param',type=float, default=0.05)
    parser.add_argument('--lambd',type=float, default=0.01)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='./data/SMD/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.0)
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--n_memory', type=int, default=5, help='number of memory items')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--temperature', type=int, default=0.1)
    parser.add_argument('--memory_initial',type=str, default=False, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--phase_type',type=str, default=None, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--topk',type=int, default=5, help='graph adjacency matrix topk')
    #

    parser.add_argument("--d_graph_embed", type=int, default=64, help="Embedding dimension for graph construction")

    # --- Patch Branch Params ---
    # 使用 nargs='+' 允许多个值, e.g., --patch_scales 1 2 4
    parser.add_argument('--patch_scales', type=int, nargs='+', default=[1], help='List of patch scales')
    parser.add_argument("--n_memory_patch", type=int, default=10, help="Number of memory items per patch scale")
    parser.add_argument("--e_layers", type=int, default=1, help="number of encoder layers") # <<< 关键：添加这个参数
    # <<< 关键：添加 d_ff 这个参数 >>>
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of feedforward network")
    parser.add_argument("--patch_len", type=int, default=7, help="dimension of feedforward network(3,5,7)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")


    # --- Fusion & Loss Params (as per H-PAD) ---
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight for fusing branch reconstructions")
    parser.add_argument("--beta", type=float, default=20.0, help="Weight for GAT score in anomaly score fusion")
    parser.add_argument("--a1", type=float, default=2.0, help="Weight for reconstruction loss (L_rec)")
    parser.add_argument("--a2", type=float, default=1.0, help="Weight for entropy loss (L_ent)")
    parser.add_argument("--a3", type=float, default=1.0,  help="Weight for GAT matching loss (L_gat)")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for anomaly score fusion")

    parser.add_argument('--seed',type=int, default=43, help='seed for train')


    ## 结束
    config = parser.parse_args()
    set_seed(config.seed)
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
