import numpy as np
import torch
from tqdm import tqdm
import signal
import time


# 1. 超时异常类
class TimeoutException(Exception):
    pass


# 2. 超时处理函数
def _timeout_handler(signum, frame):
    raise TimeoutException("K-Means clustering timed out.")


# 3. 保存原始的initialize和pairwise函数（如果需要）
from kmeans_pytorch import kmeans, initialize, pairwise_distance, pairwise_cosine


# 4. 创建带迭代限制的新版本
def kmeans_with_limit(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu'),
        max_iter=1000  # 新增：最大迭代次数
):
    """
    带迭代次数限制的kmeans版本
    """
    print(f'Running k-means on {device} with max_iter={max_iter}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')

    while True:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            if selected.dim() == 0:
                selected = selected.unsqueeze(0)

            selected = torch.index_select(X, 0, selected)

            if len(selected) > 0:
                initial_state[index] = selected.mean(dim=0)
            else:
                # 处理空簇：随机选择一个点作为新中心
                rand_idx = torch.randint(len(X), (1,), device=device)
                initial_state[index] = X[rand_idx]

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        iteration += 1

        # 检查迭代次数限制
        if iteration >= max_iter:
            print(f"\n⚠️ Reached maximum iterations ({max_iter})")
            tqdm_meter.close()
            # 即使没收敛也返回结果
            return choice_cluster.cpu(), initial_state.cpu()

        # 更新进度条
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}',
            max_iter=f'{max_iter}'
        )
        tqdm_meter.update()

        # 检查收敛
        if center_shift ** 2 < tol:
            print(f"\n✅ Converged after {iteration} iterations")
            tqdm_meter.close()
            return choice_cluster.cpu(), initial_state.cpu()


# 5. 主函数
def k_means_clustering(x, n_mem, d_model, max_iterations=1000):
    """
    K-Means聚类，带双重保护：
    1. 最多max_iterations次迭代
    2. 最多1分钟运行时间
    """
    # 设置信号处理程序
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # 1分钟超时

    try:
        start = time.time()
        x = x.view([-1, d_model])

        print(f'🚀 Starting K-Means clustering:')
        print(f'   - Clusters: {n_mem}')
        print(f'   - Data shape: {x.shape}')
        print(f'   - Max iterations: {max_iterations}')
        print(f'   - Timeout: 60 seconds')

        # 使用带迭代限制的版本
        _, cluster_centers = kmeans_with_limit(
            X=x,
            num_clusters=n_mem,
            distance='euclidean',
            tol=1e-4,
            device=torch.device('cuda:2'),
            max_iter=max_iterations
        )

        elapsed = time.time() - start
        print(f'✅ K-Means completed in {elapsed:.2f} seconds')

        if elapsed > 55:
            print(f'⚠️ Warning: Took {elapsed:.2f}s, very close to timeout!')
        elif elapsed > 30:
            print(f'ℹ️ Note: Took {elapsed:.2f}s, consider optimizing')

        return cluster_centers

    except TimeoutException:
        elapsed = time.time() - start
        print(f'⏰ Timeout after {elapsed:.2f} seconds!')
        print('💡 Suggestions:')
        print('   1. Reduce number of clusters')
        print('   2. Use data sampling')
        print('   3. Increase timeout if needed')
        raise TimeoutException(f"K-Means timed out after 60 seconds")

    except Exception as e:
        print(f'❌ Error in K-Means: {e}')
        raise

    finally:
        # 清理：取消闹钟
        signal.alarm(0)



