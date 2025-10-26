from utils import split_dataset_and_create_loaders
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sentence_transformers.util import pairwise_cos_sim, pairwise_euclidean_sim, pairwise_manhattan_sim
from enum import Enum
import torch.nn.functional as F


class DistanceMetric(Enum):
    """The metric for the triplet loss"""
    def COSINE(x, y):
        return 1 - pairwise_cos_sim(x, y)

    def EUCLIDEAN(x, y):
        return pairwise_euclidean_sim(x, y)

    def MANHATTAN(x, y):
        return pairwise_manhattan_sim(x, y)


def contrastive_nerfarneg_sample(data, anchor_indices=None, n_clusters=9, near_quantile=0.25, far_quantile=0.75,
                         min_bucket_size=5, distance_metric='cosine', random_state=42):
    """
    重采样方法:
    对每个anchor采样:
      1 个同簇正样本 + 1 个相对'近'的负样本 + 1 个相对'远'的负样本
    为适配下游数据集(期望 n*3 的矩阵)，输出两条三元组:
      (anchor, pos, near_neg) 与 (anchor, pos, far_neg)

    参数:
        data: 包含 'erased_embedding' 与 'query' 的DataFrame
        anchor_indices: 可选，指定哪些索引作为anchor；None则使用全部
        n_clusters: KMeans聚类数量
        near_quantile: 近负样本分位阈值 (0~1)
        far_quantile: 远负样本分位阈值 (0~1)
        min_bucket_size: 若分位段样本过少，退化为前/后若干个候选
        distance_metric: 'euclidean' | 'cosine'
        random_state: KMeans随机种子

    返回:
        np.ndarray 形状 (m, 3)  每行 = (anchor_idx, pos_idx, neg_idx)
    """
    assert 'erased_embedding' in data.columns, "data 必须包含 'erased_embedding' 列"
    embeddings = np.vstack(data['erased_embedding'].to_list())
    n_samples = len(embeddings)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    if anchor_indices is None:
        anchor_indices = np.arange(n_samples)
    else:
        anchor_indices = np.asarray(anchor_indices)

    def pairwise_distance(anchor_vec, cand_vecs):
        if distance_metric == 'euclidean':
            diff = cand_vecs - anchor_vec
            return np.sqrt(np.sum(diff * diff, axis=1))
        elif distance_metric == 'cosine':
            # 1 - cosine similarity
            a = anchor_vec / (np.linalg.norm(anchor_vec) + 1e-12)
            b = cand_vecs / (np.linalg.norm(cand_vecs,
                             axis=1, keepdims=True) + 1e-12)
            return 1 - np.dot(b, a)
        else:
            raise ValueError("Unsupported distance_metric")

    sample_indices = []
    rng = np.random.default_rng()

    for anchor_idx in anchor_indices:
        anchor_cluster = cluster_labels[anchor_idx]

        # 正样本候选（同簇且非自身）
        positive_candidates = np.where(cluster_labels == anchor_cluster)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]
        if positive_candidates.size == 0:
            continue
        pos_idx = int(rng.choice(positive_candidates))

        # 负样本候选（异簇）
        negative_candidates = np.where(cluster_labels != anchor_cluster)[0]
        if negative_candidates.size < 2:
            # 需要至少两个不同负样本用于 near / far
            continue

        neg_embs = embeddings[negative_candidates]
        anchor_vec = embeddings[anchor_idx]
        dists = pairwise_distance(anchor_vec, neg_embs)

        # 排序
        order = np.argsort(dists)
        sorted_neg_indices = negative_candidates[order]
        sorted_dists = dists[order]

        # 依据分位获取 near / far pools
        near_threshold = np.quantile(sorted_dists, near_quantile)
        far_threshold = np.quantile(sorted_dists, far_quantile)

        near_pool = sorted_neg_indices[sorted_dists <= near_threshold]
        far_pool = sorted_neg_indices[sorted_dists >= far_threshold]

        # 退化处理：若分位段过小，则取前/后 min_bucket_size
        if near_pool.size < 1:
            near_pool = sorted_neg_indices[:min(
                min_bucket_size, sorted_neg_indices.size)]
        if far_pool.size < 1:
            far_pool = sorted_neg_indices[-min(min_bucket_size,
                                               sorted_neg_indices.size):]

        # 避免 near 与 far 取到同一个（若全集很小）
        near_neg = int(rng.choice(near_pool))
        attempt = 0
        max_attempts = 10
        far_neg = int(rng.choice(far_pool))
        while far_neg == near_neg and attempt < max_attempts:
            far_neg = int(rng.choice(far_pool))
            attempt += 1
        # 生成两条三元组
        sample_indices.append([anchor_idx, pos_idx, near_neg])
        sample_indices.append([anchor_idx, pos_idx, far_neg])

    if len(sample_indices) == 0:
        return np.empty((0, 3), dtype=int)
    return np.array(sample_indices, dtype=int)


def contrastive_more_negative_sample(data, anchor_indices=None, negative_samples=3):
    """
    对比学习数据采样函数
    目前还没有实现,我的设想是这么做:
    1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    2.遍历anchor_indices,对于每一个anchor,随机取出同一个类里的数据索引当作正样本,随机取出其他类里的数据索引当作负样本
    3.需要注意的是,每个(anchor_indice, positive_indice/negative_indice)只能出现一次
    4.返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    """
    # 1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    erased_embeddings = np.array(data['erased_embedding'].tolist())
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=9, random_state=42).fit(erased_embeddings)
    cluster_labels = kmeans.labels_
    # 如果没有提供anchor_indices，使用所有索引
    if anchor_indices is None:
        anchor_indices = np.arange(len(data))

    sample_indices = []
    used_pairs = set()  # 用于记录已经使用过的组合

    # AI生成,性能不太行,但是效果经过我的修改应该是符合要去的
    for anchor_idx in anchor_indices:
        anchor_cluster = cluster_labels[anchor_idx]

        # 获取同类的所有索引（排除anchor自身）
        positive_candidates = np.where(cluster_labels == anchor_cluster)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]

        # 获取其他类的所有索引
        negative_candidates = np.where(cluster_labels != anchor_cluster)[0]

        # 随机选择正样本和负样本
        if len(positive_candidates) > 0 and len(negative_candidates) > 0:
            # 尝试找到未使用过的组合
            pos_idx = np.random.choice(positive_candidates)
            neg_idx = np.random.choice(
                negative_candidates, size=negative_samples, replace=False).tolist()
            sample_indices.extend([
                [anchor_idx, pos_idx, neg] for neg in neg_idx
            ])

    # 返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    return np.array(sample_indices)


def contrastive_sample(data, anchor_indices=None):
    """
    对比学习数据采样函数
    目前还没有实现,我的设想是这么做:
    1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    2.遍历anchor_indices,对于每一个anchor,随机取出同一个类里的数据索引当作正样本,随机取出其他类里的数据索引当作负样本
    3.需要注意的是,每个(anchor_indice, positive_indice/negative_indice)只能出现一次
    4.返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    """
    # 1.对所有数据取出erased_embedding,使用这个信息聚类,得到分类信息
    erased_embeddings = np.array(data['erased_embedding'].tolist())
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=42).fit(erased_embeddings)
    cluster_labels = kmeans.labels_
    # 如果没有提供anchor_indices，使用所有索引
    if anchor_indices is None:
        anchor_indices = np.arange(len(data))

    sample_indices = []
    used_pairs = set()  # 用于记录已经使用过的组合

    # AI生成,性能不太行,但是效果经过我的修改应该是符合要去的
    for anchor_idx in anchor_indices:
        anchor_cluster = cluster_labels[anchor_idx]

        # 获取同类的所有索引（排除anchor自身）
        positive_candidates = np.where(cluster_labels == anchor_cluster)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]

        # 获取其他类的所有索引
        negative_candidates = np.where(cluster_labels != anchor_cluster)[0]

        # 随机选择正样本和负样本
        if len(positive_candidates) > 0 and len(negative_candidates) > 0:
            # 尝试找到未使用过的组合
            max_attempts = 100
            for _ in range(max_attempts):
                pos_idx = np.random.choice(positive_candidates)
                neg_idx = np.random.choice(negative_candidates)

                # 检查组合是否已经使用过
                def judge_set_pair(anchor_idx, pos_idx, neg_idx, used_pairs):
                    if (anchor_idx, pos_idx) in used_pairs or (anchor_idx, neg_idx) in used_pairs:
                        return False
                    # 翻转
                    elif (pos_idx, anchor_idx) in used_pairs or (neg_idx, anchor_idx) in used_pairs:
                        return False
                    else:
                        used_pairs.add((anchor_idx, pos_idx))
                        used_pairs.add((anchor_idx, neg_idx))
                        return True

                if judge_set_pair(anchor_idx, pos_idx, neg_idx, used_pairs):
                    sample_indices.append([anchor_idx, pos_idx, neg_idx])
                    break
            else:
                # 如果无法找到未使用的组合，跳过这个anchor
                continue

    # 返回一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
    return np.array(sample_indices)


# 这里有一点纠结的地方在于我究竟是将采样的过程放在数据集中，还是在数据集外
# 希望能够对数据集的自定义程度较高，因此我倾向于将采样过程放在数据集外
class ContrastiveQueryDataset(Dataset):
    def __init__(self, data, sample_indices=None):
        """
        对比学习数据集：包含原始查询和对应的擦除后embedding
        同时需要构建正负样本对
        data: 数据集
        sample_indices: 采样后的索引,这是一个n*3的矩阵,每一行是一个(样本索引,正样本索引,负样本索引)
        """
        self.data = data
        # self.target_embeddings = np.array(
        #     self.data['erased_embedding'].tolist())
        self.query_list = np.array(self.data['query'].tolist())
        if sample_indices is None:
            sample_indices = self.default_sample_method(
                sample_num=len(self.data))
        self.sample_indices = sample_indices

    def default_sample_method(self, sample_num):
        """
        默认的采样方法：尚未定义好,暂时定义为随机采样
        """
        return contrastive_sample(self.data, np.random.randint(0, len(self.data), sample_num))

    def __len__(self):
        # 这里是个容易写错的地方,因为数据量取决于采样的数目而非整个数据的大小
        return len(self.sample_indices)

    def __getitem__(self, idx):
        # raw_query = self.data.iloc[idx]['query']
        # target_embedding = torch.tensor(
        #     self.target_embeddings[idx], dtype=torch.float32)

        # # 随机选择负样本
        # neg_idx = np.random.choice([i for i in range(len(self)) if i != idx])
        # neg_embedding = torch.tensor(
        #     self.target_embeddings[neg_idx], dtype=torch.float32)

        return {
            'query': self.query_list[self.sample_indices[idx, 0]],
            'positive_query': self.query_list[self.sample_indices[idx, 1]],
            'negative_query': self.query_list[self.sample_indices[idx, 2]]
        }


def InfoNCE_loss(anchor, positive, negative, margin=0.2):
    """
    对比损失函数：让anchor和positive更接近，和negative更远离
    """
    pos_distance = torch.nn.functional.cosine_similarity(anchor, positive)
    neg_distance = torch.nn.functional.cosine_similarity(anchor, negative)

    # InfoNCE风格的对比损失
    loss = -torch.log(torch.exp(pos_distance) /
                      (torch.exp(pos_distance) + torch.exp(neg_distance)))

    return loss.mean()


def triplet_loss(anchor, positive, negative, positive_weight=1, negative_weight=1, margin=0.1, metric=DistanceMetric.COSINE):
    """
    triplet loss实现
    计算triplet loss
    anchor: 锚点样本的embedding
    positive: 正样本的embedding
    negative: 负样本的embedding
    margin: triplet loss的margin参数
    metric: 距离度量指标,默认是cosine相似度
    """
    pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    loss = F.relu(positive_weight * pos_dist -
                  negative_weight * neg_dist + margin)
    return loss.mean()


class WeightedTripletLoss(nn.Module):
    def __init__(self, margin=0.5, init_alpha=1.0, init_beta=1.0):
        super(WeightedTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, anchor, positive, negative):
        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        # 使用可学习的权重
        loss = F.relu(self.beta * pos_dist -
                      self.alpha * neg_dist + self.margin)
        return loss.mean()


def batch_hard_triplet_loss(embeddings, labels, margin=0.2, squared=False):
    """
    实现批量硬三元组挖掘的Triplet Loss

    embeddings: 特征向量 [batch_size, embed_dim]
    labels: 标签 [batch_size]
    """
    # 计算欧氏距离矩阵
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # 对每个锚点，找到最难的正例(最远的同类样本)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = mask_pos.float() - \
        torch.eye(mask_pos.shape[0]).to(mask_pos.device)
    mask_pos[mask_pos < 0] = 0
    hardest_positive_dist = (pairwise_dist * mask_pos).max(dim=1)[0]

    # 对每个锚点，找到最难的负例(最近的不同类样本)
    mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
    hardest_negative_dist = (pairwise_dist * mask_neg.float()).min(dim=1)[0]

    # 计算三元组损失
    triplet_loss = F.relu(hardest_positive_dist -
                          hardest_negative_dist + margin)
    return triplet_loss.mean()


class ContrastiveMARLOEmbeddingModel(nn.Module):
    def __init__(self, base_model='google-bert/bert-base-uncased', device=None):
        super(ContrastiveMARLOEmbeddingModel, self).__init__()
        self.model = SentenceTransformer(base_model)
        self.model.max_seq_length = 256
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        # self.adapter = nn.Linear(self.embedding_dim, target_embedding_dim)

    def forward(self, texts):
        tokenized = self.model.tokenize(texts)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        out = self.model(tokenized)              # returns dict
        embeddings = out['sentence_embedding']      # (B, D)
        return F.normalize(embeddings, dim=-1)


def train_contrastive_model(model_path, data_path, sample_method='default',
                            batch_size=32,
                            save_path='./model/contrastive_model/best_model.pth',
                            epoch_num=10,
                            positive_weight=1.0,
                            negative_weight=1.0,
                            device=None):
    """
    对比学习训练函数
    """
    data = pd.read_parquet(data_path, columns=[
                           'query', 'erased', 'erased_embedding'])
    if sample_method == 'default':
        sample_indices = None
    elif sample_method == 'more_negative':
        sample_indices = contrastive_more_negative_sample(
            data,
            anchor_indices=np.random.randint(0, len(data), len(data)),
            negative_samples=3
        )
    elif sample_method == 'nerfarneg':
        sample_indices = contrastive_nerfarneg_sample(
            data, anchor_indices=np.random.randint(0, len(data), len(data)))
    else:
        raise ValueError("未知的采样方法")
    dataset = ContrastiveQueryDataset(data, sample_indices)
    model = ContrastiveMARLOEmbeddingModel(
        base_model=model_path,
        device=device
    )
    train_loader, val_loader, test_loader = split_dataset_and_create_loaders(
        dataset, train_ratio=0.85, val_ratio=0.05, test_ratio=0.10, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # triplet_loss_fn = losses.TripletLoss(
    #     model, losses.TripletDistanceMetric.COSINE, triplet_margin=0.01)

    # 训练历史记录
    train_history = []
    val_history = []
    best_val_loss = float('inf')

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0

        # 使用tqdm进度条
        pbar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{epoch_num}', leave=False)

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            # anchor_list = batch['query']
            # pos_list = batch['positive_query']
            # neg_list = batch['negative_query']
            # B = len(anchor_list)
            # all_texts = anchor_list + pos_list + neg_list   # 长度 3B
            # all_emb = model(all_texts)
            # anchor_embeddings, positive_embeddings, negative_embeddings = torch.split(all_emb, B, dim=0)
            # 获取模型输出的embedding
            anchor_embeddings = model(batch['query'])
            positive_embeddings = model(batch['positive_query'])
            negative_embeddings = model(batch['negative_query'])

            # 计算对比损失
            # loss = triplet_loss_fn(
            #     (anchor_embeddings, positive_embeddings, negative_embeddings),
            #     # (anchor_embeddings, positive_embeddings, negative_embeddings),
            #     labels=None
            # )
            loss = triplet_loss(anchor_embeddings,
                                positive_embeddings, negative_embeddings,
                                positive_weight=positive_weight, negative_weight=negative_weight)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条显示
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss_so_far:.4f}'
            })
        _ = [torch.cuda.empty_cache() for i in range(5)]
        # 计算epoch平均loss
        epoch_avg_loss = total_loss / len(train_loader)
        train_history.append(epoch_avg_loss)

        print(f'Epoch {epoch+1}/{epoch_num} completed. '
              f'Average Loss: {epoch_avg_loss:.4f}, '
              f'Total Loss: {total_loss:.4f}')

        # 每5个epoch打印一次训练进度总结
        if (epoch + 1) % 5 == 0:
            print(f'\n--- Training Summary after {epoch+1} epochs ---')
            print(f'Best loss: {min(train_history):.4f}')
            print(f'Current loss: {epoch_avg_loss:.4f}')
            print('---' + '-' * 30 + '\n')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save(model.state_dict(),
                       f"{save_path}/model_atep{epoch+1}.pth")

        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    anchor_embeddings = model(batch['query'])
                    positive_embeddings = model(batch['positive_query'])
                    negative_embeddings = model(batch['negative_query'])

                    # loss = triplet_loss_fn(
                    #     (anchor_embeddings, positive_embeddings, negative_embeddings),
                    #     labels=None
                    # )
                    loss = triplet_loss(
                        anchor_embeddings, positive_embeddings, negative_embeddings, positive_weight=positive_weight, negative_weight=negative_weight)
                    total_val_loss += loss.item()

            epoch_val_loss = total_val_loss / len(val_loader)
            val_history.append(epoch_val_loss)

            # 保存最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(model.state_dict(),
                           f"{save_path}/best_model_atep{epoch+1}.pth")

            print(
                f'Epoch {epoch+1}/{epoch_num} -  Val Loss: {epoch_val_loss:.4f}')
        # 测试阶段（可选）
    if test_loader:
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                anchor_embeddings = model(batch['query'])
                positive_embeddings = model(batch['positive_query'])
                negative_embeddings = model(batch['negative_query'])

                # loss = triplet_loss_fn(
                #     (anchor_embeddings, positive_embeddings, negative_embeddings),
                #     labels=None
                # )
                loss = triplet_loss(anchor_embeddings,
                                    positive_embeddings, negative_embeddings, positive_weight=positive_weight, negative_weight=negative_weight)
                total_test_loss += loss.item()

        test_loss = total_test_loss / len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
    return train_history


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a contrastive learning model.")
    parser.add_argument('--model_path', type=str,
                        default='google-bert/bert-base-uncased', help='Base model path or name.')
    parser.add_argument('--data_path', type=str,
                        default='./data/ignore/20250909_AIMI全量_AIO_erased_embedding.parquet',
                        help='Path to the training data (parquet file).')
    parser.add_argument('--sample_method', type=str, default='default',
                        choices=['default', 'more_negative', 'nerfarneg'], help='Sampling method for contrastive learning.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--epoch_num', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--positive_weight', type=float, default=1.0,
                        help='Weight for positive pairs in the loss function.')
    parser.add_argument('--negative_weight', type=float, default=1.0,
                        help='Weight for negative pairs in the loss function.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (default 0). Ignored if no CUDA.')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    # 定义使用设备
    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f'Invalid --gpu {args.gpu}, available count = {torch.cuda.device_count()}')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # 定义保存路径
    if args.save_path is None:
        # "best_model_aimicorrect_ex0922_0925_moreneg_500"
        if any([kw in args.data_path for kw in ['all', '全量']]):
            datased_symbol = 'aimiall'
        elif any([kw in args.data_path for kw in ['correct', '正确']]):
            dataset_symbol = 'aimicorrect'
        else:
            dataset_symbol = os.path.basename(args.data_path).split('.')[0]

        ex_symbol = [ex for ex in ['ex0916', 'ex0922',
                                   'ex0925', 'ex0928_85',
                                   'ex0928_90', 'ex0928_95'] if ex in args.data_path]
        ex_symbol = ex_symbol[0] if len(ex_symbol) > 0 else 'exno'
        monthday = datetime.now().strftime("%m%d")
        sample_symbol = args.sample_method
        modeldir = args.model_path.split('/')[-1]
        if not os.path.exists(f'./model/{modeldir}'):
            os.makedirs(f'./model/{modeldir}')

        args.save_path = f'./model/{modeldir}/contrastive_{dataset_symbol}_{ex_symbol}_{monthday}_{sample_symbol}_pw{args.positive_weight}_nw{args.negative_weight}_ep{args.epoch_num}'
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        print("The model will save to", args.save_path)

    train_contrastive_model(
        model_path=args.model_path,
        data_path=args.data_path,
        sample_method=args.sample_method,
        batch_size=args.batch_size,
        save_path=args.save_path,
        epoch_num=args.epoch_num,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        device=device
    )
