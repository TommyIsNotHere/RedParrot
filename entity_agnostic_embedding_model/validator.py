import argparse
import torch
import numpy as np
import pandas as pd
from entity_agnostic_embedding_model.contrastive_learning import ContrastiveMARLOEmbeddingModel


class ContrastiveModelValidator:
    """
    对比学习模型验证器
    验证主干相似的问题是否产生相似的embedding
    """

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def encode_queries(self, queries):
        """批量编码查询"""
        with torch.no_grad():
            for i in range(0, len(queries), 32):
                batch = queries[i:i+32]
                batch_embeddings = self.model(batch).cpu()
                if i == 0:
                    embeddings = batch_embeddings
                else:
                    embeddings = torch.cat(
                        (embeddings, batch_embeddings), dim=0)
        return embeddings

    def calculate_similarity_matrix(self, embeddings, metric='cosine'):
        """计算相似度矩阵"""
        if metric == 'cosine':
            # 余弦相似度
            embeddings_norm = torch.nn.functional.normalize(
                embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        elif metric == 'euclidean':
            # 欧氏距离
            n = embeddings.size(0)
            similarity_matrix = torch.zeros(n, n)
            for i in range(n):
                for j in range(n):
                    similarity_matrix[i, j] = torch.nn.functional.pairwise_distance(
                        embeddings[i:i+1], embeddings[j:j+1])
        return similarity_matrix

    def semantic_similarity_validation(self, test_queries, similar_pairs, dissimilar_pairs):
        """
        语义相似性验证
        验证相似问题的embedding距离是否更近
        """
        print("=== 语义相似性验证 ===")

        # 编码所有查询
        all_queries = list(set(test_queries))
        embeddings = self.encode_queries(all_queries)

        # 创建查询到索引的映射
        query_to_idx = {query: idx for idx, query in enumerate(all_queries)}

        # 计算相似对的距离
        similar_distances = []
        for q1, q2 in similar_pairs:
            if q1 in query_to_idx and q2 in query_to_idx:
                idx1, idx2 = query_to_idx[q1], query_to_idx[q2]
                dist = torch.nn.functional.cosine_similarity(
                    embeddings[idx1:idx1+1], embeddings[idx2:idx2+1])
                similar_distances.append(dist.item())

        # 计算不相似对的距离
        dissimilar_distances = []
        for q1, q2 in dissimilar_pairs:
            if q1 in query_to_idx and q2 in query_to_idx:
                idx1, idx2 = query_to_idx[q1], query_to_idx[q2]
                dist = torch.nn.functional.cosine_similarity(
                    embeddings[idx1:idx1+1], embeddings[idx2:idx2+1])
                dissimilar_distances.append(dist.item())

        # 计算统计指标
        avg_similar = np.mean(similar_distances) if similar_distances else 0
        avg_dissimilar = np.mean(
            dissimilar_distances) if dissimilar_distances else 0

        print(f"相似问题平均余弦相似度: {avg_similar:.4f}")
        print(f"不相似问题平均余弦相似度: {avg_dissimilar:.4f}")
        print(f"差异: {avg_similar - avg_dissimilar:.4f}")

        return {
            'similar_distances': similar_distances,
            'dissimilar_distances': dissimilar_distances,
            'avg_similar': avg_similar,
            'avg_dissimilar': avg_dissimilar
        }

    def clustering_consistency_validation(self, queries, true_labels, n_clusters=5):
        """
        聚类一致性验证
        验证相似问题是否聚类到同一类别
        """
        print("=== 聚类一致性验证 ===")

        embeddings = self.encode_queries(queries)
        embeddings_np = embeddings.cpu().detach().numpy()

        # 使用K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        predicted_labels = kmeans.fit_predict(embeddings_np)

        # 计算聚类指标
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        print(f"调整兰德指数 (ARI): {ari:.4f}")
        print(f"归一化互信息 (NMI): {nmi:.4f}")

        return {
            'ari': ari,
            'nmi': nmi,
            'predicted_labels': predicted_labels
        }

    def retrieval_performance_validation(self, queries, ground_truth, top_k=5):
        """
        检索性能验证
        验证embedding在检索任务中的表现
        """
        print("=== 检索性能验证 ===")

        embeddings = self.encode_queries(queries)
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        # 计算检索指标
        precisions = []
        recalls = []

        for i, query in enumerate(queries):
            # 获取真实相关的查询
            true_relevant = ground_truth.get(query, [])
            if not true_relevant:
                continue

            # 获取top-k相似查询
            similarities = similarity_matrix[i]
            _, top_indices = torch.topk(
                similarities, k=min(top_k, len(queries)))

            retrieved = [queries[idx] for idx in top_indices.cpu().numpy()]

            # 计算precision和recall
            relevant_retrieved = set(retrieved) & set(true_relevant)
            precision = len(relevant_retrieved) / \
                len(retrieved) if retrieved else 0
            recall = len(relevant_retrieved) / \
                len(true_relevant) if true_relevant else 0

            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0

        print(f"平均Precision@{top_k}: {avg_precision:.4f}")
        print(f"平均Recall@{top_k}: {avg_recall:.4f}")
        print(f"F1分数: {2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0:.4f}")

        return {
            'precisions': precisions,
            'recalls': recalls,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall
        }

    def generate_synthetic_test_data(self, original_data_path, sample_size=100):
        """
        生成合成测试数据
        基于原始数据创建相似和不相似的查询对
        """
        data = pd.read_parquet(original_data_path, columns=['query', 'erased'])

        # 随机采样
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        queries = data['query'].tolist()

        # 创建相似对（基于相同erased文本）
        similar_pairs = []
        erased_groups = data.groupby('erased')['query'].apply(list).to_dict()

        for erased_text, group_queries in erased_groups.items():
            if len(group_queries) >= 2:
                # 同一组内的查询视为相似
                for i in range(len(group_queries)):
                    for j in range(i+1, len(group_queries)):
                        similar_pairs.append(
                            (group_queries[i], group_queries[j]))

        # 创建不相似对（不同erased文本）
        dissimilar_pairs = []
        # erased_list = list(erased_groups.keys())

        for i in range(len(queries)):
            for j in range(i+1, len(queries)):
                q1, q2 = queries[i], queries[j]
                erased1 = data[data['query'] == q1]['erased'].iloc[0]
                erased2 = data[data['query'] == q2]['erased'].iloc[0]

                if erased1 != erased2:
                    dissimilar_pairs.append((q1, q2))

        # 限制对数
        max_pairs = min(100, len(similar_pairs), len(dissimilar_pairs))
        similar_pairs = similar_pairs[:max_pairs]
        dissimilar_pairs = dissimilar_pairs[:max_pairs]

        return queries, similar_pairs, dissimilar_pairs

    def generate_ground_truth_from_erased(self, original_data_path, sample_size=1000, similarity_threshold=0.9):
        """
        基于erased_embedding字段自动生成ground truth
        使用余弦相似度计算，相似度大于阈值的视为相似查询
        """
        print("正在基于erased_embedding字段生成ground truth...")

        # 读取包含erased_embedding的数据
        data = pd.read_parquet(original_data_path, columns=[
                               'query', 'erased_embedding'])

        # 随机采样
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        queries = data['query'].tolist()
        embeddings = data['erased_embedding'].tolist()

        # 将字符串格式的embedding转换为numpy数组
        import numpy as np
        import ast

        # 转换embedding格式
        embedding_vectors = []
        for emb_str in embeddings:
            if isinstance(emb_str, str):
                # 如果是字符串格式，解析为列表
                emb_list = ast.literal_eval(emb_str)
                embedding_vectors.append(np.array(emb_list))
            else:
                # 如果已经是数组格式，直接使用
                embedding_vectors.append(np.array(emb_str))

        # 计算余弦相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embedding_vectors)

        # 创建ground truth字典
        ground_truth = {}

        # 为每个查询构建相似查询列表
        for i, query in enumerate(queries):
            similar_indices = []

            # 查找相似度大于阈值的查询
            for j in range(len(queries)):
                if i != j and similarity_matrix[i][j] > similarity_threshold:
                    similar_indices.append(j)

            if similar_indices:
                ground_truth[query] = [queries[idx] for idx in similar_indices]

        print(f"成功为 {len(ground_truth)} 个查询生成ground truth")
        print(f"相似度阈值: {similarity_threshold}")
        return queries, ground_truth

    def generate_synthetic_test_data_with_ground_truth(self, original_data_path, sample_size=1000):
        """
        同时生成测试数据和ground truth
        """
        queries, ground_truth = self.generate_ground_truth_from_erased(
            original_data_path, sample_size)

        # 生成相似和不相似对用于语义验证
        similar_pairs = []
        dissimilar_pairs = []

        data = pd.read_parquet(original_data_path, columns=['query', 'erased'])
        sampled_data = data[data['query'].isin(queries)]

        # 创建相似对（相同erased）
        for erased_text, group in sampled_data.groupby('erased'):
            group_queries = group['query'].tolist()
            for i in range(len(group_queries)):
                for j in range(i+1, len(group_queries)):
                    similar_pairs.append((group_queries[i], group_queries[j]))

        # 创建不相似对（不同erased）
        # erased_values = sampled_data['erased'].unique()
        for i, row1 in sampled_data.iterrows():
            for j, row2 in sampled_data.iterrows():
                if i < j and row1['erased'] != row2['erased']:
                    dissimilar_pairs.append((row1['query'], row2['query']))

        # 限制对数
        max_pairs = min(50, len(similar_pairs), len(dissimilar_pairs))
        similar_pairs = similar_pairs[:max_pairs]
        dissimilar_pairs = dissimilar_pairs[:max_pairs]

        return queries, similar_pairs, dissimilar_pairs, ground_truth

    def run_comprehensive_validation(self, test_data_path=None,
                                     queries=None, similar_pairs=None,
                                     dissimilar_pairs=None, ground_truth=None,
                                     sample_size=1000):
        """
        运行完整的验证流程
        """
        print("开始对比学习模型验证...")
        print("=" * 50)

        results = {}

        # 如果没有提供测试数据，则使用自动生成
        if test_data_path and queries is None:
            print("自动生成测试数据和ground truth...")
            queries, similar_pairs, dissimilar_pairs, ground_truth = \
                self.generate_synthetic_test_data_with_ground_truth(
                    test_data_path, sample_size=sample_size)
            print(f"生成了 {len(queries)} 个查询和 {len(ground_truth)} 个ground truth条目")

        # 1. 语义相似性验证
        if queries and similar_pairs and dissimilar_pairs:
            semantic_results = self.semantic_similarity_validation(
                queries, similar_pairs, dissimilar_pairs)
            results['semantic'] = semantic_results

        # # 2. 聚类一致性验证（如果有真实标签）
        # if ground_truth:
        #     cluster_results = self.clustering_consistency_validation(
        #         queries, list(ground_truth.values()))
        #     results['clustering'] = cluster_results

        # 3. 检索性能验证
        if ground_truth:
            retrieval_results = self.retrieval_performance_validation(
                queries, ground_truth)
            results['retrieval'] = retrieval_results

        # 综合评估
        print("\n=== 综合评估 ===")
        if 'semantic' in results:
            semantic = results['semantic']
            if semantic['avg_similar'] > semantic['avg_dissimilar']:
                print("✓ 语义相似性验证通过")
            else:
                print("✗ 语义相似性验证未通过")

        if 'clustering' in results:
            clustering = results['clustering']
            if clustering['ari'] > 0.3 and clustering['nmi'] > 0.3:
                print("✓ 聚类一致性验证通过")
            else:
                print("✗ 聚类一致性验证未通过")

        if 'retrieval' in results:
            retrieval = results['retrieval']
            if retrieval['avg_precision'] > 0.5 and retrieval['avg_recall'] > 0.3:
                print("✓ 检索性能验证通过")
            else:
                print("✗ 检索性能验证未通过")

        print("=" * 50)
        return results


def validate_contrastive_model(model_path, data_path, test_data_path=None, sample_size=1000):
    """
    验证对比学习模型的主函数
    """
    # 加载模型
    model = ContrastiveMARLOEmbeddingModel()
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建验证器
    validator = ContrastiveModelValidator(model, device)

    # 运行验证
    results = validator.run_comprehensive_validation(
        test_data_path=test_data_path or data_path, sample_size=sample_size
    )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str,
                        default='./model/contrastive_model/best_model_0917.pth', help="对比学习模型路径")
    parser.add_argument("-d", "--data_path", type=str,
                        default="./data/ignore/20250909_AIMI全量_AIO_erased_embedding.parquet", help="用于生成测试数据的原始数据路径")
    parser.add_argument("--sample_size", type=int,
                        default=1000, help="可选的测试数据样本大小，如果不提供则自动生成")
    parser.add_argument("-t", "--test_data_path", type=str,
                        default=None, help="可选的测试数据路径，如果不提供则自动生成")
    return parser


# 修改主函数以支持验证
if __name__ == '__main__':
    args = build_parser().parse_args()
    validate_contrastive_model(
        args.model_path,
        args.data_path,
        sample_size=args.sample_size,
    )
