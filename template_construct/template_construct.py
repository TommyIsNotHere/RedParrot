import csv
import os
import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from template_construct.embed_tool import get_embeddings, embedding_L2_normalization, get_embeddings_by_model
CLUSTERNUM = 9
SIMILARITY_THRESHOLD = 0.95
COMPONENT_NUM_THRESHOLD = 4
# with open(file_url, 'r', encoding='utf-8') as f:
#     data = json.load(f)


def deduplicate_dicts_by_key(dict_list, key='query'):
    """
    按照某个key对字典列表进行去重
    :param dict_list: 字典列表
    :param key: 去重的key,默认是query
    :return: 去重后的字典列表
    """
    seen = set()
    result = []
    for d in dict_list:
        value = d.get(key)
        if value not in seen:
            seen.add(value)
            result.append(d)
    return result


def kmeans(erased_embeddings, selected_clusters=9):
    """
    对数据进行kmeans聚类
    :param data: 数据
    :param selected_clusters: 聚类数量
    :return: 聚类结果
    """
    # ======================
    # K-Means聚类分析模块
    # ======================
    import os
    os.environ["OMP_NUM_THREADS"] = "1"  # 限制OpenMP线程数
    embeddings_array = embedding_L2_normalization(erased_embeddings)
    # 执行K-Means聚类
    final_kmeans = KMeans(n_clusters=selected_clusters,
                          init='k-means++', random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(embeddings_array)
    return final_kmeans, cluster_labels

# 计算连通度和连接图的代码


def find_connected_components(matrix):
    """
    找到矩阵中的连通分量
    :param matrix: 输入的矩阵,是一个二维数组,这个代表着无向图森林的矩阵
    :return: 输出组的信息,是一个列表,列表中的每个元素是一个列表,代表某个组的所有数据节点的索引
    """
    n = len(matrix)
    visited = [False] * n
    groups = []

    def dfs(node, current_group):
        if visited[node]:
            return
        visited[node] = True
        current_group.append(node)
        for neighbor in range(n):
            if matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, current_group)
    for i in range(n):
        if not visited[i]:
            current_group = []
            dfs(i, current_group)
            groups.append(current_group)
    return groups


def calculate_degrees(matrix, groups):
    """
    计算连通分量中的度数
    :param matrix: 输入的矩阵,是一个二维数组
    :param groups: 输入的连通分量,是一个列表,列表中的每个元素是一个列表,代表某个组所有数据节点的索引
    :return: 输出的度数,是一个字典,字典的键是整数,表示矩阵中的行号,字典的值是整数,表示该节点的度数
    """
    degrees = {}
    for group in groups:
        # 创建子图邻接字典
        subgraph = {node: [] for node in group}
        for i in group:
            for j in group:
                if matrix[i][j] == 1 and i != j:
                    subgraph[i].append(j)
        # 统计每个节点的度数
        for node in group:
            degrees[node] = len(subgraph[node])
    return degrees


def template_construct(data, model=None,
                       similarity_threshold=SIMILARITY_THRESHOLD,
                       component_num_threshold=COMPONENT_NUM_THRESHOLD,
                       template_idx=list(range(2))):
    """
    模板构造函数
    :param data: 输入数据,必须包含query和erased字段,是一个字典列表,这个列表需要是query_erase函数的输出
    :return: 输出数据,包含模板和测试数据
    """
    print(f"相似度阈值: {similarity_threshold}, 子图数量阈值: {component_num_threshold}")
    data = deduplicate_dicts_by_key(data, 'query')
    raw_query_list = [item['query'] for item in data]
    erased_query_list = [item['erased'] for item in data]
    # 获得主干问题的词嵌入
    if model is None:
        erased_embeddings = get_embeddings(erased_query_list)
    else:
        erased_embeddings = get_embeddings_by_model(raw_query_list, model)
    # 计算所有主干问题的相似矩阵
    erased_embeddings_np = np.array(erased_embeddings)
    cos_sim = np.dot(erased_embeddings_np, erased_embeddings_np.T)
    # 聚类得到聚类结果
    selected_clusters = CLUSTERNUM  # 这里可以修改
    final_kmeans, cluster_labels = kmeans(erased_embeddings, selected_clusters)

    total_groups_num = 0
    total_data_num = 0
    groups_record = []
    # 记录有效组数据
    for i in range(selected_clusters):
        cluster_indices = np.where(cluster_labels == i)[0].tolist()
        cos_sim_cluster = cos_sim[cluster_indices, :][:, cluster_indices]
        cos_sim_cluster_graph = cos_sim_cluster >= similarity_threshold
        cluster_groups = find_connected_components(cos_sim_cluster_graph)
        cluster_groups = [group for group in cluster_groups if len(
            group) >= component_num_threshold]
        total_groups_num += len(cluster_groups)

        degrees = calculate_degrees(cos_sim_cluster_graph, cluster_groups)

        for idx, group in enumerate(cluster_groups):
            group_graph = cos_sim_cluster_graph[group, :][:, group]
            total_data_num += len(group)
            groups_record.append({
                'cluster': i,
                'group': idx,
                'data_num': len(group),
                'data_inner_indices': group,
                'data_indices': [cluster_indices[index] for index in group],
                'degrees': [degrees[index] for index in group],
                'group_graph': group_graph,
            })

    print(f"总有效组数量为{total_groups_num}, 涉及数据{total_data_num}")

    template_records = []
    test_records = []

    # 组数据拆分模板和测试
    for group in groups_record:
        # print(group['data_indices'])
        data_num = group['data_num']
        # for index, item in enumerate(group['data_indices']):
        #     print(raw_query_list[item], "连通度", group['degrees'][index])
        # 假设A是元素列表，B是整数列表

        combined = np.array(sorted(
            zip(group['data_indices'], group['degrees']), key=lambda x: x[1], reverse=True))
        template_idx = template_idx
        # top_two = [item[0] for item in combined[template_idx]]
        # top_two_degrees = [item[1] for item in combined[template_idx]]
        # print(top_two)
        # print(raw_query_list[top_two[0]], "连通度", top_two_degrees[0])
        # print(raw_query_list[top_two[1]], "连通度", top_two_degrees[1])
        template_records.append({
            'cluster': group['cluster'],
            'group': group['group'],
            'index': [int(item[0]) for item in combined[template_idx]],
            'degree': [int(item[1]) for item in combined[template_idx]]
        })

        # for item in combined[2:]:
        #     print(item[0],raw_query_list[item[0]], "连通度", item[1])
        test_idx = [i for i in range(data_num) if i not in template_idx]
        test_records.append({
            'cluster': group['cluster'],
            'group': group['group'],
            'index': [int(item[0]) for item in combined[test_idx]],
            'degree': [int(item[1]) for item in combined[test_idx]]
        })
    # 生成对应模板和测试数据
    template = []
    for record in template_records:
        for index, degree in zip(record['index'], record['degree']):
            template.append({
                'cluster': record['cluster'],
                'group': record['group'],
                'degree': int(degree),
                **{k: int(v) if isinstance(v, np.integer) else v for k, v in data[index].items()}
            })

    test = []
    for record in test_records:
        for index, degree in zip(record['index'], record['degree']):
            test.append({
                'cluster': record['cluster'],
                'group': record['group'],
                'degree': int(degree),
                **{k: int(v) if isinstance(v, np.integer) else v for k, v in data[index].items()}
            })

    return template, test
