import requests
import json
from typing import List, Union
import numpy as np
import os
from tenacity import retry, stop_after_attempt, wait_exponential
MAX_PROCESS_NUM = 256
TARGET_MODEL = 'qwen3-embedding-4b'
AuthorizationList = os.getenv("AUTHORIZATION_LIST", "").split(",")
EmbeddingModel = {
    'qwen3-embedding-0.6b': {
        'url': os.getenv("QWEN3_EMBEDDING_0_6B_URL"),
        'model': "qwen3-embedding-0.6b",
        'encoding_format': "float"
    },
    'qwen3-embedding-4b': {
        'url': os.getenv("QWEN3_EMBEDDING_4B_URL"),
        'model': "qwen3-embedding-4b",
        'encoding_format': "float"
    }
}


class EmbeddingAPIError(Exception):
    """Embedding 接口语义或结构异常"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1), reraise=True)
def _fetch_embedding_batch(model: str,
                           batch: List[str],
                           timeout: int = 10) -> List[List[float]]:
    """
    单批次请求（带自动重试）
    """
    url = EmbeddingModel[model]['url']
    headers = {'Content-Type': 'application/json'}
    if model == 'qwen3-embedding-0.6b':
        headers['Authorization'] = AuthorizationList[np.random.randint(
            0, len(AuthorizationList))]

    payload = {
        "model": model,
        "input": batch,
        "encoding_format": "float"
    }

    try:
        if model == 'qwen3-embedding-0.6b':
            resp = requests.post(url, data=json.dumps(
                payload), headers=headers, timeout=timeout)
        else:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EmbeddingAPIError(f"HTTP请求失败: {e}") from e

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise EmbeddingAPIError(f"响应JSON解析失败: {e}") from e

    if 'data' not in data or not data['data']:
        raise EmbeddingAPIError(f"响应缺少 data 字段或为空: {data}")
    if 'embedding' not in data['data'][0]:
        raise EmbeddingAPIError(f"响应缺少 embedding 字段: {data}")

    embeddings = [item['embedding'] for item in data['data']]

    cleaned = []
    for emb in embeddings:
        if any(np.isnan(x) for x in emb):
            emb = [0.0 if (isinstance(x, float) and np.isnan(x))
                   else x for x in emb]
        cleaned.append(emb)
    return cleaned


def get_embeddings(texts: List[str], model: str = TARGET_MODEL, batch_size=MAX_PROCESS_NUM) -> Union[List[List[float]], str]:
    """
    分批次获取文本embedding

    Args:
        texts: 需要编码的文本列表，自动分批次处理（每批最多128条）

    Returns:
        成功返回二维embedding列表，失败返回错误信息
    """
    url = EmbeddingModel[model]['url']
    headers = {
        'Content-Type': 'application/json',
    }
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"总文本数量: {len(texts)}", f"批次大小: {batch_size}")
    print("使用模型", model)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1

        # 简单的文本进度条
        progress = current_batch / total_batches
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(
            f'\r[{bar}] {progress:.0%} 批次 {current_batch}/{total_batches}', end='')
        try:
            embeddings = _fetch_embedding_batch(
                model=model,
                batch=batch
            )
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"\n批次 {current_batch} 最终失败: {e}")
            raise

    print()  # 换行
    return all_embeddings


def get_embeddings_by_model(texts, model, batch_size=64):
    all_embeddings = []
    model.eval()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"总文本数量: {len(texts)}", f"批次大小: {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        current_batch = i // batch_size + 1
        # 简单的文本进度条
        progress = current_batch / total_batches
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(
            f'\r[{bar}] {progress:.0%} 批次 {current_batch}/{total_batches}', end='')
        output = model(batch)
        output = output.tolist()
        all_embeddings.extend(output)
    return all_embeddings


def embedding_L2_normalization(embeddings: List[List[float]]):
    """
    L2归一化
    """
    embeddings_array = np.array(embeddings)
    # 计算每个embedding的L2范数
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    # 归一化
    embeddings_array = embeddings_array / norms
    return embeddings_array


def get_cosine_similarity_matrix(embeddings: List[List[float]]):
    """
    计算余弦相似度矩阵
    """
    embeddings_array = np.array(embeddings)
    # 计算余弦相似度
    cosine_sim = np.dot(embeddings_array, embeddings_array.T)
    return cosine_sim


def cosine_similarity(a, b):
    dot_product = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = sum(a[i] ** 2 for i in range(len(a))) ** 0.5
    norm_b = sum(b[i] ** 2 for i in range(len(b))) ** 0.5
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5
