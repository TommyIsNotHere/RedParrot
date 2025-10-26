import json
import torch
import requests
import time
import jieba
import os

from rank_bm25 import BM25Okapi
from sqlitedict import SqliteDict
from copy import deepcopy
from typing import List, Dict, Union, Tuple

from entity_agnostic_embedding_model.teacher_student import MARLOEmbeddingModel
from entity_agnostic_embedding_model.contrastive_learning import ContrastiveMARLOEmbeddingModel

from template_construct.embed_tool import get_embeddings, get_embeddings_by_model

from hybrid_rewrite.serving.recall_service import dim_member_retriever_v2, dataset_field_retriever_v2, redbi_api_service
from hybrid_rewrite.serving.searcher.recall_models import RecallFilterRequest
from hybrid_rewrite.serving.dataset_selection_v2 import DatasetRecallInfo

NEED_AUTHORIZATION_ENDPOINT = os.getenv("NEED_AUTHORIZATION_ENDPOINT", "")
EMB_KEY = os.getenv("EMB_KEY", "")
MODEL2URL = {
    "qwen3-embedding-0.6b": os.getenv("QWEN3_EMBEDDING_0_6B_ENDPOINT", ""),
    "qwen3-embedding-4b": os.getenv("QWEN3_EMBEDDING_4B_ENDPOINT", "")
}
EmbeddingModelList = ["qwen3-embedding-0.6b",
                      "qwen3-embedding-4b",
                      "google-bert/bert-base-uncased",
                      "Qwen/Qwen3-Embedding-0.6B",
                      "teacher-student"]
ContrastiveModel = ["Qwen/Qwen3-Embedding-0.6B",
                    "google-bert/bert-base-uncased"]
ApiModelList = ["qwen3-embedding-0.6b",
                "qwen3-embedding-4b"]


class LLMCache():
    def __init__(self, model_code):
        self._cache = SqliteDict(
            "llm_cache.sqlite", tablename=model_code, autocommit=True)

    def get_cache(self, query: str) -> Tuple[bool, str]:
        if not isinstance(query, str):
            query = json.dumps(query)
        cache_hit = self._cache.get(query, None)
        if cache_hit:
            cache_hit = str(cache_hit)
        return cache_hit is not None, cache_hit

    def set_cache(self, query: str, cache: any, overwrite: bool = False) -> bool:
        if not isinstance(query, str):
            query = json.dumps(query)
        if not overwrite:
            # try to get
            cache_hit = self._cache.get(query, None)
            if cache_hit:
                return False
        self._cache[query] = cache
        return True

    def _build_key_(self, prompt: any, temperature: float) -> str:
        key_json = {
            "prompt": prompt,
            "temperature": temperature
        }
        return json.dumps(key_json, ensure_ascii=False)

    def get_cache_by_params(self, prompt: any, temperature: float) -> Tuple[bool, str]:
        is_hit, cache = self.get_cache(self._build_key_(
            prompt=prompt, temperature=temperature))
        return is_hit, cache

    def set_cache_by_params(self, prompt: any, temperature: float, cache: any, overwrite: bool = False) -> bool:
        cache_key = self._build_key_(prompt=prompt, temperature=temperature)
        updated = self.set_cache(cache_key, cache, overwrite=overwrite)
        return updated


class EmbeddingEncoder():
    def __init__(self, model: str, model_path=None, use_cache=None):
        """
        Args:
            model: 支持的模型见 EmbeddingModelList
            model_path: 如果是本地模型则指定模型路径
            use_cache: 是否使用缓存，None表示自动判断（API模型使用缓存，本地模型不使用缓存）"""
        assert model in EmbeddingModelList, f"model {model} not in {EmbeddingModelList}"
        self._model = model
        self._need_authorization = False
        self.use_cache = use_cache
        if use_cache is None:
            if self._model in ApiModelList:
                # 是api的话就使用cache
                self.use_cache = True
            else:
                self.use_cache = False

        if self.use_cache:
            self._cache_hit = 0
            _file_name = f"{model.replace('/', '_')}.sqlite"
            if model in ApiModelList:
                self._cache = SqliteDict(
                    _file_name, autocommit=True)
            else:
                self._cache = SqliteDict(
                    _file_name, tablename=model_path.replace('/', '_'), autocommit=True)

        self._query_count = 0
        self._model_path = model_path
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if self._model == "teacher-student":
            self._embedding_model = MARLOEmbeddingModel(
                'sentence-transformers/all-MiniLM-L6-v2', 2560)
            if self._model_path:
                self._embedding_model.load_state_dict(
                    torch.load(self._model_path))
            self._embedding_model = self._embedding_model.to(device)
            self._embedding_model.eval()

        elif self._model in ContrastiveModel:
            self._embedding_model = ContrastiveMARLOEmbeddingModel(base_model=self._model, device=device)
            if self._model_path:
                self._embedding_model.load_state_dict(
                    torch.load(self._model_path))

            self._embedding_model = self._embedding_model.to(device)
            self._embedding_model.eval()

    def _get_embeddings(self, texts: List[str], batch_size=192) -> Union[List[List[float]], str]:
        """
        分批次获取文本embedding

        Args:
            texts: 需要编码的文本列表，自动分批次处理（每批最多128条）
        Returns:
            成功返回二维embedding列表，失败返回错误信息
        """
        if self._model in ["teacher-student", *ContrastiveModel]:
            return get_embeddings_by_model(texts, self._embedding_model)
        elif self._model in ApiModelList:
            url = MODEL2URL[self._model]
            headers = {
                'Content-Type': 'application/json'
            }
            if url == NEED_AUTHORIZATION_ENDPOINT:
                self._need_authorization = True
                headers["Authorization"] = EMB_KEY

            all_embeddings = []
            print(f"总文本数量: {len(texts)}", f"批次大小: {batch_size}")
            for i in range(0, len(texts), batch_size):
                batch = texts[i:min(i+batch_size, len(texts))]
                print(f"处理批次 {i//batch_size + 1}: {len(batch)} 条文本")
                llm_req = {
                    "model": self._model,
                    "input": batch,
                    "encoding_format": "float"
                }
                response = requests.post(
                    url, data=json.dumps(llm_req), headers=headers)
                response.raise_for_status()
                resp_json = response.json()
                if 'data' not in resp_json:
                    raise ValueError(f"API响应格式错误: {resp_json}")

                all_embeddings.extend([item['embedding']
                                      for item in resp_json['data']])

            return all_embeddings

    def encode(self, query: str, return_tensor: bool = True):
        """带流量控制的encode，获取输入的向量表示"""
        self._query_count += 1
        if self.use_cache:
            _cached_emb = self._cache.get(query)
            if _cached_emb:
                self._cache_hit += 1
                _emb = json.loads(_cached_emb)
                return torch.tensor(_emb) if return_tensor else _emb
            else:
                if self._need_authorization:
                    sleep_sec = 0.5
                    print(
                        f"Didn't hit any cache, sleep for {sleep_sec} seconds!")
                    time.sleep(sleep_sec)
                emb = self._get_embeddings([query])
                self._cache[query] = json.dumps(emb, ensure_ascii=False)
                print(f"Current cache hit rate: {self.cache_hit_rate():.4f}")
                return torch.tensor(emb) if return_tensor else emb
        else:
            emb = self._get_embeddings([query])
            return torch.tensor(emb) if return_tensor else emb

    def batch_encode(self, query_list: List[str], return_tensor: bool = True):
        if self.use_cache:
            uncached_index_list = []
            uncached_query_list = []
            emb_list = []
            for idx, query in enumerate(query_list):
                # probe cache
                cached = self._cache.get(query)
                if not cached:
                    uncached_index_list.append(idx)
                    uncached_query_list.append(query)
                    emb_list.append([])
                elif cached:
                    _cached_emb = json.loads(cached)
                    if isinstance(_cached_emb, list) and len(_cached_emb) == 1:
                        _cached_emb = _cached_emb[0]
                    emb_list.append(_cached_emb)

            # query uncached embedding
            if len(uncached_index_list) > 0:
                uncached_emb_list = self._get_embeddings(uncached_query_list)
                # write to cache
                for idx, uncached_emb in enumerate(uncached_emb_list):
                    write_back_idx = uncached_index_list[idx]
                    write_back_query = uncached_query_list[idx]
                    emb_list[write_back_idx] = uncached_emb
                    self._cache[write_back_query] = json.dumps(
                        uncached_emb, ensure_ascii=False)

            # print(f"Current cache hit rate: {self.cache_hit_rate():.4f}")
            return torch.tensor(emb_list) if return_tensor else emb_list
        else:
            emb = self._get_embeddings(query_list)
            return torch.tensor(emb) if return_tensor else emb

    def cache_hit_rate(self):
        if self.use_cache:
            return float(self._cache_hit) / self._query_count
        else:
            return "no cache"

    def close(self):
        if self.use_cache:
            self._cache.close()


# ========= 2.1 Embedding =========
class EmbeddingRetriever():
    def __init__(self, model: str = "qwen3-embedding-0.6b", model_path: str = None):
        self.model_path = model_path
        self.encoder = EmbeddingEncoder(
            model=model, model_path=model_path, use_cache=False)
        self.cache = {}

    def retrieve_topk(self, query_id,
                      query: str,
                      examples: List[Dict[str, str]],
                      top_k: int,
                      use_cos: bool = False,
                      example_key: str = "erased_query") -> List[Dict[str, str]]:
        """返回与 query 最相似的 top_k 条 example"""
        if top_k <= 0:
            return [], {
                "id": query_id,
                "query": query,
                "top-k": top_k,
                "example_key": example_key,
                "retrieved": []
            }
        example_input_list = [example[example_key] for example in examples]
        key = f"{len(example_input_list)}_{self.model_path}_{example_input_list[0]}_{example_input_list[-1]}"
        if key in self.cache:
            example_input_emb = self.cache[key]
        else:
            example_input_emb = self.encoder.batch_encode(example_input_list)
            self.cache[key] = example_input_emb
        query_emb = self.encoder.encode(query)
        if use_cos:
            example_input_emb = torch.nn.functional.normalize(
                example_input_emb, p=2, dim=-1)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        scores = (example_input_emb @ query_emb.T).squeeze()
        top_idx = torch.topk(scores, top_k, largest=True, sorted=True).indices
        retrieved = [[examples[i],
                      float(scores[i])] for i in top_idx]
        retrieved_obj = {
            "id": query_id,
            "query": query,
            "top-k": top_k,
            "example_key": example_key,
            "retrieved": retrieved
        }
        return retrieved, retrieved_obj

# ========= 2.2 BM25 =========


class BM25Retriever():
    def __init__(self):
        return

    def _tokenize(self, text: str) -> List[str]:
        raw_input = deepcopy(text)
        if isinstance(raw_input, list):
            text = " ".join(raw_input)
        elif isinstance(raw_input, dict):
            text = json.dumps(raw_input, ensure_ascii=False)
        else:
            text = str(raw_input)
        return list(jieba.cut(text, cut_all=False))

    def _build_bm25(self, examples: List[Dict[str, any]], example_key: str) -> BM25Okapi:
        raw_corpus = [item[example_key] for item in examples]
        tokenized_corpus = [self._tokenize(q) for q in raw_corpus]
        return BM25Okapi(tokenized_corpus), raw_corpus

    def retrieve_topk(self,
                      query,
                      examples: List[Dict[str, str]],
                      top_k: int,
                      example_key: str = "erased_query",
                      normalize: bool = True) -> List[Dict[str, str]]:
        if top_k <= 0:
            return [], {
                "query": query,
                "top-k": top_k,
                "example_key": example_key,
                "retrieved": []
            }

        bm25, raw_corpus = self._build_bm25(examples, example_key=example_key)
        bm25: BM25Okapi
        scores = bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:top_k]
        if normalize:
            normalized_scores = (scores - scores.min()) / \
                (scores.max() - scores.min())
            scores = normalized_scores
        retrieved = [[examples[i], float(scores[i])] for i in top_idx]
        retrieved_obj = {
            "query": query,
            "top-k": top_k,
            "example_key": example_key,
            "retrieved": retrieved
        }
        return retrieved, retrieved_obj


# ========= 2.3 Dimension Values =========
class DimensionValueRetriever():
    def __init__(self):
        return

    def retrieve_topk_by_query(self,
                      query: str,
                      project_name: str,
                      dataset_id: int,
                      top_k: int,
                      threshold: float) -> List:
        if not isinstance(dataset_id, int):
            dataset_id = int(dataset_id)
        dim_val_recalled = dim_member_retriever_v2.search(
            project=project_name, query=query, query_emd=None,
            dataset_id=dataset_id,
            search_top_k=top_k,
            score_threshold=threshold,
            filters=RecallFilterRequest().add_expr(f"enable_recall == true"),
            strategy="custom_hybrid_with_text_sim"
        )
        return dim_val_recalled

    def retrieve_topk_with_dim_checking(self,
                                        project_name: str,
                                        dataset_id: int,
                                        dimension_name: str,
                                        dimension_top_k: int,
                                        dimension_threshold: float,
                                        value_names: List[str],
                                        value_top_k: int,
                                        value_threshold: float) -> List:

        if not isinstance(dataset_id, int):
            dataset_id = int(dataset_id)

        table_info = redbi_api_service.get_table_info_by_id(
            dataset_id, None, project_name)
        recall_info = DatasetRecallInfo(project_name, dataset_id, table_info)

        # 根据筛选条件左侧维度名dimension_name召回对应字段
        field_docs = dataset_field_retriever_v2.search(
            project=project_name,
            dataset_id=dataset_id,
            query=dimension_name,
            query_emd=None,
            search_top_k=dimension_top_k,
            group_by_field="field_id",
            score_threshold=0,
            filters=RecallFilterRequest().add_expr("field_role == 'Dimension'"),
            strategy="hybrid_with_emd_score"
        )
        recall_info.add_field_doc_list(field_docs)
        available_dimension_id_field_name_dict = {}
        for field_id, doc_score_recall in recall_info.field_id_score_map.items():
            doc, score, recall = doc_score_recall
            meta = doc.metadata
            # print(meta)
            # 用threshold卡score
            if score > dimension_threshold:
                available_dimension_id_field_name_dict[meta["dimension_id"]
                                                       ] = meta["name"]

        # 根据筛选条件右侧维值名召回对应维值
        retrieved_dim_vals = []
        for value_name in value_names:
            dimval_recall_info = DatasetRecallInfo(
                project_name, dataset_id, table_info)
            values_docs = dim_member_retriever_v2.search(
                project=project_name,
                query=value_name,
                query_emd=None,
                dataset_id=dataset_id,
                search_top_k=value_top_k,
                score_threshold=value_threshold,
                filters=RecallFilterRequest().add_expr(f"enable_recall == true"),
                strategy="custom_hybrid_with_text_sim"
            )
            dimval_recall_info.add_dim_doc_list(values_docs)

            # 只保留对应维度存在在上一步字段召回中的维值
            current_available_dim_vals = []
            for dimval_id, doc_score_recall in dimval_recall_info.dim_score_map.items():
                doc, score, recall = doc_score_recall
                meta = doc.metadata
                dim_id = meta["dimension_id"]
                # print(meta)
                # print(meta["field_alias"], dim_id, meta["text"])
                available_dimension_name = available_dimension_id_field_name_dict.get(
                    dim_id, None)
                if available_dimension_name:
                    current_available_dim_vals.append(
                        (available_dimension_name, meta["text"]))
            # 对每个<dim, val>的召回结果只保留第一个，避免出现“用户-搜索”同时召回“用户-搜索”和别名“用搜”的情况
            retrieved_dim_vals += current_available_dim_vals[:1 if len(
                current_available_dim_vals) > 1 else -1]
        return retrieved_dim_vals

    def retrieve_topk_by_keywords(self,
                                project_name: str,
                                dataset_id: int,
                                value_names: List[str],
                                value_top_k: int,
                                value_threshold: float) -> List:
        table_info = redbi_api_service.get_table_info_by_id(
            dataset_id, None, project_name)
        retrieved_dim_vals = []
        for value_name in value_names:
            dimval_recall_info = DatasetRecallInfo(
                project_name, dataset_id, table_info)
            values_docs = dim_member_retriever_v2.search(
                project=project_name,
                query=value_name,
                query_emd=None,
                dataset_id=dataset_id,
                search_top_k=value_top_k,
                score_threshold=value_threshold,
                filters=RecallFilterRequest().add_expr(f"enable_recall == true"),
                strategy="custom_hybrid_with_text_sim"
            )
            dimval_recall_info.add_dim_doc_list(values_docs)

            # 只保留对应维度存在在上一步字段召回中的维值
            current_available_dim_vals = []
            for dimval_id, doc_score_recall in dimval_recall_info.dim_score_map.items():
                doc, score, recall = doc_score_recall
                meta = doc.metadata
                dim_id = meta["dimension_id"]
                # print(meta)
                # print(meta["field_alias"], dim_id, meta["text"])
                # available_dimension_name = available_dimension_id_field_name_dict.get(
                #     dim_id, None)
                # if available_dimension_name:
                #     current_available_dim_vals.append(
                #         (available_dimension_name, meta["text"]))
                current_available_dim_vals.append(
                    (dim_id, meta["text"]))
            # 对每个<dim, val>的召回结果只保留第一个，避免出现“用户-搜索”同时召回“用户-搜索”和别名“用搜”的情况
            retrieved_dim_vals += current_available_dim_vals[:1 if len(
                current_available_dim_vals) > 1 else -1]
        return retrieved_dim_vals
