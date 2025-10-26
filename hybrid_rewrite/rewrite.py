import json
import random
import json
import os
import time
import os.path as osp
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
import torch
from hybrid_rewrite.retriever import EmbeddingRetriever, BM25Retriever, LLMCache, DimensionValueRetriever
from hybrid_rewrite.util import safe_parse_json, compare_config, Voter, BinaryMetric, judge_json
from hybrid_rewrite.prompt import get_prompt
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from typing import List, Dict
from datetime import datetime
from llmtools import CLIENT_POOL
from openai import OpenAIError, RateLimitError
from utils import timeit, _TIMEIT_DATA, print_timeit_summary, save_timeit

LLM_CACHE = None
USE_LLM_CACHE = False
DIMENSION_VALUE_RETRIEVER = None
PATTERN_RETRIEVER_CACHE = {}
TIME_STATISTICS = True
T = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# log_path = Path(osp.join("log", f"evaluate_log_{T}.txt")).open("wt", encoding="utf-8", buffering=1)
# output_log_path = Path(osp.join("log", f"output_log_{T}.txt")).open(
#     "wt", encoding="utf-8", buffering=1)
# retrieved_log_path = osp.join("log", f"retrieved_{T}.json")
# ---------------------------------
SESSION = requests.Session()
_retries = Retry(
    total=0,  # 连接级别的自动重试交给 tenacity；这里设 0 避免与上层重复
    connect=3,
    read=0,
    status=0,
    backoff_factor=0.0,
)
SESSION.mount("https://", HTTPAdapter(pool_connections=20,
              pool_maxsize=50, max_retries=_retries))
SESSION.mount("http://", HTTPAdapter(pool_connections=20,
              pool_maxsize=50, max_retries=_retries))


# ------------- 配置 -------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_keywords", action="store_true")
    parser.add_argument("--keywords_for_dimvals", action="store_true")
    parser.add_argument("-d", "--dataset", type=str,
                        default=None)
    parser.add_argument("-t", "--test_path", type=str,
                        default=None)
    parser.add_argument("-e", "--example_path", type=str,
                        default=None)
    parser.add_argument("-m", "--model", type=str,
                        default="qwen3-32b")
    parser.add_argument("--emb_model", type=str,
                        default="qwen3-embedding-0.6b")
    parser.add_argument("--emb_model_path", type=str,
                        default=None)
    parser.add_argument("-p", "--prompt", type=str,
                        default="0_0_3")
    parser.add_argument("--pattern_key", type=str,
                        default="erased_query")
    parser.add_argument("--pattern_top_k", type=int,
                        default=10)
    parser.add_argument("--keyword_key", type=str,
                        default="erased_core")
    parser.add_argument("--keyword_top_k", type=int,
                        default=5)
    parser.add_argument("--dim_top_k", type=int,
                        default=1)
    return parser


# ========= 1. 数据加载 & 切分 =========
def load_split(json_path: str,
               split_ratio: float = 0.9,
               seed: int = 42,
               already_erased: bool = False,
               already_split: bool = False) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    """
    读取 json_path，返回 (q_dsl_data, test_set)
    q_dsl_data: [{"q": str, "dsl": dict}, ...]
    test_set  : [{"q": str, "config": dict}, ...]  # 作为标准答案
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    if not already_erased:
        for raw_case in raw:
            # 目前仅擦除test set（即online的输入），train set（索引）不擦除
            query = raw_case["query"].strip()
            erased_query = query
            config = raw_case["config"]
            rewritten = raw_case["rewritten"]["要素"]
            time_scales = rewritten.get("时间范围", [])
            if isinstance(time_scales, str):
                time_scales = [time_scales]
            filters = rewritten.get("筛选条件", [])

            for time_scale in time_scales:
                _scales = time_scale.split(" ")
                for scale in _scales:
                    erased_query = erased_query.replace(scale, "")
            for filter in filters:
                _filters = filter.split(" ")
                for filter_item in _filters:
                    fs = filter_item.split("=")
                    for f in fs:
                        erased_query.replace(f, "")

            samples.append(
                {"query": query, "erased_query": erased_query, "dsl": config}
            )
    elif already_erased:
        samples = [{"id": item.get("id", None), "project_name": item.get("project_name", None), "dataset_id": item.get("dataset_id", None),
                    "query": item["query"].strip(), "erased_query": item["erased"], "erased_hanlp": item.get("erased_hanlp", None),
                    "dsl": item["config"], "erased_dimension_value": item.get("erased_dimension_value"),
                    "erased_core": item.get("erased_core"), "erased_all": item.get("erased_all"), "predict_url": item.get("predict_url"),
                    "erased_ner": item.get("erased_ner"), "expect_url": item.get("expect_url", None)} for item in raw]

    # =================== 单一文件读取只返回一个set ===================
    if already_split:
        return samples

    # =================== 临时解法，针对当前数据不存在唯一id的情况 ==================
    for idx, sample in enumerate(samples):
        sample["id"] = idx

    random.seed(seed)
    random.shuffle(samples)
    split_idx = int(len(samples) * split_ratio)

    q_dsl_data = samples[:split_idx]
    test_set = samples[split_idx:]
    return q_dsl_data, test_set


# ========= 2. 预测 + 评测 =========
def call_diff_api(gt_url: str,
                  dataset_id: int,
                  analysis_elms: dict,
                  need_fe_config: bool = True,
                  timeout: int = 10,
                  max_retries: int = 2) -> dict:
    """
    调用 /mark/openapi/analysis/diff 接口
    """
    url = os.getenv("DIFF_API_URL")
    if not gt_url:
        return {"success": False, "error": "gt_url is required", "data": {}}
    payload = {
        "gtUrl": gt_url,
        "datasetId": dataset_id,
        "analysisElms": analysis_elms,
        "needFeConfig": need_fe_config
    }
    # try:
    # response = requests.post(url, json=payload, timeout=10)
    # response.raise_for_status()
    # return response.json()
    # except Exception as e:
    #     print(f"Diff API 调用失败: {e}")
    #     return {"error": str(e)}
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = SESSION.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # 兜底：接口不规范时也保证结构存在
            if not isinstance(data, dict):
                return {"success": False, "error": "Invalid JSON structure", "data": {}}
            return data
        except (requests.ReadTimeout, requests.ConnectTimeout) as e:
            # 指数退避：1,2,4,8...
            if attempt >= max_retries:
                return {"success": False, "error": f"timeout after {attempt} attempts: {e}", "data": {}}
            backoff = 2 ** (attempt - 1)
            print(f"[GT_URL] {gt_url}")
            print(
                f"[diff_api] timeout attempt {attempt}/{max_retries}, backoff {backoff}s")
            import time
            time.sleep(backoff)
        except requests.RequestException as e:
            # 不可恢复错误（4xx）直接返回
            status = getattr(e.response, "status_code", None)
            if status and 400 <= status < 500 and status not in (408, 429):
                return {"success": False, "error": f"non-retriable status {status}: {e}", "data": {}}
            if attempt >= max_retries:
                return {"success": False, "error": f"request failed after {attempt} attempts: {e}", "data": {}}
            backoff = 2 ** (attempt - 1)
            print(
                f"[diff_api] error attempt {attempt}/{max_retries}, backoff {backoff}s -> {e}")
            import time
            time.sleep(backoff)


@timeit()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
def query_llm(model, msgs, temperature, max_tokens):
    global LLM_CACHE
    resp = None
    if LLM_CACHE:
        is_hit, cache = LLM_CACHE.get_cache_by_params(
            prompt=msgs,
            temperature=temperature
        )
        if is_hit:
            return cache
    key, CLIENT = CLIENT_POOL.get_client()
    t0 = time.time()
    try:
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens
        )
        pred_str = resp.choices[0].message.content.strip()
        CLIENT_POOL.mark_success(key, latency=time.time() - t0)
        # 正常回答了才应该缓存
        if LLM_CACHE:
            _ = LLM_CACHE.set_cache_by_params(
                prompt=msgs,
                temperature=temperature,
                cache=pred_str
            )
    except RateLimitError:
        # 当前 key 降级冷却，再抛出触发 retry 自动换下一个
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        print(f"当前key {key} 请求被限流，已切换到下一个 API Key")
        raise
    except OpenAIError as e:
        # 其他错误也尝试换 key，并输出详细错误信息
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        status = getattr(getattr(e, "response", None), "status_code", None)
        # 有些错误对象可能有 .response.json()
        body = None
        if getattr(e, "response", None) is not None:
            try:
                body = e.response.json()
            except Exception:
                try:
                    body = e.response.text
                except Exception:
                    body = None
        print(
            f"当前key {key} 请求出错，已切换到下一个 API Key | "
            f"type={e.__class__.__name__} status={status} message={e} body={body}"
        )
        import traceback
        traceback.print_exc()
        raise
    # print(CLIENT_POOL._meta)
    return pred_str


# 这个是没有维值召回的预测
@timeit()
def predict(query_obj: dict,
            examples: List[Dict[str, any]],
            split_keywords: bool,
            pattern_key: str,
            pattern_top_k: int,
            keyword_key: str,
            keyword_top_k: int,
            model: str,
            emb_model: str,
            emb_model_path: str = None,
            temperature: float = 0.2,
            prompt_version: str = "0_0_4",
            merge_retrieval: bool = False,
            device=None) -> Dict[str, any]:
    global PATTERN_RETRIEVER_CACHE
    pattern_retriever = PATTERN_RETRIEVER_CACHE.get(emb_model_path, None)
    if pattern_retriever is None:
        pattern_retriever = EmbeddingRetriever(
            model=emb_model, model_path=emb_model_path)
        PATTERN_RETRIEVER_CACHE[emb_model_path] = pattern_retriever
    keyword_retriever = PATTERN_RETRIEVER_CACHE.get("keyword_retriever", None)
    if keyword_retriever is None:
        keyword_retriever = BM25Retriever()
        PATTERN_RETRIEVER_CACHE["keyword_retriever"] = keyword_retriever

    """单条预测，返回 dict"""
    pattern_retrieved_examples, pattern_retrieved_info = pattern_retriever.retrieve_topk(
        query_id=query_obj["id"],
        query=query_obj[pattern_key],
        examples=examples,
        top_k=pattern_top_k,
        use_cos=True,
        example_key=pattern_key
    )

    voter = Voter()
    for _template in pattern_retrieved_examples:
        voter.vote(_template[0]["dataset_id"], score=_template[1])
    voted_dataset_id_score_list = voter.get_top_k_voted_key(1)
    voted_dataset_id = int(voted_dataset_id_score_list[0][0]) if len(
        voted_dataset_id_score_list) > 0 else None
    keyword_retrieved_examples = []
    keyword_retrieved_info = []
    if keyword_key:
        if split_keywords:
            keywords = query_obj[keyword_key]
            for keyword in keywords:
                current_retrieval, current_retrieval_info = keyword_retriever.retrieve_topk(
                    query=keyword,
                    examples=examples,
                    top_k=keyword_top_k,
                    example_key=keyword_key
                )
                keyword_retrieved_examples += current_retrieval
                keyword_retrieved_info.append(current_retrieval_info)
        else:
            keyword_retrieved_examples, keyword_retrieved_info = keyword_retriever.retrieve_topk(
                query=query_obj[keyword_key],
                examples=examples,
                top_k=keyword_top_k,
                example_key=keyword_key
            )

        # merge
    all_retrieved = pattern_retrieved_examples + keyword_retrieved_examples
    merged_retrieval = None
    if merge_retrieval:
        unindexed_retrieval = []
        indexed_retrieval = {}
        for retrieval in all_retrieved:
            id = retrieval[0].get("id", None)
            if id:
                if id not in indexed_retrieval.keys():
                    indexed_retrieval[id] = retrieval
                else:
                    indexed_retrieval[id][1] += retrieval[1]
            else:
                unindexed_retrieval.append(retrieval)
        merged_retrieval = unindexed_retrieval + \
            [v for k, v in indexed_retrieval.items()]
        merged_retrieval = sorted(
            merged_retrieval, key=lambda case: case[1], reverse=True)
    elif not merge_retrieval:
        merged_retrieval = all_retrieved
    # remove score
    merged_retrieval = [_[0] for _ in merged_retrieval]

    # ============= 构建promp，可以**传入自定义参数 =============
    prompt = get_prompt(version_str=prompt_version)(
        query_obj["query"], merged_retrieval, k=pattern_top_k)
    msgs = [
        {"role": "system", "content": "你是一个专业的BI工具配置专家，你需要基于用户的问题完成对应指标的配置。"},
        {"role": "user", "content": prompt}
    ]

    pred_str = query_llm(model=model, msgs=msgs,
                         temperature=temperature, max_tokens=4096)
    pred = safe_parse_json(pred_str)

    if not judge_json(pred):
        print("LLM生成json解析出错")
        return {"pred": None, "raw": pred_str, "pred_dataset_id": voted_dataset_id, "retrieved": {"pattern": pattern_retrieved_info, "keyword": keyword_retrieved_info}}

    # log_obj = {
    #     "prompt": msgs,
    #     "pred": pred_str
    # }
    # print(json.dumps(log_obj, indent=2, ensure_ascii=False),
    #       "\n\n", file=output_log_path)
    return {"pred": pred, "raw": pred_str, "pred_dataset_id": voted_dataset_id, "retrieved": {"pattern": pattern_retrieved_info, "keyword": keyword_retrieved_info}}


# 这个是有维值召回的预测
@timeit()
def predict_dim_val(query_obj: dict,
                    examples: List[Dict[str, any]],
                    pattern_key: str,       # 模板在test/template文件里对应的key
                    pattern_top_k: int,     # 模板召回topk
                    dimval_key: str,        # 筛选条件在test/template文件里对应的key
                    dim_top_k: int,         # 字段召回topk
                    val_top_k: int,         # 维值召回topk
                    model: str,
                    emb_model: str,
                    emb_model_path: str = None,
                    temperature: float = 0.2,
                    prompt_version: str = "0_0_4",
                    device=None) -> Dict[str, any]:
    # 获取retriever
    global DIMENSION_VALUE_RETRIEVER
    if not DIMENSION_VALUE_RETRIEVER:
        DIMENSION_VALUE_RETRIEVER = DimensionValueRetriever()
    keyword_retriever = DIMENSION_VALUE_RETRIEVER
    global PATTERN_RETRIEVER_CACHE
    pattern_retriever = PATTERN_RETRIEVER_CACHE.get(emb_model_path, None)
    if pattern_retriever is None:
        pattern_retriever = EmbeddingRetriever(
            model=emb_model, model_path=emb_model_path)
        PATTERN_RETRIEVER_CACHE[emb_model_path] = pattern_retriever

    """基于主干问题的模板召回"""
    pattern_retrieved_examples, pattern_retrieved_info = pattern_retriever.retrieve_topk(
        query_id=query_obj["id"],
        query=query_obj[pattern_key],
        examples=examples,
        top_k=pattern_top_k,
        use_cos=True,
        example_key=pattern_key
    )
    # 投票获取召回模板中频数x相似度最大的dataset_id
    voter = Voter()
    for _template in pattern_retrieved_examples:
        voter.vote(_template[0]["dataset_id"], score=_template[1])
    voted_dataset_id_score_list = voter.get_top_k_voted_key(1)
    voted_dataset_id = int(voted_dataset_id_score_list[0][0]) if len(
        voted_dataset_id_score_list) > 0 else None
    # 获取模板问题,纯query
    retrieved_templates = [_[0] for _ in pattern_retrieved_examples]
    """基于关键词的维值召回"""
    keyword_retrieved_dimvals = []
    # 若不存在可用的dataset_id则关闭维值召回
    if voted_dataset_id:
        dimension_query_info = query_obj[dimval_key]
        if isinstance(dimension_query_info, dict):
            for dimension_name, value_names in dimension_query_info.items():
                current_retrieved_dimvals = keyword_retriever.retrieve_topk_with_dim_checking(
                    project_name=query_obj['project_name'],
                    # 使用投票得到的dataset_id作为维值召回锚点
                    dataset_id=voted_dataset_id,
                    dimension_name=dimension_name,
                    dimension_top_k=dim_top_k,
                    dimension_threshold=0.85,
                    value_names=value_names,
                    value_top_k=val_top_k,
                    value_threshold=0.9
                )
                keyword_retrieved_dimvals.extend(current_retrieved_dimvals)

        if isinstance(dimension_query_info, str):
            current_retrieved_dimvals = keyword_retriever.retrieve_topk_by_query(
                query=dimension_query_info,
                project_name=query_obj['project_name'],
                dataset_id=voted_dataset_id,
                top_k=val_top_k,
                threshold=0.9
            )
            keyword_retrieved_dimvals.extend(current_retrieved_dimvals)

        if isinstance(dimension_query_info, list):
            current_retrieved_dimvals = keyword_retriever.retrieve_topk_by_keywords(
                project_name=query_obj['project_name'],
                dataset_id=voted_dataset_id,
                value_names=dimension_query_info,
                value_top_k=val_top_k,
                value_threshold=0.9
            )
            keyword_retrieved_dimvals.extend(current_retrieved_dimvals)
    # ============= 构建promp，**传入自定义参数 =============
    prompt = get_prompt(version_str=prompt_version)(
        query_obj["query"], retrieved_templates, k=pattern_top_k, dimvals=keyword_retrieved_dimvals)
    msgs = [
        {"role": "system", "content": "你是一个专业的BI工具配置专家，你需要基于用户的问题完成对应指标的配置。"},
        {"role": "user", "content": prompt}
    ]

    pred_str = query_llm(model=model, msgs=msgs,
                         temperature=temperature, max_tokens=4096)
    pred = safe_parse_json(pred_str)
    if not judge_json(pred):
        print("LLM生成json解析出错")
        return {"pred": None, "raw": pred_str, "pred_dataset_id": voted_dataset_id, "retrieved": {"pattern": pattern_retrieved_info, "dimvals": keyword_retrieved_dimvals}}

    return {"pred": pred, "raw": pred_str, "pred_dataset_id": voted_dataset_id, "retrieved": {"pattern": pattern_retrieved_info, "dimvals": keyword_retrieved_dimvals}}


def predict_dim_val_by_difference(query_obj: dict,
                                  examples: List[Dict[str, any]],
                                  pattern_key: str,   # 模板在test/template文件里对应的key
                                  pattern_top_k: int,  # 模板召回topk
                                  val_top_k: int,     # 维值召回topk
                                  model: str,
                                  emb_model: str,
                                  emb_model_path: str = None,
                                  temperature: float = 0.2,
                                  prompt_version: str = "0_0_4") -> Dict[str, any]:
    global DIMENSION_VALUE_RETRIEVER
    if not DIMENSION_VALUE_RETRIEVER:
        DIMENSION_VALUE_RETRIEVER = DimensionValueRetriever()
    keyword_retriever = DIMENSION_VALUE_RETRIEVER
    global PATTERN_RETRIEVER_CACHE
    pattern_retriever = PATTERN_RETRIEVER_CACHE.get(emb_model_path, None)
    if pattern_retriever is None:
        pattern_retriever = EmbeddingRetriever(
            model=emb_model, model_path=emb_model_path)
        PATTERN_RETRIEVER_CACHE[emb_model_path] = pattern_retriever

    """基于主干问题的模板召回"""
    pattern_retrieved_examples, pattern_retrieved_info = pattern_retriever.retrieve_topk(
        query_id=query_obj["id"],
        query=query_obj[pattern_key],
        examples=examples,
        top_k=pattern_top_k,
        use_cos=True,
        example_key=pattern_key
    )
    # 投票获取召回模板中频数x相似度最大的dataset_id
    voter = Voter()
    for _template in pattern_retrieved_examples:
        voter.vote(_template[0]["dataset_id"], score=_template[1])
    voted_dataset_id_score_list = voter.get_top_k_voted_key(1)
    voted_dataset_id = int(voted_dataset_id_score_list[0][0]) if len(
        voted_dataset_id_score_list) > 0 else None
    # remove score
    retrieved_templates = [_[0] for _ in pattern_retrieved_examples]

    # 通过和模板作差得到raw keywords

    """基于关键词的维值召回"""
    keyword_retrieved_dimvals = []
    # 若不存在可用的dataset_id则关闭维值召回
    if voted_dataset_id:
        pass
        # for dimension_name, value_names in query_obj[dimval_key].items():
        #     current_retrieved_dimvals = keyword_retriever.retrieve_topk_with_dim_checking(
        #         project_name=query_obj['project_name'],
        #         # 使用投票得到的dataset_id作为维值召回锚点
        #         dataset_id=voted_dataset_id,
        #         dimension_name=dimension_name,
        #         dimension_top_k=dim_top_k,
        #         dimension_threshold=0.85,
        #         value_names=value_names,
        #         value_top_k=val_top_k,
        #         value_threshold=0.9
        #     )
        #     keyword_retrieved_dimvals += current_retrieved_dimvals

    # ============= 构建promp，**传入自定义参数 =============
    prompt = get_prompt(version_str=prompt_version)(
        query_obj["query"], retrieved_templates, k=pattern_top_k, dimvals=keyword_retrieved_dimvals)
    msgs = [
        {"role": "system", "content": "你是一个专业的BI工具配置专家，你需要基于用户的问题完成对应指标的配置。"},
        {"role": "user", "content": prompt}
    ]

    pred_str = query_llm(model=model, msgs=msgs,
                         temperature=temperature, max_tokens=4096)
    pred = safe_parse_json(pred_str)

    log_obj = {
        "prompt": msgs,
        "pred": pred_str
    }
    # print(json.dumps(log_obj, indent=2, ensure_ascii=False),
    #       "\n\n", file=output_log_path)
    return {"pred": pred, "raw": pred_str, "retrieved": {"pattern": pattern_retrieved_info, "dimvals": keyword_retrieved_dimvals}}


def evaluate(json_path: str,
             test_path: str,
             example_path: str,
             name: str = 'unnamed',
             split_ratio: float = 0.9,
             split_keywords: bool = False,
             keywords_for_dimvals: bool = False,
             pattern_key: str = "erased_query",
             pattern_top_k: int = 10,
             keyword_key: str = "erased_core",
             dim_top_k: int = 1,
             keyword_top_k: int = 5,
             model: str = "qwen2.5-72b-instruct",
             emb_model: str = "qwen3-embedding-0.6b",
             emb_model_path: str = None,
             prompt_version: str = "0_0_8",
             temperature: float = 0.2,
             merge_retrieval: bool = False,
             device=None):
    global LLM_CACHE
    if USE_LLM_CACHE:
        LLM_CACHE = LLMCache(model_code=model)
    # log_path = Path(osp.join(
    #     "log", f"eval_pattern-{pattern_key}-{pattern_top_k}_keyword-{keyword_key}-{keyword_top_k}_{T}.txt")).open("wt", encoding="utf-8", buffering=1)
    examples, test_set = None, None
    dataset_name = "default"
    # 这里是如果不是直接传入的json文件，而是传入的test/template文件，则不进行切分
    if json_path:
        examples, test_set = load_split(
            json_path, split_ratio, already_erased=dataset_name.endswith("erased"))
        dataset_name = osp.basename(json_path).replace(".json", "")
        split_data_dir = osp.join("data", dataset_name)
        if not osp.exists(split_data_dir):
            os.makedirs(split_data_dir)
        with open(osp.join(split_data_dir, f"train_q_dsl.json"), "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        with open(osp.join(split_data_dir, f"test_set.json"), "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        print(f"训练条数：{len(examples)}，测试条数：{len(test_set)}")
    elif test_path and example_path:
        examples = load_split(example_path, None,
                              already_erased=True, already_split=True)
        test_set = load_split(
            test_path, None, already_erased=True, already_split=True)
        dataset_name = osp.basename(test_path).replace(".json", "")
        print(f"训练条数：{len(examples)}，测试条数：{len(test_set)}")

    dslmatch_metric = BinaryMetric()
    dataset_metric = BinaryMetric()
    dimension_metric = BinaryMetric()
    measure_metric = BinaryMetric()
    filter_metric = BinaryMetric()
    exec_metric = BinaryMetric()
    total_metric = BinaryMetric()
    diffapi_exec_metric = BinaryMetric()

    results = []
    all_retrieved = []
    evaluate_times = []
    for idx, gold in enumerate(test_set):
        start = time.perf_counter()
        print(f"执行中{name}: {idx + 1}/{len(test_set)}")
        res = None
        if keywords_for_dimvals:
            res = predict_dim_val(
                query_obj=gold,
                examples=examples,
                pattern_key=pattern_key,
                pattern_top_k=pattern_top_k,
                dimval_key=keyword_key,
                dim_top_k=dim_top_k,
                val_top_k=keyword_top_k,
                model=model,
                emb_model=emb_model,
                emb_model_path=emb_model_path,
                temperature=temperature,
                prompt_version=prompt_version,
                device=device)
        else:
            res = predict(
                query_obj=gold,
                examples=examples,
                split_keywords=split_keywords,
                pattern_key=pattern_key,
                pattern_top_k=pattern_top_k,
                keyword_key=keyword_key,
                keyword_top_k=keyword_top_k,
                model=model,
                emb_model=emb_model,
                emb_model_path=emb_model_path,
                temperature=temperature,
                prompt_version=prompt_version,
                merge_retrieval=merge_retrieval,
                device=device)
        all_retrieved.append(res["retrieved"])
        analysis_elms = res.get("pred", None)
        if analysis_elms is None:
            print(f"LLM解析json失败，跳过该例")
            continue
        # 选表正确率
        gt_dataset_id = gold["dataset_id"]
        pred_dataset_id = res["pred_dataset_id"]
        dataset_match = dataset_metric.compare_id(pred_dataset_id, gt_dataset_id)

        # 精确匹配：预测 DSL == 标准答案
        match, dim_match, mes_match, fil_match = compare_config(res["pred"], gold["dsl"], strict_mode=False)
        match = match and dataset_match

        # --- 新增：调用 diff API ---
        gt_url = gold.get("predict_url", "")
        analysis_elms = res.get("pred", {})
        predict_result = call_diff_api(gt_url, int(pred_dataset_id), {
            "维度列表": analysis_elms.get("dimension", []),
            "指标列表": analysis_elms.get("measure", []),
            "筛选条件": analysis_elms.get("filter", [])
        })
        diff_result = predict_result

        def merge_diff_result(predict_result, expect_result):
            # diff的是错误的,那么就返回expect_result
            if not (predict_result and predict_result['success'] and predict_result['data']['diffReport']):
                return expect_result
            # expect的是错误的,就返回diff的
            if not (expect_result and expect_result['success'] and expect_result['data']['diffReport']):
                return predict_result

            for key in ['dimension', 'measure', 'filter']:
                predict_result['data']['diffReport'][key +
                                                     'Correct'] |= expect_result['data']['diffReport'][key + 'Correct']
            predict_result['data']['diffReport']['correct'] |= expect_result['data']['diffReport']['correct']
            return predict_result

        expect_result = {}
        if gold.get("expect_url"):
            expect_result = call_diff_api(gold["expect_url"], int(pred_dataset_id), {
                "维度列表": analysis_elms.get("dimension", []),
                "指标列表": analysis_elms.get("measure", []),
                "筛选条件": analysis_elms.get("filter", [])
            })
            diff_result = merge_diff_result(predict_result, expect_result)

        # success_call_diff = diff_result['success']
        if diff_result and diff_result['success'] and diff_result['data']['diffReport']:
            diffapi_exec_metric.update(True)
            diff_report = diff_result['data']['diffReport']

            exec_metric.update(diff_report['correct'])
            total_metric.update(
                match or diff_report['correct'])
            dslmatch_metric.update(match)
            dimension_metric.update(
                dim_match or diff_report['dimensionCorrect'])
            measure_metric.update(mes_match or diff_report['measureCorrect'])
            filter_metric.update(fil_match or diff_report['filterCorrect'])
        else:
            diffapi_exec_metric.update(False)
            exec_metric.update(False)
            total_metric.update(match)
            dslmatch_metric.update(match)
            dimension_metric.update(dim_match)
            measure_metric.update(mes_match)
            filter_metric.update(fil_match)
        # --- end ---
        elapsed = time.perf_counter() - start
        evaluate_times.append(elapsed)
        _TIMEIT_DATA[f'{name}_evaluate_times'] = evaluate_times
        results.append({"query": gold["query"],
                        "gold": gold["dsl"],
                        "pred": res["pred"],
                        "match": match,
                        "predict_result": predict_result,
                        "expect_result": expect_result})
        # print(f"✅" if match else "❌",
        #       f"Q: {gold['query']}\n   GOLD: {gold['dsl']}\n   PRED: {res['pred']}\n",
        #       f"dimension: {'✅' if dim_match else '❌'}, measure: {'✅' if mes_match else '❌'}, filter: {'✅' if fil_match else '❌'}\n"
        #       f"✅" if dataset_match else "❌",
        #       f"GOLD dataset: {gt_dataset_id}   PRED: {pred_dataset_id}\n",
        #       f"Current accuracy: {exec_metric.acc()} ({exec_metric.stat()})\n",
        #       f"    - Dimension acc: {dimension_metric.acc():.2f} ({dimension_metric.stat()})\n",
        #       f"    - Measure acc: {measure_metric.acc():.2f} ({measure_metric.stat()})\n",
        #       f"    - Filter acc: {filter_metric.acc():.2f} ({filter_metric.stat()})\n",
        #       f"Current dataset acc: {dataset_metric.acc()} ({dataset_metric.stat()})\n")
        # print(f"✅" if match else "❌",
        #       f"Q: {gold['query']}\n   GOLD: {gold['dsl']}\n   PRED: {res['pred']}\n",
        #       f"dimension: {'✅' if dim_match else '❌'}, measure: {'✅' if mes_match else '❌'}, filter: {'✅' if fil_match else '❌'}\n"
        #       f"✅" if dataset_match else "❌",
        #       f"GOLD dataset: {gt_dataset_id}   PRED: {pred_dataset_id}\n",
        #       f"Current accuracy: {exec_metric.acc()} ({exec_metric.stat()})\n",
        #       f"    - Dimension acc: {dimension_metric.acc():.2f} ({dimension_metric.stat()})\n",
        #       f"    - Measure acc: {measure_metric.acc():.2f} ({measure_metric.stat()})\n",
        #       f"    - Filter acc: {filter_metric.acc():.2f} ({filter_metric.stat()})\n",
        #       f"Current dataset acc: {dataset_metric.acc()} ({dataset_metric.stat()})\n")

    # dsl_acc = dslmatch_metric.acc()
    # dataset_acc = dataset_metric.acc()
    # stat = exec_metric.stat()
    # dataset_stat = dataset_metric.stat()
    # print(f"\nAccuracy = {acc:.4f} ({stat})\nDataset accuracy = {dataset_acc:.4f} ({dataset_stat})")
    # print(f"\nAccuracy = {acc:.4f} ({stat})\nDataset accuracy = {dataset_acc:.4f} ({dataset_stat})", file=log_path)
    # json.dump(all_retrieved, open(retrieved_log_path, 'w',
    #           encoding="utf-8"), ensure_ascii=False, indent=2)
    if TIME_STATISTICS:
        print_timeit_summary()
        save_timeit(f"./log/timeit/times.json")
    return {"results": results,
            "dsl_acc": dslmatch_metric.acc(),
            "exec_acc": exec_metric.acc(),
            "total_acc": total_metric.acc(),
            "dataset_acc": dataset_metric.acc(),
            "dimension_acc": dimension_metric.acc(),
            "measure_acc": measure_metric.acc(),
            "filter_acc": filter_metric.acc(),
            "diffapi_exec_acc": diffapi_exec_metric.acc(),
            "retrieved": all_retrieved}


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    evaluate(json_path=args.dataset,
             test_path=args.test_path,
             example_path=args.example_path,
             prompt_version=args.prompt,
             split_ratio=0.9,
             split_keywords=args.split_keywords,
             keywords_for_dimvals=args.keywords_for_dimvals,
             pattern_key=args.pattern_key,
             pattern_top_k=args.pattern_top_k,
             keyword_key=args.keyword_key,
             keyword_top_k=args.keyword_top_k,
             dim_top_k=args.dim_top_k,
             model=args.model,
             emb_model=args.emb_model,
             emb_model_path=args.emb_model_path,
             temperature=0.1,
             merge_retrieval=False,
             device="cuda:0" if torch.cuda.is_available() else "cpu")
