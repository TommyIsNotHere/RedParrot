import json
import random
import json
import os
import time
import os.path as osp
import argparse
from requests.adapters import HTTPAdapter, Retry
from hybrid_rewrite.retriever import EmbeddingRetriever, BM25Retriever, LLMCache, DimensionValueRetriever
from hybrid_rewrite.util import safe_parse_json, compare_config, Voter, BinaryMetric, dict_schema_subset
from hybrid_rewrite.prompt import get_prompt
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from typing import List, Dict
from datetime import datetime
from llmtools import CLIENT_POOL
from openai import OpenAIError, RateLimitError
from utils import timeit

LLM_CACHE = None
DIMENSION_VALUE_RETRIEVER = None
PATTERN_RETRIEVER_CACHE = {}
T = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SPIDER_SCHEMA_CACHE = {}
BIRD_SCHEMA_CACHE = {}


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
        samples = [{"id": int(item.get("id", None)),
                    "db_id": item.get("db_id", None),
                    "table_name": item.get("table_name", None),
                    "query": item["question"].strip(),
                    "dsl": item.get("config", {"dimension": [], "measure": [], "filter": []})
                    } for item in raw]

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
    except RateLimitError:
        # 当前 key 降级冷却，再抛出触发 retry 自动换下一个
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        print(f"当前key {key} 请求被限流，已切换到下一个 API Key")
        raise
    except OpenAIError:
        # 其他错误也尝试换 key
        CLIENT_POOL.cooldown(key, seconds=3, escalate=True)
        print(f"当前key {key} 请求出错，已切换到下一个 API Key")
        raise
    print(CLIENT_POOL._meta)
    if LLM_CACHE:
        _ = LLM_CACHE.set_cache_by_params(
            prompt=msgs,
            temperature=temperature,
            cache=pred_str
        )

    return pred_str


def get_spider_schema(db_id, table_name):
    global SPIDER_SCHEMA_CACHE
    if not SPIDER_SCHEMA_CACHE:
        with open("./data/spider_dsl/spider_dev_schema.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            SPIDER_SCHEMA_CACHE[f"{item['db_id']}_{item['table_name']}"] = item['schema']

    return SPIDER_SCHEMA_CACHE.get(f"{db_id}_{table_name}", "")


def get_bird_schema(db_id, table_name):
    global BIRD_SCHEMA_CACHE
    if not BIRD_SCHEMA_CACHE:
        with open("./data/bird_dsl/bird_dev_schema.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            BIRD_SCHEMA_CACHE[f"{item['db_id']}_{item['table_name']}"] = item['schema']

    return BIRD_SCHEMA_CACHE.get(f"{db_id}_{table_name}", "")


@timeit()
# 这个是没有维值召回的预测
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
            prompt_version: str = "spider_1",
            merge_retrieval: bool = False) -> Dict[str, any]:
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
    print("测试问题表名:", query_obj["table_name"], "问题编号:", query_obj["id"])
    for _template in pattern_retrieved_examples:
        print("本次被选中的模板其表名为:", _template[0]["table_name"],
              "问题编号为:", _template[0]["id"], "相似度得分为:", _template[1])
        voter.vote(_template[0]["table_name"], score=_template[1])
    voted_dataset_id_score_list = voter.get_top_k_voted_key(1)
    voted_dataset_id = voted_dataset_id_score_list[0][0] if len(
        voted_dataset_id_score_list) > 0 else None
    print("投票表名为:", voted_dataset_id)
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
    schema = get_bird_schema(query_obj.get(
        "db_id", None), query_obj.get("table_name", None))
    # ============= 构建promp，可以**传入自定义参数 =============
    system_prompt, user_prompt = get_prompt(version_str=prompt_version)(
        query_obj["query"], merged_retrieval, k=pattern_top_k, schema=schema)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    pred_str = query_llm(model=model, msgs=msgs,
                         temperature=temperature, max_tokens=4096)
    pred = safe_parse_json(pred_str)
    from hybrid_rewrite.util import dict_schema_subset

    def judge_json_schema(pred: dict, schema: dict) -> bool:
        # 判断预测结果的结构是否符合schema
        target_dict = {
            "question": "<question_text>",
            "SQL": "<SQL_text>",
            "db_id": "<db_id_text>",
            "table_name": "table_name",
            "config": {
                "dimension": [],
                "measure": [],
                "filter": [],
            }
        }
        return dict_schema_subset(target_dict, pred)

    if not judge_json_schema(pred, schema):
        return {"pred": None, "raw": pred_str,
                "pred_dataset_id": voted_dataset_id,
                "retrieved": {"pattern": pattern_retrieved_info, "keyword": keyword_retrieved_info}}

    return {"pred": pred, "raw": pred_str,
            "pred_dataset_id": voted_dataset_id,
            "retrieved": {"pattern": pattern_retrieved_info, "keyword": keyword_retrieved_info}}


def evaluate(json_path: str,
             test_path: str,
             example_path: str,
             split_ratio: float = 0.9,
             split_keywords: bool = False,
             keywords_for_dimvals: bool = False,
             pattern_key: str = "erased_query",
             pattern_top_k: int = 5,
             keyword_key: str = "erased_core",
             dim_top_k: int = 1,
             keyword_top_k: int = 5,
             model: str = "qwen2.5-72b-instruct",
             emb_model: str = "qwen3-embedding-0.6b",
             emb_model_path: str = None,
             prompt_version: str = "0_0_8",
             temperature: float = 0.2,
             merge_retrieval: bool = False):
    global LLM_CACHE
    LLM_CACHE = LLMCache(model_code=model)

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

    results = []
    all_retrieved = []
    for idx, gold in enumerate(test_set):
        print(f"执行中,{os.path.basename(test_path)}: {idx + 1}/{len(test_set)}")
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
            merge_retrieval=merge_retrieval)
        if res["pred"] is None:
            print("预测结果无法解析为目标json，跳过")
            continue
        all_retrieved.append(res["retrieved"])
        # print(res)
        # 选表正确率
        gt_table_name = gold["table_name"]
        pred_table_name = res["pred_dataset_id"]
        dataset_match = dataset_metric.compare_id(
            pred_table_name, gt_table_name)
        # print(res, "\n", gold["dsl"])
        # 精确匹配：预测 DSL == 标准答案
        match, dim_match, mes_match, fil_match = compare_config(
            res["pred"]["config"], gold["dsl"], strict_mode=False)
        match = match and dataset_match if pattern_top_k else match
        dslmatch_metric.update(match)
        dimension_metric.update(dim_match)
        measure_metric.update(mes_match)
        filter_metric.update(fil_match)
        # --- end ---

        results.append({"query": gold["query"],
                        "gold": gold,
                        "pred": res["pred"],
                        "match": match})

        print(f"Match: {match}, 整体准确率{dslmatch_metric.acc():.3f}, {dslmatch_metric.stat()}"
              f"\n 维度准确率{dimension_metric.acc():.3f}, {dimension_metric.stat()}"
              f"\n 指标准确率{measure_metric.acc():.3f}, {measure_metric.stat()}"
              f"\n 筛选准确率{filter_metric.acc():.3f}, {filter_metric.stat()}"
              f"\n 选表准确率{dataset_metric.acc():.3f}, {dataset_metric.stat()}"
              "\n\n")

    return {"results": results,
            "dsl_acc": dslmatch_metric.acc(),
            "dataset_acc": dataset_metric.acc(),
            "dimension_acc": dimension_metric.acc(),
            "measure_acc": measure_metric.acc(),
            "filter_acc": filter_metric.acc(),
            "retrieved": all_retrieved}


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
                        default="./data/bird_dsl/template/bird_dev_dsl_clean.json")
    parser.add_argument("-m", "--model", type=str,
                        default="qwen2.5-72b-instruct")
    parser.add_argument("--emb_model", type=str,
                        default="qwen3-embedding-0.6b")
    parser.add_argument("--emb_model_path", type=str,
                        default=None)
    parser.add_argument("-p", "--prompt", type=str,
                        default="spider_2")
    parser.add_argument("--pattern_key", type=str,
                        default="query")
    parser.add_argument("--pattern_top_k", type=int,
                        default=2)
    parser.add_argument("--keyword_key", type=str,
                        default=None)
    parser.add_argument("--keyword_top_k", type=int,
                        default=5)
    parser.add_argument("--dim_top_k", type=int,
                        default=1)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    res = evaluate(json_path=args.dataset,
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
             merge_retrieval=False)
    test_basename = os.path.basename(args.test_path)
    time_str = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    if not os.path.exists("./log/bird"):
        os.makedirs("./log/bird")
    with open(f"./log/bird/{test_basename.split('.')[0]}_pattern{args.pattern_top_k}_{time_str}.json", "w", encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
