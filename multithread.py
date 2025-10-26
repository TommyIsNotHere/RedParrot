import os
import threading
import concurrent.futures
import time
import copy
from datetime import datetime
import logging
from hybrid_rewrite.rewrite import evaluate


class RecallPipeline:
    def __init__(self):
        pass

    def run(self, name, args):
        return evaluate(
            json_path=args.get('dataset', None),
            test_path=args.get('test_path', None),
            example_path=args.get('example_path', None),
            name=name,
            prompt_version=args.get('prompt', None),
            split_ratio=args.get('split_ratio', 0.9),
            split_keywords=args.get('split_keywords', False),
            keywords_for_dimvals=args.get('keywords_for_dimvals', True),
            pattern_key=args.get('pattern_key', 'query'),
            pattern_top_k=args.get('pattern_top_k', 10),
            keyword_key=args.get('keyword_key', 'erased_dimension_value'),
            keyword_top_k=args.get('keyword_top_k', 5),
            dim_top_k=args.get('dim_top_k', 1),
            model=args.get('model', None),
            emb_model=args.get('emb_model', None),
            emb_model_path=args.get('emb_model_path', None),
            temperature=args.get('temperature', 0.1),
            merge_retrieval=args.get('merge_retrieval', False)
        )


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)


class MultiThreadRecallPipeline:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()

    def run_single_experiment(self, experiment_config):
        """运行单个实验"""
        experiment_name = experiment_config.get('name', 'unnamed')
        config = experiment_config.get('config', {})

        try:
            logging.info(f"开始实验: {experiment_name}")
            start_time = time.time()

            # 创建pipeline实例
            pipe = RecallPipeline()

            # 运行实验
            result = pipe.run(experiment_name, config)

            end_time = time.time()
            duration = end_time - start_time

            # 线程安全地保存结果
            with self.lock:
                self.results[experiment_name] = {
                    'status': 'success',
                    'result': result,
                    'duration': duration,
                    'config': config
                }

            logging.info(f"实验完成: {experiment_name}, 耗时: {duration:.2f}秒")
            return True

        except Exception as e:
            logging.error(f"实验失败: {experiment_name}, 错误: {str(e)}")

            with self.lock:
                self.results[experiment_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'config': config
                }
            return False

    def run_batch_experiments(self, experiments):
        """批量运行实验"""
        logging.info(
            f"开始批量实验，共{len(experiments)}个实验，最大并发数: {self.max_workers}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_experiment = {
                executor.submit(self.run_single_experiment, exp): exp['name']
                for exp in experiments
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_experiment):
                experiment_name = future_to_experiment[future]
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f"实验 {experiment_name} 产生异常: {exc}")

        logging.info("所有实验完成")
        return self.results


BASECONFIG = {
    "dataset": None,
    "test_path": "./data/20250916/commerce_test_0.95_4.json",
    "example_path": "./data/20250916/commerce_template_0.95_4.json",
    "prompt": "0_0_9",
    "split_ratio": 0.9,
    "split_keywords": False,
    "keywords_for_dimvals": True,
    "pattern_key": "query",  # 核心
    "pattern_top_k": 5,
    "keyword_key": "erased_dimension_value",
    "keyword_top_k": 20,
    "dim_top_k": 2,
    "model": "qwen2.5-72b-instruct",            # 核心，改写主模型
    "emb_model": "qwen3-embedding-4b",
    "emb_model_path": None,
    "temperature": 0.1,
    "merge_retrieval": False
}


# 定义实验配置列表
def create_experiments():
    experiments = []

    # 不同数据集
    datasets = [
        ("trading", "./data/20250916/trading_test_0.95_4.json",
         "./data/20250916/trading_template_0.95_4.json"),
        ("commerce", "./data/20250916/commerce_test_0.95_4.json",
         "./data/20250916/commerce_template_0.95_4.json")
    ]

    # 不同的embedding模型
    emb_models = [
        ("qwen3-0.6b", "qwen3-embedding-0.6b", None),
        ("qwen3-4b", "qwen3-embedding-4b", None),
        ("contrastive-bert", "contrastive-bert", None),
        ("contrastive-bert-trained", "contrastive-bert",
         "./model/contrastive_model/best_model.pth")
    ]

    # 不同的pattern_key (对应不同的召回方法)
    pattern_keys = [
        ("original", "query"),
        ("hanlp-ner", "erased_hanlp"),
        ("manual-erase", "erased_query")
    ]

    # 生成所有组合
    for dataset_name, test_path, example_path in datasets:
        for emb_name, emb_model, emb_model_path in emb_models:
            for pattern_name, pattern_key in pattern_keys:
                # 创建配置副本
                config = copy.deepcopy(BASECONFIG)
                config.update({
                    "test_path": test_path,
                    "example_path": example_path,
                    "emb_model": emb_model,
                    "emb_model_path": emb_model_path,
                    "pattern_key": pattern_key
                })

                experiment_name = f"{dataset_name}_{emb_name}_{pattern_name}"

                experiments.append({
                    "name": experiment_name,
                    "config": config
                })

    return experiments

# 使用示例


def run_all_experiments(experiments, max_workers=4):
    # 创建实验配置
    # experiments = create_experiments()

    # 打印实验计划
    print(f"准备运行 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"  - {exp['name']}")

    # 创建多线程pipeline
    multi_pipeline = MultiThreadRecallPipeline(
        max_workers=max_workers)  # 根据你的资源调整

    # 运行实验
    start_time = time.time()
    results = multi_pipeline.run_batch_experiments(experiments)
    total_time = time.time() - start_time

    # 输出结果摘要
    print(f"\n实验完成! 总耗时: {total_time:.2f}秒")
    print(
        f"成功: {sum(1 for r in results.values() if r['status'] == 'success')}")
    print(f"失败: {sum(1 for r in results.values() if r['status'] == 'failed')}")

    # 保存详细结果
    import json
    nowtime = datetime.now()
    ymd = nowtime.strftime("%Y%m%d")
    hms = nowtime.strftime("%H%M%S")
    log_dir = f"./log/{ymd}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(f"{log_dir}/{hms}_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    only_acc_result = {}
    for k, v in results.items():
        result = v.get("result", {})
        only_acc_result[k] = {
            "status": v.get("status"),
            # "acc": result.get("acc"),
            # "dataset_acc": result.get("dataset_acc"),
            # "dimension_acc": result.get("dimension_acc"),
            # "measure_acc": result.get("measure_acc"),
            # "filter_acc": result.get("filter_acc"),
            "error": v.get("error"),
            "duration": v.get("duration"),
            "config": v.get("config")
        }
        for name, value in result.items():
            if "acc" in name:
                only_acc_result[k][name] = value
    with open(f"{log_dir}/{hms}_only_acc.json", "w", encoding="utf-8") as f:
        json.dump(only_acc_result, f, ensure_ascii=False,
                  indent=2, default=str)

    only_retrieved = {}
    for k, v in results.items():
        only_retrieved[k] = {
            "status": v.get("status"),
            "acc": v.get("result", {}).get("acc"),
            "retrieved": v.get("result", {}).get("retrieved"),
            "duration": v.get("duration"),
            "config": v.get("config")
        }
    with open(f"{log_dir}/{hms}_only_retrieved.json", "w", encoding="utf-8") as f:
        json.dump(only_retrieved, f, ensure_ascii=False,
                  indent=2, default=str)

    return results

# 如果你想要更细粒度的控制，可以手动创建实验列表


def create_custom_experiments(test_path, example_path):
    experiments = []
    basename = os.path.basename(test_path).replace(".json", "").replace("test", "")
    # 示例：只测试trading数据集的不同方法
    custom_config = copy.deepcopy(BASECONFIG)
    custom_config.update({
        "test_path": test_path,
        "example_path": example_path,
        "emb_model": "contrastive-bert"
    })

    # 从头到尾部分别是name, pattern_key, emb_model, emb_model_path
    test_cases = [
        ("original_query", "query", "qwen3-embedding-4b", None),
        ("hanlp_ner", "erased_hanlp", "qwen3-embedding-4b", None),
        ("manual_erase", "erased_query", "qwen3-embedding-4b", None),
        ("entity_agnostic", "query", "contrastive-bert",
         "./model/contrastive_model/best_model.pth")
    ]

    for name, pattern_key, emb_model, emb_model_path in test_cases:
        config = copy.deepcopy(custom_config)
        config["pattern_key"] = pattern_key
        config["emb_model"] = emb_model
        config["emb_model_path"] = emb_model_path

        experiments.append({
            "name": f"{basename}_{name}",
            "config": config
        })

    return experiments
