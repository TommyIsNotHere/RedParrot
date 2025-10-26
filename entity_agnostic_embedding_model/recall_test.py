import argparse
from multithread import run_all_experiments, BASECONFIG
import os
import copy
from hybrid_rewrite.rewrite import TIME_STATISTICS


def create_custom_experiments(test_path, example_path, emb_model=None, emb_model_path=None):
    experiments = []
    basename = os.path.basename(test_path).replace(
        ".json", "").replace("test", "")
    # 示例：只测试trading数据集的不同方法
    custom_config = copy.deepcopy(BASECONFIG)
    custom_config.update({
        "test_path": test_path,
        "example_path": example_path,
    })

    # 从头到尾部分别是name, pattern_key, emb_model, emb_model_path, keywords_for_dimvals,keyword_key
    test_cases = [
        ("original_query", "query", "qwen3-embedding-0.6b", None, False, None),
        ("hanlp_ner", "erased_hanlp", "qwen3-embedding-0.6b", None, True, "erased_ner"),
        ("manual_erase", "erased_query", "qwen3-embedding-0.6b",
         None, True, "erased_dimension_value"),
        ("entity_agnostic", "query", emb_model,
         emb_model_path, True, "erased_dimension_value")
    ]

    for _name, _pattern_key, _emb_model, _emb_model_path, _keywords_for_dimvals, _keyword_key in test_cases:
        config = copy.deepcopy(custom_config)
        config["pattern_key"] = _pattern_key
        config["emb_model"] = _emb_model
        config["emb_model_path"] = _emb_model_path
        config["keywords_for_dimvals"] = _keywords_for_dimvals
        config["keyword_key"] = _keyword_key

        experiments.append({
            "name": f"{basename}_{_name}",
            "config": config
        })

    return experiments


# ------------- 配置 -------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_path", type=str,
                        default=None)
    parser.add_argument("-e", "--example_path", type=str,
                        default=None)
    parser.add_argument("--emb_model", type=str,
                        default="qwen3-embedding-0.6b")
    parser.add_argument("--emb_model_path", type=str,
                        default=None)
    parser.add_argument("--timeit", action="store_true", default=False)
    parser.add_argument("--run0916", action="store_true", default=False)
    parser.add_argument("--run0922", action="store_true", default=False)
    parser.add_argument("--run0928_95", action="store_true", default=False)
    parser.add_argument("--run0928_90", action="store_true", default=False)
    parser.add_argument("--run0928_85", action="store_true", default=False)
    parser.add_argument("--runrandom", action="store_true", default=False)
    parser.add_argument("--run0916leaveone",
                        action="store_true", default=False)
    parser.add_argument("--onlyone", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    experiments = []
    if args.run0916:
        print("Running 0916 experiments...")
        test_templates = [
            ("./data/20250916/trading_test_0.95_4.json",
             "./data/20250916/trading_template_0.95_4.json"),
            ("./data/20250916/community_test_0.95_4.json",
             "./data/20250916/community_template_0.95_4.json"),
            ("./data/20250916/commerce_test_0.95_4.json",
             "./data/20250916/commerce_template_0.95_4.json"),
        ]
    elif args.run0916leaveone:
        print("Running 0916 leave-one-out experiments...")
        test_templates = [
            ("./data/20250916/trading_test_0.95_4_leaveone.json",
             "./data/20250916/trading_template_0.95_4.json"),
            ("./data/20250916/community_test_0.95_4_leaveone.json",
             "./data/20250916/community_template_0.95_4.json"),
            ("./data/20250916/commerce_test_0.95_4_leaveone.json",
             "./data/20250916/commerce_template_0.95_4.json"),
        ]
    elif args.run0922:
        print("Running 0922 experiments...")
        test_templates = [
            ("./data/20250922/trading_test_0.90_4_3.json",
             "./data/20250922/trading_template_0.90_4_3.json"),
            ("./data/20250922/community_test_0.90_4_3.json",
             "./data/20250922/community_template_0.90_4_3.json"),
            ("./data/20250922/commerce_test_0.90_4_3.json",
             "./data/20250922/commerce_template_0.90_4_3.json"),
        ]
    elif args.run0928_95:
        print("Running 0928_95 experiments...")
        test_templates = [
            ("./data/20250928-0.95/trading_test_0.95_4_2.json",
             "./data/20250928-0.95/trading_template_0.95_4_2.json"),
            ("./data/20250928-0.95/community_test_0.95_4_2.json",
             "./data/20250928-0.95/community_template_0.95_4_2.json"),
            ("./data/20250928-0.95/commerce_test_0.95_4_2.json",
             "./data/20250928-0.95/commerce_template_0.95_4_2.json"),
        ]
    elif args.run0928_90:
        print("Running 0928_90 experiments...")
        test_templates = [
            ("./data/20250928-0.90/trading_test_0.90_4_3.json",
             "./data/20250928-0.90/trading_template_0.90_4_3.json"),
            ("./data/20250928-0.90/community_test_0.90_4_3.json",
             "./data/20250928-0.90/community_template_0.90_4_3.json"),
            ("./data/20250928-0.90/commerce_test_0.90_4_3.json",
             "./data/20250928-0.90/commerce_template_0.90_4_3.json"),
        ]
    elif args.run0928_85:
        print("Running 0928_85 experiments...")
        test_templates = [
            ("./data/20250928-0.85/trading_test_0.85_4_3.json",
             "./data/20250928-0.85/trading_template_0.85_4_3.json"),
            ("./data/20250928-0.85/community_test_0.85_4_3.json",
             "./data/20250928-0.85/community_template_0.85_4_3.json"),
            ("./data/20250928-0.85/commerce_test_0.85_4_3.json",
             "./data/20250928-0.85/commerce_template_0.85_4_3.json"),
        ]
    elif args.runrandom:
        print("Running random experiments...")
        test_templates = [
            ("./data/20250916/trading_test_0.95_4.json",
             "./data/randomtemplate/trading_24.json"),
            ("./data/20250916/community_test_0.95_4.json",
             "./data/randomtemplate/community_61.json"),
            ("./data/20250916/commerce_test_0.95_4.json",
             "./data/randomtemplate/commerce_373.json"),
        ]
    else:
        print("Running custom experiments...")
        test_templates = [(args.test_path, args.example_path)]

    TIME_STATISTICS = args.timeit
    for test_path, example_path in test_templates:
        experiments_ = create_custom_experiments(test_path=test_path,
                                                example_path=example_path,
                                                emb_model=args.emb_model,
                                                emb_model_path=args.emb_model_path)
        experiments.extend(experiments_)

    if args.onlyone:
        run_all_experiments([experiments[args.onlyone-1]])
    else:
        run_all_experiments(experiments)
