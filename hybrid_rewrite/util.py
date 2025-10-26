import json
import re

from typing import List, Any, Dict


class BinaryMetric():
    def __init__(self):
        self.t = 0
        self.f = 0

    def compare_id(self, x: Any, y: Any):
        ret = str(x) == str(y)
        self.update(ret)
        return ret

    def update(self, x: Any):
        if isinstance(x, bool):
            if x:
                self.t += 1
            elif not x:
                self.f += 1
        elif isinstance(x, int):
            if x == 1:
                self.t += 1
            elif x == 0:
                self.f += 1
        else:
            print(f"[WARNING]Unsupported input for BinaryMetric!")

    def acc(self):
        return self.t / float(self.t + self.f)

    def correct(self):
        return self.t

    def all(self):
        return self.t + self.f

    def stat(self):
        return f"{self.t}/{self.all()}"


class Voter():
    """
    用于对<key, score>形式的数据进行投票
    """

    def __init__(self):
        self._voting_score = {}

    def vote(self, key: Any, score: float):
        if key not in self._voting_score.keys():
            self._voting_score[key] = 0
        self._voting_score[key] += score

    def get_top_k_voted_key(self, k: int) -> List[Any]:
        all_key_scores = list(self._voting_score.items())
        all_key_scores.sort(key=lambda x: x[1], reverse=True)
        return all_key_scores[:k]


def dict_schema_subset(sub: dict, sup: dict) -> bool:
    """
    判断 sub 的结构是否为 sup 的子集（忽略 sup 里多余的 key）
    规则：
      1. sub 的每个 key 必须出现在 sup 中
      2. 对应 value 如果是 dict 递归判断
      3. 其它类型仅要求 type 相同
      4. list 当前只比较类型本身，不下钻元素（需要更细可再扩展）
    """
    if not isinstance(sub, dict) or not isinstance(sup, dict):
        return False
    if not set(sub.keys()).issubset(sup.keys()):
        return False
    for k, v_sub in sub.items():
        v_sup = sup[k]
        if isinstance(v_sub, dict) and isinstance(v_sup, dict):
            if not dict_schema_subset(v_sub, v_sup):
                return False
        else:
            if type(v_sub) is not type(v_sup):
                return False
    return True


def same_dict_schema(d1: dict, d2: dict) -> bool:
    if d1.keys() != d2.keys():
        return False
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not same_dict_schema(v1, v2):
                return False
        else:
            if type(v1) is not type(v2):
                return False
    return True


def judge_json(data: dict):
    if not data:
        return False
    if not isinstance(data, dict):
        return False
    sub_dict = {
        "dimension": [],
        "measure": [],
        "filter": [],
    }
    return dict_schema_subset(sub_dict, data)


def safe_parse_json(raw: str) -> dict:
    """
    1. 去掉 ```json 与 ``` 包裹
    2. 尝试 json.loads
    返回解析后的 dict；失败返回 空列表
    """
    # 1. 去掉 <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    # 2. 去掉 ```json / ``` 包裹
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.I)
    try:
        return json.loads(cleaned)
    except Exception as e:
        # print(cleaned)
        # print(e)
        return {}


def compare_config(pred: dict, gold: dict, strict_mode: bool = True) -> bool:
    """
    判断pred中的元素是否和gold中的元素一致。
    strict_mode开启时要求元素全等（无视顺序），包含关系视为错
    """
    if isinstance(pred, str):
        pred = json.loads(pred)
    if isinstance(gold, str):
        gold = json.loads(gold)

    if (not isinstance(pred, dict)) or (not isinstance(gold, dict)):
        print(f"pred/gold only accepts dict obj!")
        return False, False, False, False

    if strict_mode:
        # whole_match = pred == gold
        dim_match = pred.get("dimension") == gold.get("dimension")
        mes_match = pred.get("measure") == gold.get("measure")
        filter_match = pred.get("filter") == gold.get("filter")
        whole_match = dim_match and mes_match and filter_match
        return whole_match, dim_match, mes_match, filter_match
    else:
        # whole_match = compare(pred, gold)
        dim_match = compare(pred.get("dimension"), gold.get("dimension"))
        mes_match = compare(pred.get("measure"), gold.get("measure"))
        filter_match = compare(pred.get("filter"), gold.get("filter"))
        whole_match = dim_match and mes_match and filter_match
        return whole_match, dim_match, mes_match, filter_match


def compare_item(item1: Any, item2: Any) -> bool:
    item1 = str(item1).lower()
    item2 = str(item2).lower()
    item1 = item1.replace(" ", "").replace(
        "\n", "").replace('"', "").replace("'", "")
    item2 = item2.replace(" ", "").replace(
        "\n", "").replace('"', "").replace("'", "")
    return item1 == item2


def compare(o1: Any, o2: Any):
    if (o1 is None and o2 is not None) or (o2 is None and o1 is not None):
        return False
    # if type(o1) is not type(o2):
    #     return False
    if isinstance(o1, dict) and isinstance(o2, dict):
        return compare_dict(o1, o2)
    elif isinstance(o1, list) and isinstance(o2, list):
        return compare_list(o1, o2)
    else:
        return compare_item(o1, o2)


def compare_list(l1: List, l2: List):
    if (l1 is None and l2 is not None) or (l2 is None and l1 is not None):
        return False
    if len(l1) != len(l2):
        return False

    # 安全构造排序 key，避免不同类型直接比较
    def _sort_key(x):
        result = ""
        if isinstance(x, dict):
            try:
                # 使用稳定且可比较的 JSON 序列化
                result = "dict:" + \
                    json.dumps(x, sort_keys=True, ensure_ascii=False)
            except Exception:
                result = "dict:" + str(sorted(x.items()))
        elif isinstance(x, list):
            # 递归降维字符串
            result = "list:" + str([_sort_key(e) for e in x])
        else:
            result = f"{type(x).__name__}:{repr(x)}"
        return result.lower()

    try:
        l1_sorted = sorted(l1, key=_sort_key)
        l2_sorted = sorted(l2, key=_sort_key)
    except Exception:
        # 回退：不排序直接按原顺序比较
        l1_sorted = l1
        l2_sorted = l2
    for v1, v2 in zip(l1_sorted, l2_sorted):
        # if type(v1) is not type(v2):
        #     return False
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not compare_dict(v1, v2):
                return False
        elif isinstance(v1, list) and isinstance(v2, list):
            if not compare_list(v1, v2):
                return False
        else:
            if not compare_item(v1, v2):
                return False
    return True


def compare_dict(d1: Dict, d2: Dict):
    if (d1 is None and d2 is not None) or (d2 is None and d1 is not None):
        return False
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return False
    # 先比较 key 集合是否一致，避免遗漏
    if set(d1.keys()) != set(d2.keys()):
        return False

    for k, v1 in d1.items():
        v2 = d2.get(k)
        # if type(v1) is not type(v2):
        #     return False
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not compare_dict(v1, v2):
                return False
        elif isinstance(v1, list) and isinstance(v2, list):
            if not compare_list(v1, v2):
                return False
        else:
            if not compare_item(v1, v2):
                return False
    return True
