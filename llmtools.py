from openai import OpenAI
# python
# 1. 在 hybrid_rewrite.py 顶部去掉硬编码 CLIENT，改成：
import os
import time
import threading
import random
import itertools
from openai import OpenAI
from collections import Counter

API_KEYS = os.getenv("DIRECT_LLM_API_KEYS", "").split(",")
BASE_URL = os.getenv("DIRECT_LLM_BASE_URL")


class KeyRotator:
    def __init__(self, api_keys, mode="round_robin"):
        self.mode = mode
        self._lock = threading.Lock()
        self._weights = Counter(api_keys)
        unique_keys = list(self._weights.keys())
        self._meta = {
            k: {
                "cool_until": 0.0,
                "use_count": 0,
                "last_used": 0.0,
                "fail_streak": 0,
                "weight": self._weights[k],
                "avg_latency": None,   # 新增: 平均时延
            } for k in unique_keys
        }
        self._last_key = None
        self._clients = {k: OpenAI(api_key=k, base_url=BASE_URL, timeout=40)
                         for k in unique_keys}

    def _eligible_keys(self, now):
        return [k for k, m in self._meta.items() if m["cool_until"] <= now]

    def _score(self, k):
        m = self._meta[k]
        # 引入 avg_latency 让慢 key 在平衡时被降权 (None 视作 0)
        latency = m["avg_latency"] if m["avg_latency"] is not None else 0
        return (m["use_count"] / m["weight"], latency, m["last_used"])

    def _pick_key(self):
        now = time.time()
        eligible = self._eligible_keys(now)
        if not eligible:
            k = min(self._meta.items(), key=lambda kv: kv[1]["cool_until"])[0]
            return k
        # 不再无条件复用上一次，给高速 key 更多机会但仍轮换
        if self.mode == "random":
            # 加权随机 + 轻度惩罚慢 key
            weights = []
            for k in eligible:
                w = self._meta[k]["weight"]
                lat = self._meta[k]["avg_latency"]
                if lat and lat > 30:  # >30s 视作慢，降低权重
                    w *= 0.5
                weights.append(w)
            total = sum(weights)
            r = random.uniform(0, total)
            acc = 0
            for k, w in zip(eligible, weights):
                acc += w
                if acc >= r:
                    return k
            return eligible[-1]
        eligible.sort(key=lambda k: self._score(k))
        return eligible[0]

    def get_client(self):
        with self._lock:
            k = self._pick_key()
            self._meta[k]["use_count"] += 1
            self._meta[k]["last_used"] = time.time()
            self._last_key = k
            return k, self._clients[k]

    def cooldown(self, key, seconds=3, escalate=False):
        with self._lock:
            m = self._meta[key]
            if escalate:
                m["fail_streak"] += 1
            else:
                m["fail_streak"] = 0
            extra = (2 ** (m["fail_streak"] - 1)
                     ) if m["fail_streak"] > 0 else 0
            adj = seconds + extra
            adj /= m["weight"]
            m["cool_until"] = time.time() + adj

    def mark_success(self, key, latency=None):
        with self._lock:
            m = self._meta[key]
            m["fail_streak"] = 0
            if latency is not None:
                if m["avg_latency"] is None:
                    m["avg_latency"] = latency
                else:
                    # 指数滑动平均
                    m["avg_latency"] = m["avg_latency"] * 0.7 + latency * 0.3


CLIENT_POOL = KeyRotator(API_KEYS, mode="round_robin")  # 或 mode="random"
