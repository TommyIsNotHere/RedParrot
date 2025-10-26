import os
import sys
import json
import random
import time
import json
import os
import os.path as osp
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from typing import List, Dict
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import functools
import threading
from collections import defaultdict   # 新增
# 简单版本：记录每次调用耗时
_TIMEIT_DATA = defaultdict(list)
_TIMEIT_LOCK = threading.Lock()


def timeit(name: str = None, logger=None, return_elapsed: bool = False, log: bool = True):
    """
    简单耗时装饰器：把每次执行耗时追加进列表。
    """
    def decorator(func):
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                return (result, elapsed) if return_elapsed else result
            finally:
                elapsed = time.perf_counter() - start
                with _TIMEIT_LOCK:
                    _TIMEIT_DATA[label].append(elapsed)
                if log:
                    msg = f"[TIME] {label} took {elapsed:.4f}s"
                    (logger.info(msg) if logger else print(msg))
        return wrapper
    return decorator


def get_timeit_raw(name: str = None):
    """
    返回原始列表(浅复制)。
    name=None => 返回 {name: [..]} dict；否则返回对应列表或空列表。
    """
    with _TIMEIT_LOCK:
        if name is None:
            return {k: v[:] for k, v in _TIMEIT_DATA.items()}
        return _TIMEIT_DATA.get(name, [])[:]


def summarize_timeit(name: str = None, filter_outliers: bool = True):
    """
    按需计算统计指标（count,total,avg,min,max,p50,p95,p99,last）。
    name=None => 返回所有函数的统计 dict。
    """
    def _calc(arr):
        if not arr:
            return {}
        arr_sorted = sorted(arr)
        n = len(arr)

        def pct(p):
            if n == 1:
                return arr_sorted[0]
            idx = min(n - 1, int(p * (n - 1)))
            return arr_sorted[idx]
        total = sum(arr)
        return {
            'count': n,
            'total': total,
            'avg': total / n,
            'min': arr_sorted[0],
            'max': arr_sorted[-1],
            'p50': pct(0.50),
            'p90': pct(0.90),
            'p95': pct(0.95),
            'p99': pct(0.99),
            'last': arr[-1],
        }

    with _TIMEIT_LOCK:
        import copy
        time_data = copy.deepcopy(_TIMEIT_DATA)
        if filter_outliers:
            # 简单过滤掉 >10s 的异常值
            time_data = {k: [i for i in v if i < 10]
                         for k, v in time_data.items()}
        if name is None:
            return {k: _calc(v) for k, v in time_data.items()}
        return _calc(time_data.get(name, []))


def print_timeit_summary(sort_by='total', top=None, logger=None):
    """
    打印统计汇总（即时计算）。
    sort_by 可选: total/avg/count/max/min/p95/p99/last
    """
    stats = summarize_timeit()
    if not stats:
        msg = "[TIME] No data."
        (logger.info(msg) if logger else print(msg))
        return
    if sort_by not in {'total', 'avg', 'count', 'max', 'min', 'p50', 'p90', 'p95', 'p99', 'last'}:
        sort_by = 'total'
    rows = sorted(stats.items(), key=lambda x: x[1].get(
        sort_by, 0), reverse=True)
    if top:
        rows = rows[:top]
    header = f"{'Name':30} {'Cnt':>5} {'Total':>9} {'Avg':>9} {'Min':>9} {'Max':>9} {'P50':>9} {'P90':>9} {'P95':>9} {'Last':>9}"
    line = "-" * len(header)
    out = [header, line]
    for name, d in rows:
        out.append(
            f"{name:30} {d['count']:5d} {d['total']:9.4f} {d['avg']:9.4f} {d['min']:9.4f} {d['max']:9.4f} {d['p50']:9.4f} {d['p90']:9.4f} {d['p95']:9.4f} {d['last']:9.4f}")
    for line in out:
        (logger.info(line) if logger else print(line))


def reset_timeit(names=None):
    """
    重置记录。names=None 清空全部；str 或 list 指定名称。
    """
    with _TIMEIT_LOCK:
        if names is None:
            _TIMEIT_DATA.clear()
        else:
            if isinstance(names, str):
                names = [names]
            for n in names:
                _TIMEIT_DATA.pop(n, None)


# ===== 新增：持久化时间统计功能 =====
def save_timeit(path: str,
                mode: str = 'summary',
                sort_by: str = 'total',
                top: int = None,
                ensure_dir: bool = True,
                overwrite: bool = True,
                logger=None,
                encoding: str = 'utf-8',
                indent: int = 2) -> str:
    """
    保存 timeit 数据到文件 (支持 .json / .csv)

    参数:
        path: 输出文件路径, 通过后缀决定格式 (.json / .csv)
        mode: 'summary' 保存统计汇总; 'raw' 保存原始每次耗时列表
        sort_by: summary 时排序字段 (total/avg/count/max/min/p95/p99/last)
        top: 只写入前 top 项 (按排序后)
        ensure_dir: 若目录不存在则创建
        overwrite: False 时若文件已存在则抛异常
        logger: 可选 logger
        encoding: 写文件编码
        indent: json 缩进
    返回:
        写入的文件路径
    """
    mode = mode.lower()
    if mode not in {'summary', 'raw'}:
        raise ValueError("mode 必须为 'summary' 或 'raw'")
    ext = os.path.splitext(path)[1].lower()
    if ext not in {'.json', '.csv'}:
        raise ValueError("文件后缀需为 .json 或 .csv")

    if ensure_dir:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"文件已存在: {path}")

    # 拷贝数据
    if mode == 'raw':
        data = get_timeit_raw()
    else:
        data = summarize_timeit()

    # 若需要排序/截取
    if mode == 'summary' and sort_by:
        if sort_by not in {'total', 'avg', 'count', 'max', 'min', 'p50', 'p90', 'p95', 'p99', 'last'}:
            sort_by = 'total'
        items = sorted(data.items(), key=lambda x: x[1].get(
            sort_by, 0), reverse=True)
        if top:
            items = items[:top]
    else:
        items = list(data.items())

    # 写文件
    if ext == '.json':
        out_obj = {
            'mode': mode,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'sort_by': sort_by if mode == 'summary' else None,
            'data': dict(items)
        }
        with open(path, 'w', encoding=encoding) as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=indent)
    else:  # CSV
        import csv
        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.writer(f)
            if mode == 'raw':
                writer.writerow(['name', 'elapsed_list'])
                for name, arr in items:
                    writer.writerow([name, ';'.join(f"{x:.6f}" for x in arr)])
            else:
                writer.writerow(['name', 'count', 'total', 'avg',
                                'min', 'max', 'p50', 'p90', 'p95', 'p99', 'last'])
                for name, stats in items:
                    writer.writerow([
                        name,
                        stats['count'],
                        f"{stats['total']:.6f}",
                        f"{stats['avg']:.6f}",
                        f"{stats['min']:.6f}",
                        f"{stats['max']:.6f}",
                        f"{stats['p50']:.6f}",
                        f"{stats['p90']:.6f}",
                        f"{stats['p95']:.6f}",
                        f"{stats['p99']:.6f}",
                        f"{stats['last']:.6f}",
                    ])
    msg = f"[TIME] Saved {mode} timeit data to {path}"
    (logger.info(msg) if logger else print(msg))
    return path


def ensure_path():
    # 获取当前notebook文件的目录
    current_dir = os.path.dirname(os.path.abspath('.'))
    sys.path.append(current_dir)
    os.chdir(current_dir)


def deduplicate_dicts(dict_list, key):
    """
    根据指定键对字典列表进行去重，保留首次出现的字典
    Args:
        dict_list: 字典列表
        key: 用于去重的键名

    Returns:
        去重后的字典列表，保持原始顺序
    """
    seen = set()
    return [
        d for d in dict_list
        if not (d.get(key) in seen or seen.add(d.get(key)))
    ]


"""
进度条工具函数
提供统一的进度显示功能，方便在多个模块中复用
"""


def progress_bar(current, total, bar_length=30, prefix="处理进度"):
    """
    显示进度条

    Args:
        current: 当前进度
        total: 总进度
        bar_length: 进度条长度（字符数）
        prefix: 进度条前缀文字
    """
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    print(
        f"\r{prefix}: [{bar}] {progress*100:.1f}% ({current}/{total})", end="")

    # 如果处理完成，换行
    if current == total:
        print()


def batch_progress_bar(batch_index, total_batches, current_items, total_items, bar_length=30):
    """
    批次处理进度条
    Args:
        batch_index: 当前批次索引（从1开始）
        total_batches: 总批次数
        current_items: 当前已处理数据量
        total_items: 总数据量
        bar_length: 进度条长度
    """
    progress = batch_index / total_batches
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    print(f"\r正在处理: [{bar}] {progress*100:.1f}% ({batch_index}/{total_batches}批次, {current_items}/{total_items}条数据)", end="")

    # 如果处理完成，换行
    if batch_index == total_batches:
        print()


def split_dataset_and_create_loaders(dataset, train_ratio=0.85, val_ratio=0.05, test_ratio=0.10,
                                     batch_size=64, random_state=42):
    """
    划分数据集为训练集、验证集、测试集，并创建对应的数据加载器

    参数:
        dataset: 完整的数据集对象
        train_ratio: 训练集比例 (默认: 0.85)
        val_ratio: 验证集比例 (默认: 0.05)
        test_ratio: 测试集比例 (默认: 0.10)
        batch_size: 批次大小 (默认: 64)
        random_state: 随机种子 (默认: 42)

    返回:
        train_loader, val_loader, test_loader: 训练、验证、测试数据加载器
    """
    train_dataset, val_dataset, test_dataset = split_dataset_only(
        dataset, train_ratio, val_ratio, test_ratio, random_state)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # 打印分割信息
    print(f"数据集分割完成:")
    print(
        f"  训练集: {len(train_dataset)} 样本 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(
        f"  验证集: {len(val_dataset)} 样本 ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(
        f"  测试集: {len(test_dataset)} 样本 ({len(test_dataset)/len(dataset)*100:.1f}%)")

    return train_loader, val_loader, test_loader


def split_dataset_only(dataset, train_ratio=0.85, val_ratio=0.05, test_ratio=0.10, random_state=42):
    """
    仅划分数据集，不创建数据加载器

    返回:
        train_dataset, val_dataset, test_dataset: 划分后的数据集
    """
    # 验证比例总和为1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和应为1.0，当前为{total_ratio:.2f}")

    # 计算测试集大小
    test_size = int(len(dataset) * test_ratio)
    # 剩余部分用于训练和验证
    train_val_size = len(dataset) - test_size

    # 首先分割测试集
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_state
    )

    # 在训练验证集中计算验证集大小
    val_size = int(train_val_size * (val_ratio / (train_ratio + val_ratio)))

    # 分割训练集和验证集
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=random_state
    )

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
