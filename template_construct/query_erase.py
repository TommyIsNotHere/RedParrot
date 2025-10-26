import re
import os
import numpy as np
import pandas as pd
import json
from utils import timeit

# placeholder可以自主设置,默认是?,注意设置成其他的时候需要考虑到remove_enclosed_text的针对处理
PLACEHOLDER = '?'


@timeit(log=False)
def remove_enclosed_text(text):
    # 匹配被双引号或尖括号包围的内容（非贪婪模式）
    exclude_symbol = ['""', '[]', '{}', '\'\'',
                      '《》', '“”', '’‘', '（）', '【】', '‘’', '<>']
    enclosed_pattern = '|'.join([f'{i[0]}[^"]*{i[1]}' for i in exclude_symbol])
    enclosed_pattern = rf'({enclosed_pattern}|\([^"]*\))'
    text = re.sub(enclosed_pattern, PLACEHOLDER, text)
    symbol_pattern = r'[\s\u3000\u3002\uff1b\uff0c\uff1a\u201c\u201d\u2026\uff08\uff09\uff1f\u300a\u3001\u300b!"#$%&\'*+,-./:;=|@\\^_`~]'
    return re.sub(symbol_pattern, '', text)

# 重组逻辑


@timeit(log=False)
def reorg_sentence_by_hanlp(texts, batch_size=512):
    """使用hanlp进行实体识别
    输入:
    texts: 输入的文本列表,每个元素是一个字符串
    输出:
    new_sentences: 输出的文本列表,每个元素是一个擦除后字符串
    remove_tokens: 输出的文本列表,每个元素是一个列表,列表中的元素是字符串,表示被移除的实体
    """
    import hanlp
    HanLP = hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    output = {'tok/fine': [], 'ner/msra': [],
              'ner/pku': [], 'ner/ontonotes': []}
    # 添加进度条显示
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_end = min(i+batch_size, len(texts))
        current_batch = (i // batch_size) + 1
        # 进度条显示
        progress = current_batch / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(
            f"\r正在处理: [{bar}] {progress*100:.1f}% ({current_batch}/{total_batches}批次, {batch_end}/{len(texts)}条数据)", end="")
        batch_texts = texts[i:batch_end]
        batch_output = HanLP(batch_texts, tasks='ner*')
        for key in output.keys():
            output[key].extend(batch_output[key])

    new_sentences = []
    remove_tokens = []
    ignore_entities = {
        "msra": ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'DURATION', 'TIME',
                 'PERCENT', 'MONEY', 'FREQUENCY', "INTEGER", "FRACTION", "DECIMAL", "ORDINAL", "RATE",
                 "EMAIL", "PHONE", "WWW", "FAX", "TELEX", "POSTALCODE"],

        "ontonotes": ['PERSON', 'NORP', 'FACILITY', 'ORGANIZATION', 'GPE', 'LOCATION', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                      "LAW", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"],

        "pku": ['nr', 'ns', 'nt', 'nw', 'nz'],
    }
    ignore_entities = ignore_entities["msra"] + \
        ignore_entities["pku"] + ignore_entities["ontonotes"]

    tokens_record = output['tok/fine']
    entities_record = [a + b + c for a, b,
                       c in zip(output['ner/msra'], output['ner/pku'], output['ner/ontonotes'])]

    for tokens, entities in zip(tokens_record, entities_record):
        # 获取需要排除的字符位置
        exclude_positions = set()
        _remove_tokens = []
        # print("原始句子:", "".join(tokens))
        # print("发现排除实体:")
        for entity in entities:
            if entity[1] in ignore_entities:
                # print("  ",entity)
                _remove_tokens.append(entity[0])
                exclude_positions.update(range(entity[2], entity[3]))
        # print(exclude_positions)
        # 过滤并重组
        filtered = [
            PLACEHOLDER if idx in exclude_positions else token
            for idx, token in enumerate(tokens)
        ]
        new_sentences.append(''.join(filtered))
        remove_tokens.append(_remove_tokens)
        # print("过滤句子:", ''.join(filtered))
        # print()
    return new_sentences, remove_tokens


@timeit(log=False)
def reorg_sentence_by_spacy(text):
    # 加载spaCy中文模型
    import spacy

    nlp = spacy.load("zh_core_web_sm")
    # 定义要移除的实体类型（例如：时间、组织、产品、地名等）
    entities_to_remove = ["DATE", "ORG", "PRODUCT", "GPE"]  # 可以根据需要添加其他类型
    doc = nlp(text)

    # 获取所有实体，并按起始字符降序排序（以便从后往前移除，避免索引变化）
    entities = list(doc.ents)
    entities_sorted = sorted(
        entities, key=lambda x: x.start_char, reverse=True)

    # 初始化新文本
    new_text = text
    for ent in entities_sorted:
        if (ent.label_ in entities_to_remove) or True:
            start = ent.start_char
            end = ent.end_char
            # 移除实体对应的字符串
            new_text = new_text[:start] + new_text[end:]

    # 清理文本：移除可能留下的多余空格或标点（例如，逗号紧跟在实体后）
    # 这里简单处理：移除首尾空格，并替换多个空格为单个空格
    new_text = new_text.strip()
    new_text = ' '.join(new_text.split())
    return new_text

    # 示例用法
    # original_text = "2024年6月，SPU名称包含华为P70手机的UGC笔记、CPS笔记、商品笔记、蒲公英笔记、主理人笔记、软广笔记、KOS授权笔记和企业号笔记的总曝光量和总点击量数据，分商业笔记类型查看"
    # result = remove_entities_and_time(original_text)
    # print(result)


@timeit(log=False)
def query_erase_by_rewritten(data):
    """
    利用大模型改写信息,进行关键词擦除,
    :param data: 输入数据,包含query和rewritten字段,是一个字典列表
    :return: 输出数据,包含擦除后的query字段和一些中间信息,以及原始数据其他的字段
    """
    df = pd.DataFrame(data)
    query_list = df['query'].tolist()
    rewritten_list = df['rewritten'].tolist()
    final_data = [{'query': query, 'rewritten': rewritten}
                  for query, rewritten in zip(query_list, rewritten_list)]
    for idx, rewritten in enumerate(rewritten_list):
        # 如果要素不存在，那么就全部置空即可，erased等于原始query
        if "要素" not in rewritten.keys():
            final_data[idx]['erased_dimension_value'] = {}
            final_data[idx]['erased'] = query_list[idx]
            final_data[idx]['erased_time'] = []
            final_data[idx]['erased_core'] = []
            continue

        yaosu = rewritten['要素']
        time_range_key = list(
            set(['time_range', '时间范围', 'time范围']).intersection(yaosu.keys()))[0]
        # if 'time_range' in yaosu.keys():
        #     time_str = [item for item in yaosu['time_range'] if item] if isinstance(yaosu['time_range'],list) else [yaosu['time_range']]
        # elif '时间范围' in yaosu.keys():
        #     time_str = [item for item in yaosu['时间范围'] if item] if isinstance(yaosu['时间范围'],list) else [yaosu['时间范围']]
        # elif 'time范围' in yaosu.keys():
        #     time_str = [item for item in yaosu['time范围'] if item] if isinstance(yaosu['time范围'],list) else [yaosu['time范围']]
        # else:
        #     assert False, query_list[idx]
        time_range = yaosu[time_range_key]
        if isinstance(time_range, dict):
            time_range = sum(time_range.values(), [])
        time_range_str = [item for item in time_range] if isinstance(
            time_range, list) else [time_range]

        replace_str = []
        filter_ = yaosu['筛选条件'] if '筛选条件' in yaosu.keys() else yaosu['筛选']
        filter_condition = [item.strip() for item in filter_ if item]

        def count_comparison_operators(s):
            import re
            # 匹配所有比较操作符（包含复合操作符）
            pattern = r'>=|<=|==|≥|≠|≤|<|>|=|='
            operators = re.findall(pattern, s)
            # 过滤掉单独的等号（当不作为比较操作符时）
            # 通过位置判断：等号前后必须有数字或字母
            filtered = []
            for i, op in enumerate(operators):
                if op == '=':
                    # 检查前后字符是否为可比较内容
                    prev_char = s[i-1] if i > 0 else None
                    next_char = s[i+1] if i < len(s)-1 else None
                    if (str(prev_char).isalnum() or prev_char in ' )') and \
                            (str(next_char).isalnum() or next_char in ' ('):
                        filtered.append(op)
                else:
                    filtered.append(op)

            return len(filtered)

        def split_by_comparison_operators(s):
            import re
            # 按操作符长度降序匹配（避免错误拆分）
            pattern = r'>=|<=|==|≥|≠|≤|<|>|=|='
            # 直接分割不保留分隔符
            parts = re.split(pattern, s)
            # 过滤空字符串并去除首尾空白
            result = [part.strip() for part in parts if part.strip()]
            # 当分割后产生多个片段时返回，否则返回原字符串
            return result if len(result) > 1 else [s]

        dimension_value = {}
        for item in filter_condition:
            op_num = count_comparison_operators(item)
            split_item = split_by_comparison_operators(item)
            if op_num == 1:
                # print(item)
                # assert len(
                #     split_item) == 2, f"筛选条件中包含未知操作符：{item},query:{query_list[idx]}"
                if len(split_item) == 2:
                    dimension_value[split_item[0]] = [split_item[1]]
                    replace_str.append(split_item[1])
                else:
                    print("特殊字串", item)
            elif op_num == 2:
                # assert len(
                #     split_item) == 3, f"筛选条件中包含未知操作符：{item},query:{query_list[idx]}"
                if len(split_item) == 3:
                    dimension_value[split_item[1]] = [
                        split_item[0], split_item[2]]
                    replace_str.append(split_item[2])
                    replace_str.append(split_item[0])
                else:
                    print("特殊字串", item)
            elif op_num == 0:
                if any([i in item for i in ['包含', '精确匹配', "模糊匹配", "LIKE", 'in']]):
                    if len(item.split()) == 3:
                        item_split = item.split()
                        dimension_value[item_split[0]
                                        ] = item_split[2].split(',')
                        replace_str.extend(item_split[2].split(','))
                    else:
                        split_str = [i for i in ['包含', '精确匹配',
                                                 "模糊匹配", "LIKE", 'in'] if i in item][0]
                        item_split = item.split(split_str)
                        item_split = [i.strip()
                                      for i in item_split if i.strip()]
                        if len(item_split) != 2:
                            print(item)
                        else:
                            if '[' in item_split[1] and ']' in item_split[1]:
                                item_split[1] = item_split[1].replace('[', '')
                                item_split[1] = item_split[1].replace(']', '')
                            dimension_value[item_split[0].replace(
                                ' ', '')] = item_split[1].split(',')
                            replace_str.extend(item_split[1].split(','))
                        # print(f"筛选条件中包含未知操作符：{item}#     query:{query_list[idx]}")
                else:
                    item_split = item.split()
                    item_split = [i.strip() for i in item_split if i.strip()]
                    if len(item_split) == 2:
                        dimension_value[item_split[0]
                                        ] = item_split[1].split(',')
                        replace_str.extend(item_split[1].split(','))
                    elif len(item_split) == 3:
                        dimension_value[item_split[0].replace(
                            ' ', '')] = item_split[2].split(',')
                        replace_str.extend(item_split[2].split(','))
                    elif len(item_split) == 5:
                        dimension_value[item_split[0].replace(' ', '')] = item_split[2].split(
                            ',')+item_split[4].split(',')
                        replace_str.extend(item_split[2].split(
                            ',')+item_split[4].split(','))
                    else:
                        print("特殊字串", item)
            else:
                print(f"筛选条件中包含未知操作符：{item}, query:{query_list[idx]}")

        replace_str = [item for item in replace_str if item]
        time_range_str = [item for item in time_range_str if item]
        query = query_list[idx].strip()

        for ss in time_range_str:
            query = query.replace(ss, PLACEHOLDER, 1)
        for ss in replace_str:
            query = query.replace(ss, PLACEHOLDER, 1)

        final_data[idx]['erased'] = query
        final_data[idx]['erased_time'] = time_range_str
        final_data[idx]['erased_dimension_value'] = dimension_value
        final_data[idx]['erased_core'] = [
            item for item in time_range_str+replace_str if item in query_list[idx]]
    return final_data


@timeit(log=False)
def query_erase(data):
    """
    问题擦除提取主干问题的主函数
    :param data: 输入数据，包含query字段和rewritten的信息,是一个字典列表
    :return: 输出数据，包含擦除后的query字段和一些中间信息,以及原始数据其他的字段

    这里面几个函数代表着三个核心过程：
    1. query_erase_by_rewritten：根据rewritten信息擦除query中的要素
    2. reorg_sentence_by_hanlp：使用hanlp进行句子组织
    3. remove_enclosed_text：正则清洗

    如果在线上,用户问了一个新的数据,可以调用这个函数,来进行擦除,data传入一个单项列表就行,这个函数既可以用来构建模板库之前的准备工作,也可以拿来在线上调用(修改适配)
    """
    final_data = query_erase_by_rewritten(data)
    erased_query = [item['erased'] for item in final_data]
    erased_query, remove_tokens = reorg_sentence_by_hanlp(erased_query)
    erased_query = [remove_enclosed_text(text) for text in erased_query]

    erased_hanlp, _ = reorg_sentence_by_hanlp(
        [item['query'] for item in data])

    for index, query in enumerate(erased_query):
        data[index]['erased'] = query
        data[index]['erased_time'] = final_data[index]['erased_time']
        data[index]['erased_dimension_value'] = final_data[index]['erased_dimension_value']
        data[index]['erased_core'] = final_data[index]['erased_core']
        data[index]['erased_ner'] = list(set(remove_tokens[index]))
        data[index]['erased_hanlp'] = erased_hanlp[index]

    return data
