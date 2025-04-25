import json
import torch
import math
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import Levenshtein
from tqdm import tqdm
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="universal_encoder")
    parser.add_argument("--target_path", type=str, default="vicuna_novel_.json")

    return parser.parse_args()


def final_result_file(target_file):
    filename, _ = os.path.splitext(os.path.basename(target_file))
    final_result_file_name = "{}.json".format(filename)
    return final_result_file_name


def filter_json_by_max_prob(input_file):
    """
    处理JSON文件，对于相同的origin只保留prob值最大的项

    :param input_file: 输入JSON文件路径
    :return: 处理后的数据
    """
    try:
        # 规范化文件路径
        input_file = os.path.normpath(input_file)

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在!")
            return None

        print(f"正在读取文件: {input_file}")

        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                print("请确保输入文件是有效的JSON格式")
                return None

        print(f"文件读取成功，共包含 {len(data)} 项数据")

        # 创建一个字典，用于存储每个origin对应的最大prob项
        origin_max_prob = {}

        # 遍历所有项，找到每个origin的最大prob项
        for item in data:
            origin = item['origin']
            prob = item['prob']

            # 如果这个origin还没有记录，或者当前prob大于已记录的prob
            if origin not in origin_max_prob or prob > origin_max_prob[origin]['prob']:
                origin_max_prob[origin] = item

        # 将结果转换回列表
        result = list(origin_max_prob.values())

        print(f"去重处理完成, 从 {len(data)} 项中筛选出 {len(result)} 项")

        return result

    except json.JSONDecodeError as e:
        print(f"错误: '{input_file}' 不是有效的JSON文件!")
        print(f"详细错误: {str(e)}")
        return None
    except Exception as e:
        print(f"处理时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_result(encoder_path, target_path):
    # 先执行去重处理
    print("第一步：执行去重处理...")
    filtered_data = filter_json_by_max_prob(target_path)

    if filtered_data is None:
        print("去重处理失败，终止后续操作")
        return

    # 使用去重后的数据继续原有的处理
    print("第二步：执行相似度分析...")
    embed = hub.load(encoder_path)
    targets = filtered_data  # 直接使用内存中已经去重的数据
    result = []

    with tf.device('/CPU:0'):
        for i, target in tqdm(list(enumerate(targets))):
            cat = {
                "loss": 0,
                "prob": 0,
                "similarity": 0,
                "edit": 0
            }

            loss = target["loss"]
            prob = target["prob"]
            origin = [target["origin"]]
            adv = [target["adv"][3:]]
            # cos_similarity = torch.cosine_similarity(torch.tensor(np.array(embed(origin))), torch.tensor(np.array(embed(adv))), dim=1)
            cos_similarity = torch.cosine_similarity(
                torch.tensor(np.array(embed(origin)), dtype=torch.float32),
                torch.tensor(np.array(embed(adv)), dtype=torch.float32),
                dim=1
            )

            final_similarity = 1 - torch.acos(cos_similarity) / math.pi
            edit = Levenshtein.distance(target["origin"], target["adv"][3:]) / len(target["origin"])
            similarity = float(final_similarity)

            cat["loss"] = loss
            cat["prob"] = prob
            cat["similarity"] = similarity
            cat["edit"] = edit
            print(cat, flush=True)
            print("=============")
            result.append(cat)

    # 使用原始输入文件名来确定结果文件名
    result_file_name = final_result_file(target_path)
    if not os.path.exists('test_result'):
        os.makedirs('test_result')
    json_path = os.path.join('test_result', result_file_name)
    json_file = open(json_path, mode='w', encoding='utf-8')
    list_json = json.dumps(result, indent=4, ensure_ascii=False)
    json_file.write(list_json)
    json_file.close()

    print(f"分析完成，结果已保存到: {json_path}")


if __name__ == '__main__':
    args = get_args()
    test_result(args.encoder_path, args.target_path)