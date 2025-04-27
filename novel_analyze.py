import json
import os
import argparse
import numpy as np
from prettytable import PrettyTable
import glob


def analyze_single_file(result_path, json_file):
    """
    分析单个JSON文件，按照语言和token长度分类计算平均值
    """
    # 拼接完整文件路径
    file_path = os.path.join(result_path, json_file)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在!")
        return None

    # 读取JSON文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 '{json_file}' 时发生错误: {str(e)}")
        return None

    if len(data) != 30:
        print(f"警告: '{json_file}' 包含 {len(data)} 条数据，预期是30条数据。分类统计可能不准确。")

    # 定义类别和每个类别的项数
    categories = [
        {"name": "英文_40tokens", "count": 5},
        {"name": "中文_40tokens", "count": 5},
        {"name": "英文_80tokens", "count": 5},
        {"name": "中文_80tokens", "count": 5},
        {"name": "英文_120tokens", "count": 5},
        {"name": "中文_120tokens", "count": 5}
    ]

    # 初始化结果存储
    results = {}
    start_idx = 0

    # 依次处理每个类别
    for category in categories:
        category_name = category["name"]
        count = category["count"]

        # 获取当前类别的数据切片
        end_idx = start_idx + count
        if end_idx > len(data):
            print(
                f"错误: 文件 '{json_file}' 中数据不足以分配给类别 '{category_name}'，需要 {count} 条但只剩 {len(data) - start_idx} 条")
            break

        category_data = data[start_idx:end_idx]

        # 计算平均值
        avg_loss = np.mean([item["loss"] for item in category_data])
        avg_prob = np.mean([item["prob"] for item in category_data])
        avg_similarity = np.mean([item["similarity"] for item in category_data])
        avg_edit = np.mean([item["edit"] for item in category_data])

        # 存储结果
        results[category_name] = {
            "count": count,
            "avg_loss": avg_loss,
            "avg_prob": avg_prob,
            "avg_similarity": avg_similarity,
            "avg_edit": avg_edit
        }

        # 更新索引
        start_idx = end_idx

    # 打印结果
    print(f"\n分析文件: {json_file}")
    print(f"总数据条数: {len(data)}")
    print("\n各类别统计结果:")

    # 使用PrettyTable美化输出
    table = PrettyTable()
    table.field_names = ["类别", "数量", "平均loss", "平均prob", "平均similarity", "平均edit"]

    # 添加数据行
    for category_name, stats in results.items():
        table.add_row([
            category_name,
            stats["count"],
            f"{stats['avg_loss']:.4f}",
            f"{stats['avg_prob']:.4f}",
            f"{stats['avg_similarity']:.4f}",
            f"{stats['avg_edit']:.4f}"
        ])

    print(table)

    # 计算总体平均值
    overall_avg_loss = np.mean([item["loss"] for item in data])
    overall_avg_prob = np.mean([item["prob"] for item in data])
    overall_avg_similarity = np.mean([item["similarity"] for item in data])
    overall_avg_edit = np.mean([item["edit"] for item in data])

    print("\n总体平均值:")
    print(f"平均loss: {overall_avg_loss:.4f}")
    print(f"平均prob: {overall_avg_prob:.4f}")
    print(f"平均similarity: {overall_avg_similarity:.4f}")
    print(f"平均edit: {overall_avg_edit:.4f}")

    # 计算按语言分组的结果
    language_results = {
        "英文": {
            "count": 15,
            "avg_loss": np.mean(
                [item["loss"] for i, item in enumerate(data) if i < 5 or (i >= 10 and i < 15) or (i >= 20 and i < 25)]),
            "avg_prob": np.mean(
                [item["prob"] for i, item in enumerate(data) if i < 5 or (i >= 10 and i < 15) or (i >= 20 and i < 25)]),
            "avg_similarity": np.mean([item["similarity"] for i, item in enumerate(data) if
                                       i < 5 or (i >= 10 and i < 15) or (i >= 20 and i < 25)]),
            "avg_edit": np.mean(
                [item["edit"] for i, item in enumerate(data) if i < 5 or (i >= 10 and i < 15) or (i >= 20 and i < 25)])
        },
        "中文": {
            "count": 15,
            "avg_loss": np.mean([item["loss"] for i, item in enumerate(data) if
                                 (i >= 5 and i < 10) or (i >= 15 and i < 20) or (i >= 25 and i < 30)]),
            "avg_prob": np.mean([item["prob"] for i, item in enumerate(data) if
                                 (i >= 5 and i < 10) or (i >= 15 and i < 20) or (i >= 25 and i < 30)]),
            "avg_similarity": np.mean([item["similarity"] for i, item in enumerate(data) if
                                       (i >= 5 and i < 10) or (i >= 15 and i < 20) or (i >= 25 and i < 30)]),
            "avg_edit": np.mean([item["edit"] for i, item in enumerate(data) if
                                 (i >= 5 and i < 10) or (i >= 15 and i < 20) or (i >= 25 and i < 30)])
        }
    }

    # 计算按token长度分组的结果
    token_results = {
        "40tokens": {
            "count": 10,
            "avg_loss": np.mean([item["loss"] for i, item in enumerate(data) if i < 10]),
            "avg_prob": np.mean([item["prob"] for i, item in enumerate(data) if i < 10]),
            "avg_similarity": np.mean([item["similarity"] for i, item in enumerate(data) if i < 10]),
            "avg_edit": np.mean([item["edit"] for i, item in enumerate(data) if i < 10])
        },
        "80tokens": {
            "count": 10,
            "avg_loss": np.mean([item["loss"] for i, item in enumerate(data) if i >= 10 and i < 20]),
            "avg_prob": np.mean([item["prob"] for i, item in enumerate(data) if i >= 10 and i < 20]),
            "avg_similarity": np.mean([item["similarity"] for i, item in enumerate(data) if i >= 10 and i < 20]),
            "avg_edit": np.mean([item["edit"] for i, item in enumerate(data) if i >= 10 and i < 20])
        },
        "120tokens": {
            "count": 10,
            "avg_loss": np.mean([item["loss"] for i, item in enumerate(data) if i >= 20 and i < 30]),
            "avg_prob": np.mean([item["prob"] for i, item in enumerate(data) if i >= 20 and i < 30]),
            "avg_similarity": np.mean([item["similarity"] for i, item in enumerate(data) if i >= 20 and i < 30]),
            "avg_edit": np.mean([item["edit"] for i, item in enumerate(data) if i >= 20 and i < 30])
        }
    }

    # 打印语言分组结果
    print("\n按语言分组统计:")
    lang_table = PrettyTable()
    lang_table.field_names = ["语言", "数量", "平均loss", "平均prob", "平均similarity", "平均edit"]
    for lang, stats in language_results.items():
        lang_table.add_row([
            lang,
            stats["count"],
            f"{stats['avg_loss']:.4f}",
            f"{stats['avg_prob']:.4f}",
            f"{stats['avg_similarity']:.4f}",
            f"{stats['avg_edit']:.4f}"
        ])
    print(lang_table)

    # 打印token长度分组结果
    print("\n按Token长度分组统计:")
    token_table = PrettyTable()
    token_table.field_names = ["Token长度", "数量", "平均loss", "平均prob", "平均similarity", "平均edit"]
    for token, stats in token_results.items():
        token_table.add_row([
            token,
            stats["count"],
            f"{stats['avg_loss']:.4f}",
            f"{stats['avg_prob']:.4f}",
            f"{stats['avg_similarity']:.4f}",
            f"{stats['avg_edit']:.4f}"
        ])
    print(token_table)

    # 保存结果到JSON文件
    result_summary = {
        "categories": results,
        "by_language": language_results,
        "by_token_length": token_results,
        "overall": {
            "avg_loss": overall_avg_loss,
            "avg_prob": overall_avg_prob,
            "avg_similarity": overall_avg_similarity,
            "avg_edit": overall_avg_edit
        }
    }

    # 创建summary目录
    summary_dir = "summary_results"
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # 保存结果
    output_file = os.path.join(summary_dir, f"summary_{os.path.basename(json_file)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, indent=4, ensure_ascii=False)

    print(f"\n详细统计结果已保存到: {output_file}")

    return result_summary


def process_all_novel_files(result_path):
    """处理所有带novel的JSON文件"""
    # 查找所有带novel的JSON文件
    search_pattern = os.path.join(result_path, "*novel*.json")
    novel_files = glob.glob(search_pattern)

    if not novel_files:
        print(f"未找到任何带'novel'的JSON文件在 {result_path} 目录中!")
        return

    print(f"找到 {len(novel_files)} 个带'novel'的JSON文件:")
    for f in novel_files:
        print(f"- {os.path.basename(f)}")
    print("\n开始批量处理...")

    # 汇总结果
    all_results = {}

    # 处理每个文件
    for file_path in novel_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'=' * 50}")
        print(f"处理文件: {file_name}")
        print(f"{'=' * 50}")

        result = analyze_single_file(result_path, file_name)
        if result:
            all_results[file_name] = result

    # 创建汇总报告
    if all_results:
        summary_dir = "summary_results"
        all_summary_path = os.path.join(summary_dir, "all_novels_summary.json")
        with open(all_summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print(f"\n所有文件处理完成。汇总报告已保存到: {all_summary_path}")
    else:
        print("\n没有成功处理任何文件")


def main():
    parser = argparse.ArgumentParser(description="分析JSON数据中的小说类数据，按语言和token长度分类计算平均值")
    parser.add_argument("--result_path", type=str, default="test_result", help="分析结果文件所在目录")
    parser.add_argument("--json_file", type=str, help="要分析的单个JSON文件名(可选)")
    args = parser.parse_args()

    # 检查参数
    if args.json_file:
        # 处理单个文件
        analyze_single_file(args.result_path, args.json_file)
    else:
        # 处理所有带novel的文件
        process_all_novel_files(args.result_path)


if __name__ == "__main__":
    main()