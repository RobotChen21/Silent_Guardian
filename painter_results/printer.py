import json
import matplotlib.pyplot as plt

def load_data(filepath):
    """读取json文件并提取epoch和avg_prob"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    epochs = [entry['epoch'] for entry in data]
    avg_probs = [entry['avg_prob'] for entry in data]
    return epochs, avg_probs

def main():

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,         # 全局字体大小
        'axes.labelsize': 16,    # x轴、y轴标签字体大小
        'legend.fontsize': 14,   # 图例字体大小
        'xtick.labelsize': 14,   # x轴刻度字体大小
        'ytick.labelsize': 14    # y轴刻度字体大小
    })

    # 直接指定文件名
    deepseek_en_file = 'deepseek_example_20250427_english_results.json'
    deepseek_zh_file = 'deepseek_example_20250427_chinese_results.json'
    llamaguard_en_file = 'deepseek_example_40_20250427_chinese_results.json'
    llamaguard_zh_file = 'deepseek_example_40_20250427_english_results.json'

    # 读取所有数据
    deepseek_en_epochs, deepseek_en_probs = load_data(deepseek_en_file)
    deepseek_zh_epochs, deepseek_zh_probs = load_data(deepseek_zh_file)
    llamaguard_en_epochs, llamaguard_en_probs = load_data(llamaguard_en_file)
    llamaguard_zh_epochs, llamaguard_zh_probs = load_data(llamaguard_zh_file)

    # 绘图
    plt.figure(figsize=(10, 6))

    # deepseek: 蓝色
    plt.plot(deepseek_en_epochs, deepseek_en_probs, color='blue', linestyle='-', marker='o', label='DeepSeek English')
    plt.plot(deepseek_zh_epochs, deepseek_zh_probs, color='blue', linestyle='--', marker='x', label='DeepSeek Chinese')

    # llamaguard: 红色
    plt.plot(llamaguard_en_epochs, llamaguard_en_probs, color='red', linestyle='-', marker='o', label='LlamaGuard English')
    plt.plot(llamaguard_zh_epochs, llamaguard_zh_probs, color='red', linestyle='--', marker='x', label='LlamaGuard Chinese')

    plt.xlabel('Epoch')
    plt.ylabel('PSR')
    # plt.title('Avg Prob over Epochs')
    plt.ylim(0, 1)  # y轴范围固定到0~1
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()