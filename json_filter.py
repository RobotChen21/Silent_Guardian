import json
import sys
import os

def filter_json_by_max_prob(input_file):
    """
    处理JSON文件，对于相同的origin只保留prob值最大的项
    
    :param input_file: 输入JSON文件路径
    :return: 处理后的数据和输出文件路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在!")
            return None, None
            
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        
        # 创建输出目录
        output_dir = os.path.join(os.getcwd(), "handled_method")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 获取输入文件名（不包括路径）
        input_filename = os.path.basename(input_file)
        
        # 创建输出文件路径
        output_file = os.path.join(output_dir, input_filename)
        
        # 将结果写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"处理完成, 从 {len(data)} 项中筛选出 {len(result)} 项")
        print(f"结果已保存到: {output_file}")
        
        return result, output_file
    
    except json.JSONDecodeError:
        print(f"错误: '{input_file}' 不是有效的JSON文件!")
        return None, None
    except Exception as e:
        print(f"处理时发生错误: {str(e)}")
        return None, None

def print_usage():
    """打印使用说明"""
    print("用法: python script.py <输入文件路径>")
    print("示例: python script.py input.json")
    print("输出将保存在当前目录下的handled_method文件夹中，文件名与输入文件相同")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("错误: 需要提供输入文件路径!")
        print_usage()
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 处理文件
    filter_json_by_max_prob(input_file)
