import torch
from transformers import AutoTokenizer
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.device_count())  # 检查 GPU 数量
print(torch.cuda.current_device())  # 返回当前设备索引
print(torch.cuda.get_device_name(0))  # 检查 GPU 名称
print(torch.__version__)
# 创建一个张量并移动到 GPU
x = torch.randn(3, 3).cuda()
print(x)  # 应该显示 tensor(..., device='cuda:0')

# 进行一次简单的计算
y = x @ x  # 矩阵乘法
print(y)

# 加载 DeepSeek 模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/LlamaGuard-7B-GPTQ")

# 查看 EOS token的ID
eos_token_id = tokenizer.eos_token_id
print(f"EOS token ID: {eos_token_id}")