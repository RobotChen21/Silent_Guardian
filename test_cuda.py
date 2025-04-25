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
tokenizer = AutoTokenizer.from_pretrained("TheBloke/deepseek-llm-7B-base-GPTQ")

# 查看 EOS token的ID
eos_token_id = tokenizer.eos_token_id
print(f"EOS token ID: {eos_token_id}")
#pytorch2.0.0版本需要CUDA11.8版本

#python create.py --STP "STP" --path "TheBloke/vicuna-7B-v1.3-GPTQ" --bert_path "bert" --agg_path "llama" --target_file "dataset/target.json" --instructions_file "dataset/instructions.json" --epoch 15 --batch_size 128 --topk 5 --topk_semanteme 10
#conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#pip install https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.0.0%2Bcu118-cp39-cp39-linux_x86_64.whl
#conda activate Silent_Guardian
#curl -I https://huggingface.co
#存模型的地方
# cd /root/autodl-tmp/cache/hub
# cd /tmp/pycharm_project_405
#学术加速
#source /etc/network_turbo
#设置hugging-face镜像
#export HF_ENDPOINT=https://hf-mirror.com
#将模型搬到系统盘
#export HF_HOME=/root/autodl-tmp/cache/
#使用之前的hugging-face缓存
#export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/hub
#训练小说
#python create.py --STP "STP" --path "TheBloke/vicuna-7B-v1.3-GPTQ" --bert_path "bert" --agg_path "llama" --target_file "dataset/example.json" --instructions_file "dataset/instructions.json" --epoch 12 --batch_size 64 --topk 5 --topk_semanteme 10
#python create.py --STP "STP" --path "TheBloke/deepseek-llm-7B-chat-GPTQ" --bert_path "bert" --agg_path "llama" --target_file "dataset/novel.json" --instructions_file "dataset/instructions.json" --epoch 12 --batch_size 64 --topk 5 --topk_semanteme 10
#python create.py --STP "STP" --path "TheBloke/LlamaGuard-7B-GPTQ" --bert_path "bert" --agg_path "llama" --target_file "dataset/target.json" --instructions_file "dataset/instructions.json" --epoch 12 --batch_size 64 --topk 5 --topk_semanteme 10
#TheBloke/deepseek-llm-7B-chat-GPTQ
#TheBloke/LlamaGuard-7B-GPTQ
#ps -ef | grep create.py
#screen -ls
# -U -S name #新建窗口 名称为name
#screen -U -r 窗口名称
#screen -S 窗口名称 -X quit

#去重
#python json-processor.py commom_method\vicuna-7B-v1_target_20250412_151544.json

#测试代码
#python test_result.py --encoder_path "https://tfhub.dev/google/universal-sentence-encoder/4" --target_path "commom_method/deepseek-llm-7B-chat-GPTQ_target_20250424_121521.json"
#python vicuna_analyze.py --result_path test_result --json_file deepseek-llm-7B-chat-GPTQ_target_20250424_121521.json