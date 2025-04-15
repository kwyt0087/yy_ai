from transformers import AutoModelForCausalLM, AutoTokenizer


#可以自定义huggingface模型下载的位置
#setx HF_HOME “D:\software\huggingface“  默认位置c:\\users\用户名\.cache\huggingface

# 下载模型和分词器，并保存到指定目录
#model_name = "BAAI/bge-small-zh-v1.5"

model_name = " deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 下载模型
AutoModelForCausalLM.from_pretrained(model_name)

# 下载分词器
AutoTokenizer.from_pretrained(model_name)

print(f"模型和分词器已下载到")
