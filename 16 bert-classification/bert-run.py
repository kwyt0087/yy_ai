from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 设置具体包含 config.json 的目录
#官方模型
#model_dir = r"D:\software\huggingface\hub\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"  # 替换为实际路径
#自己训练的模型
model_dir = r"./sentiment_model"

# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用加载的模型和分词器创建分类任务的 pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,device="cpu")

# 执行分类任务
#正向情绪分类
#
#LABEL_0：在二分类情感分析任务中，0 通常表示“负面”情感。
#LABEL_1：相应地，1 通常表示“正面”情感。
output = classifier("我今天心情很好")
print(output)
#[{'label': 'LABEL_1', 'score': 0.5915976762771606}]

output = classifier("你好，我是AI助手")
print(output)
#[{'label': 'LABEL_1', 'score': 0.5109264254570007}]

output = classifier("我今天很生气")
print(output)
#[{'label': 'LABEL_1', 'score': 0.6152875423431396}]