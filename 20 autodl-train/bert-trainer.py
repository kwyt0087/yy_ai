# 步骤 1：环境准备
# pip install torch transformers datasets scikit-learn
from transformers import BertTokenizer, BertForSequenceClassification

# 步骤 2：加载中文 BERT 预训练模型
# 加载 BERT 中文预训练模型和分词器
#可以自定义huggingface模型下载的位置
#setx HF_HOME “D:\software\huggingface“  默认位置c:\\users\用户名\.cache\huggingface
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)

# 步骤 3：加载 ChnSentiCorp 数据集并进行清洗
from datasets import load_dataset

# 加载 ChnSentiCorp 数据集
# 数据集地址：https://huggingface.co/datasets/lansinuote/ChnSentiCorp
dataset = load_dataset('lansinuote/ChnSentiCorp')

import re
# 定义数据清洗函数
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.strip()  # 去除前后空格
    return text


# 对数据集中的文本进行清洗
dataset = dataset.map(lambda x: {'text': clean_text(x['text'])})


# 步骤 4：数据预处理
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


# 对数据集进行分词和编码
encoded_dataset = dataset.map(tokenize_function, batched=True)

# 步骤 5：训练模型
from transformers import Trainer, TrainingArguments

# 定义训练参数
# 定义训练参数，创建一个TrainingArguments对象
training_args = TrainingArguments(
    output_dir='./results',  # 指定训练输出的目录，用于保存模型和其他输出文件
    num_train_epochs=1,  # 设置训练的轮数，这里设置为1轮
    per_device_train_batch_size=1,  # 每个设备（如GPU）上的训练批次大小，这里设置为1
    per_device_eval_batch_size=1,  # 每个设备上的评估批次大小，这里设置为1
    evaluation_strategy="epoch",  # 设置评估策略为每个epoch结束后进行评估
    logging_dir='./logs',  # 指定日志保存的目录，用于记录训练过程中的日志信息
)

# 使用 Trainer 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
)

# 开始训练
trainer.train()
#{'loss': 0.7493, 'grad_norm': 31.590713500976562, 'learning_rate': 1.3541666666666666e-05, 'epoch': 0.73}

# 步骤 6：评估模型性能
from sklearn.metrics import accuracy_score


# 定义评估函数
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# 在测试集上评估模型
trainer.evaluate(encoded_dataset['test'], metric_key_prefix="eval")
#{'eval_loss': 0.2, 'eval_accuracy': 0.85}
#eval_loss: 0.2：这是模型在测试集上的损失值。
#损失值是一个衡量模型预测与实际标签之间差异的指标。
#较低的损失值通常表示模型的预测更接近于真实标签。

#eval_accuracy: 0.85：这是模型在测试集上的准确率。
# 准确率是指模型正确预测的样本数量占总样本数量的比例。
# 在这个例子中，准确率为 0.85，意味着模型在测试集上有 85% 的样本被正确分类。

# 步骤 7：导出模型
# 保存模型和分词器
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
