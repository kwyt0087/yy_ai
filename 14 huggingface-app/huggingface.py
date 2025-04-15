# 导入必要的库
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# 定义数据集名称和任务类型
dataset_name = "imdb"
task = "sentiment-analysis"

# 下载数据集
dataset = load_dataset(dataset_name)
# 打乱数据
dataset = dataset.shuffle()

# 初始化分词器和模型
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#num_labels=2 意味着模型被设置为进行二分类任务，例如情感分析中的正面和负面分类。模型的输出层将有两个节点，每个节点对应一个类别的概率。
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#获取前10条数据集数据
#参考：https://huggingface.co/datasets/stanfordnlp/imdb?row=0
#I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it's not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn't have much of a plot.# 将文本编码为模型期望的张量格式
data = dataset["train"]["text"][:10]
#print(data)
inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

# 将编码后的张量输入模型进行预测
outputs = model(**inputs)

# 获取预测结果和标签
#outputs.logits 是一个张量，其中包含模型对每个输入样本的每个类别的预测分数。
#argmax(dim=-1) 会在最后一个维度上找到最大值的索引，这个索引对应于预测的类别
predictions = outputs.logits.argmax(dim=-1)
labels = dataset["train"]["label"][:10]
#标签 0：在二分类情感分析任务中，0 通常表示“负面”情感。
#标签 1：相应地，1 通常表示“正面”情感。

#遍历预测结果和真实标签，并打印每个样本的预测结果和真实标签
for i, (prediction, label) in enumerate(zip(predictions, labels)):
    prediction_label = "正面评论" if prediction == 1 else "负面评论"
    true_label = "正面评论" if label == 1 else "负面评论"
    print(f"Example {i+1}: Prediction: {prediction_label}, True label: {true_label}")