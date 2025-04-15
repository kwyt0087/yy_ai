from transformers import BertModel, BertConfig  # 导入transformers库中的BertModel和BertConfig类，用于加载和配置BERT模型

# 从预训练模型中加载BERT配置
config = BertConfig.from_pretrained(r"bert-base-chinese")  # 使用BertConfig类从预训练的中文BERT模型中加载配置，类似于从网上下载一个已经配置好的模型参数
# 设置模型的最大位置嵌入为1500，表示模型可以处理的最大序列长度为1500个标记
config.max_position_embeddings = 1500  
 # 打印配置对象，查看配置的详细信息，例如最大序列长度、隐藏层数等

print(config) 
# 使用加载的配置初始化一个BERT模型，类似于根据配置创建一个新的BERT实例
model = BertModel(config)  
 # 打印模型的结构信息，帮助我们了解模型的层次和参数
print(model) 

# 获取模型的配置
config = model.config  # 从模型中获取其配置对象，方便查看或修改模型的配置参数
print(config)  # 打印获取的配置对象，查看当前模型的配置参数