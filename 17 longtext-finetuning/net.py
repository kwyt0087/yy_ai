from transformers import BertModel, BertConfig  # 导入Bert模型和Bert配置类

import torch  # 导入PyTorch库

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为CUDA（如果可用），否则为CPU
#pretrained = BertModel.from_pretrained(r"bert-base-chinese").to(DEVICE)
#pretrained.embeddings.position_embeddings = torch.nn.Embedding(1500,768).to(DEVICE)
# print(pretrained)
# config = pretrained.config

# 从预训练模型加载配置
configuration = BertConfig.from_pretrained(r"bert-base-chinese")
configuration.max_position_embeddings = 1500  # 设置最大位置嵌入为1500
print(configuration)  # 打印配置信息
#
# 初始化模型
pretrained = BertModel(configuration).to(DEVICE)  # 使用配置初始化Bert模型并移动到指定设备
print(pretrained.embeddings.position_embeddings)  # 打印模型的位置嵌入
print(pretrained)  # 打印整个模型

# 定义下游任务模型（将主干网络所提取的特征分类）
class Model(torch.nn.Module):  # 定义一个PyTorch模型
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.fc = torch.nn.Linear(768, 10)  # 定义一个全连接层，输入768维，输出10维

    # def forward(self,input_ids,attention_mask,token_type_ids):
    #     # with torch.no_grad():
    #     #     out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    #     out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     out = self.fc(out.last_hidden_state[:,0])
    #     out = out.softmax(dim=1)
    #     return out
    # 调整模型的前向计算，让embeddings部分参与到模型的训练过程（修改了embeddings）
    def forward(self, input_ids, attention_mask, token_type_ids):  # 定义模型的前向传播
        # 让embeddings参与训练
        embeddings_output = pretrained.embeddings(input_ids=input_ids)  # 获取嵌入层的输出
        attention_mask = attention_mask.to(torch.float)  # 将注意力掩码转换为浮点类型
        # 将数据形状转换为[N,1,1,sequence_length]使其匹配attention层的输入形状
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 增加两个维度
        attention_mask = attention_mask.to(embeddings_output.dtype)  # 将注意力掩码的数据类型与嵌入层输出对齐
        # 冻结encoder和pooler,使用torch.no_grade()节省显存
        with torch.no_grad():  # 在不计算梯度的情况下执行
            encoder_output = pretrained.encoder(embeddings_output, attention_mask=attention_mask)  # 获取编码器的输出
        out = self.fc(encoder_output.last_hidden_state[:, 0])  # 使用全连接层处理编码器的最后一个隐藏状态
        return out  # 返回输出