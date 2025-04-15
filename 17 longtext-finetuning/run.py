# 导入torch库，用于深度学习中的张量操作和神经网络构建
import torch
# 从net模块中导入Model类，用于加载自定义的模型
from net import Model
# 从transformers库中导入BertTokenizer类，用于加载BERT模型的分词器
from transformers import BertTokenizer

# 设置设备为GPU（如果可用）或CPU，用于模型的计算
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义类别的名称列表，["教育”、"娱乐”、"时尚”、"金融”、"游戏”、"政治”、"社会”、"体育”、"股票”、"技术”]
names = ["education", "entertainment", "fashion", "finance", "game", "politic", "society", "sport", "stock", "technology"]
# 打印当前使用的设备信息
print(DEVICE)
# 实例化Model类并将模型移动到指定的设备上
model = Model().to(DEVICE)

# 加载预训练的BERT分词器
token = BertTokenizer.from_pretrained(r"bert-base-chinese")  # 加载预训练的Bert分词器

# 定义一个函数用于处理输入数据
def collate_fn(data):
    # 初始化一个空列表用于存储句子
    sentes = []
    # 将输入数据添加到句子列表中
    sentes.append(data)
    # 使用BERT分词器对句子进行编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                            truncation=True,
                            padding="max_length",
                            max_length=1500,
                            return_tensors="pt",
                            return_length=True)
    # 获取编码后的input_ids
    input_ids = data["input_ids"]
    # 获取编码后的attention_mask
    attention_mask = data["attention_mask"]
    # 获取编码后的token_type_ids
    token_type_ids = data["token_type_ids"]

    # 返回编码后的张量
    return input_ids,attention_mask,token_type_ids

# 定义一个测试函数
def test():
    # 加载模型的预训练参数
    model.load_state_dict(torch.load("save/0-bert.pth"))
    # 设置模型为评估模式
    model.eval()
    # 进入一个无限循环用于测试
    while True:
        # 提示用户输入测试数据
        data = input("请输入测试数据(输入'q'退出)：")
        # 如果输入为'q'，则退出循环
        if data == "q":
            print("测试结束")
            break
        # 使用collate_fn函数处理输入数据，将用户输入的文本转换为模型可接受的格式
        input_ids, attention_mask, token_type_ids = collate_fn(data)  # 例如，将句子"我很高兴"转换为BERT模型所需的张量格式

        # 将处理后的数据移动到指定设备上（GPU或CPU），以便进行计算
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)  # 例如，将数据从内存移动到显存以加速计算

        # 在不计算梯度的上下文中执行操作，节省内存并加快推理速度
        with torch.no_grad():  # 在推理阶段不需要更新模型参数，因此不需要计算梯度
            # 使用模型进行前向传播，获取输出，即模型对输入数据的预测结果
            out = model(input_ids, attention_mask, token_type_ids)  # 例如，模型可能预测输入句子的情感类别为"happiness"

            # 获取输出中最大值的索引，确定模型预测的类别
            out = out.argmax(dim=1)  # 例如，如果输出是[0.1, 0.2, 0.6, 0.1]，则最大值索引为2，对应类别"happiness"
            # 打印模型的预测结果
            print("模型判定：",names[out],"\n")
# 如果当前模块是主程序入口，则执行测试函数
if __name__ == '__main__':
    test()
