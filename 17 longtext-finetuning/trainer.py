import torch  # 导入PyTorch库
from custom_data import CustomDataset  # 导入自定义数据集类
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器
from net import Model  # 导入自定义的模型类
from transformers import AdamW, BertTokenizer  # 导入AdamW优化器和Bert分词器

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为CUDA（如果可用），否则为CPU
EPOCH = 1  # 设置训练的总轮数
#setx HF_HOME “d:\\software\huggingface”
token = BertTokenizer.from_pretrained(r"bert-base-chinese")  # 加载预训练的Bert分词器

def collate_fn(data):
    sentes = [i[0] for i in data]  # 提取数据集中的句子
    label = [i[1] for i in data]  # 提取数据集中的标签
    # print(sentes)
    # 编码句子
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,  # 使用BertTokenizer对句子进行批量编码，类似于将句子转换为模型可以理解的数字形式
                                   truncation=True,  # 启用截断功能，确保每个句子不会超过指定的最大长度
                                   padding="max_length",  # 使用最大长度进行填充，确保所有句子长度一致
                                   max_length=1500,  # 设置最大长度为1500，超过的部分会被截断，未达到的部分会被填充
                                   return_tensors="pt",  # 返回PyTorch张量格式的数据，方便后续在模型中使用
                                   return_length=True)  # 返回每个句子的实际长度，便于后续处理
    input_ids = data["input_ids"]  # 获取输入ID，这些ID是词汇表中每个词的唯一标识符
    attention_mask = data["attention_mask"]  # 获取注意力掩码，用于指示哪些词是填充的，哪些是实际内容
    token_type_ids = data["token_type_ids"]  # 获取token类型ID，用于区分句子对中的不同句子
    labels = torch.LongTensor(label)  # 将标签转换为LongTensor格式，便于在PyTorch中进行计算
    # print(input_ids,attention_mask,token_type_ids)
    return input_ids, attention_mask, token_type_ids, labels  # 返回处理后的数据，包括输入ID、注意力掩码、token类型ID和标签

# 创建数据集
train_dataset = CustomDataset("train")  # 创建训练数据集，类似于准备好一组用于训练的样本
val_dataset = CustomDataset("validation")  # 创建验证数据集，用于在训练过程中评估模型性能
# 创建数据加载器
train_laoder = DataLoader(dataset=train_dataset,  # 使用DataLoader为训练数据集创建数据加载器，便于批量处理数据
                          batch_size=1,  # 设置每个批次的大小为1，即每次只处理一个样本
                          shuffle=True,  # 启用随机打乱功能，确保每次训练时数据顺序不同，增加模型的泛化能力
                          drop_last=True,  # 如果最后一个批次的数据量不足，则丢弃该批次，确保每个批次的数据量一致
                          collate_fn=collate_fn)  # 使用自定义的collate_fn函数来处理每个批次的数据

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)  # 打印使用的设备，告诉用户当前使用的是CPU还是GPU
    model = Model().to(DEVICE)  # 初始化模型并将其移动到指定设备上，确保计算在合适的硬件上进行
    optimizer = AdamW(model.parameters(), lr=5e-4)  # 使用AdamW优化器，设置学习率为5e-4，优化器用于更新模型参数
    loss_func = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数，常用于分类问题中计算预测值与真实值之间的差异

    model.train()  # 设置模型为训练模式，启用dropout等训练时特有的功能
    for epoch in range(EPOCH):  # 训练EPOCH轮，EPOCH表示完整遍历一次训练数据集
        sum_val_acc = 0  # 初始化验证准确率的累加器
        sum_val_loss = 0  # 初始化验证损失的累加器
        # 训练
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_laoder):  # 遍历训练数据集
            # print(input_ids)
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)  # 将数据移动到指定设备上
            out = model(input_ids, attention_mask, token_type_ids)  # 前向传播，计算模型的输出

            loss = loss_func(out, labels)  # 计算损失，衡量模型输出与真实标签之间的差异
            optimizer.zero_grad()  # 清空优化器的梯度，防止梯度累积
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数，根据计算出的梯度调整参数值

            if i % 5 == 0:  # 每5个批次打印一次信息，便于观察训练过程
                out = out.argmax(dim=1)  # 获取预测结果，选择概率最大的类别
                acc = (out == labels).sum().item() / len(labels)  # 计算准确率，预测正确的样本数除以总样本数
                print(epoch, i, loss.item(), acc)  # 打印当前轮数、批次、损失和准确率

        torch.save(model.state_dict(), f"save/{epoch}-bert.pth")  # 保存模型参数到文件，便于后续加载和使用
        print(epoch, "参数保存成功！")  # 打印成功信息，告知用户模型参数已保存
