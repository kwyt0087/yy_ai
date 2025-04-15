# 导入transformers库中的AdamW优化器
# AdamW是一种优化算法，用于更新神经网络的权重参数。它在Adam优化器的基础上加入了权重衰减，帮助防止过拟合。
from transformers import AdamW

# 导入transformers库中的优化器调度器获取函数
# get_scheduler函数用于创建学习率调度器，帮助在训练过程中动态调整学习率。
from transformers.optimization import get_scheduler

# 导入PyTorch库
# PyTorch是一个流行的深度学习框架，提供了构建和训练神经网络的工具。
import torch

# 导入自定义的数据集类MyDataset
# MyDataset是一个自定义的数据集类，用于加载和处理训练数据。
from data import MyDataset

# 导入transformers库中的自动分词器
# AutoTokenizer用于自动加载预训练的分词器，将文本转换为模型可以理解的数字格式。
from transformers import AutoTokenizer

# 导入transformers库中的因果语言模型和GPT2模型
# AutoModelForCausalLM用于加载因果语言模型，GPT2Model是GPT2模型的具体实现。
from transformers import AutoModelForCausalLM, GPT2Model

# 实例化自定义数据集
# 创建MyDataset类的实例，用于加载和管理训练数据。
dataset = MyDataset()

# 定义模型的标识符：gpt2中文文章模型
# model_id是预训练模型的标识符，用于指定要加载的模型。
model_id = "uer/gpt2-chinese-cluecorpussmall"

# 加载预训练的编码器（分词器）
# 使用AutoTokenizer从预训练模型中加载分词器，将文本转换为模型可以理解的格式。
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载预训练的模型
# 使用AutoModelForCausalLM从预训练模型中加载因果语言模型，用于生成文本。
model = AutoModelForCausalLM.from_pretrained(model_id)

# 打印模型结构（已注释）
# 打印模型的结构和参数信息，帮助了解模型的组成。
# print(model)

# 定义数据预处理函数，用于将文本编码成模型所需的格式
# collate_fn函数用于对数据进行批量处理，编码文本并添加必要的填充和截断。
def collate_fn(data):
    # 使用分词器对数据进行编码，并添加必要的填充和截断
    # batch_encode_plus方法对输入数据进行编码，返回PyTorch张量。
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,  # 填充序列
                                       truncation=True,  # 截断序列
                                       max_length=512,  # 最大序列长度
                                       return_tensors='pt')  # 返回PyTorch张量

    # 创建标签，与输入ID相同
    # 将输入ID的副本作为标签，用于计算损失。
    data['labels'] = data['input_ids'].clone()
    return data

# 创建数据加载器，用于批量加载数据
# DataLoader用于批量加载数据，支持多线程数据加载和数据打乱。
loader = torch.utils.data.DataLoader(
    dataset=dataset,  # 指定数据集
    batch_size=10,  # 指定批量大小
    collate_fn=collate_fn,  # 指定预处理函数
    shuffle=True,  # 打乱数据
    drop_last=True,  # 如果最后一个批次不足，则丢弃
)

# 打印数据加载器中的批次数量，帮助了解数据集的大小。
print(len(loader))

# 定义训练函数
# train函数用于执行模型的训练过程。
def train():
    global model  # 使用全局变量model
    # 确定使用CPU还是GPU
    # 根据系统环境选择使用CPU还是GPU进行训练。
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 将模型加载到指定的设备上，以便进行计算。
    model = model.to(device)

    # 实例化优化器，AdamW优化器用于更新模型的权重参数。
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 创建学习率调度器，帮助在训练过程中动态调整学习率。
    scheduler = get_scheduler(name='linear',  # 线性调度器
                              num_warmup_steps=0,  # 预热步数
                              num_training_steps=len(loader),  # 总训练步数
                              optimizer=optimizer)

    # 设置模型为训练模式
    model.train()

    # 遍历数据加载器中的每个批次，进行训练。
    for i, data in enumerate(loader):
        # 将数据加载到指定的设备上，以便进行计算。
        for k in data.keys():
            data[k] = data[k].to(device)

        # 通过模型进行前向传播，计算输出。
        out = model(**data)

        # 从模型输出中获取损失值，用于反向传播。
        loss = out['loss']

        # 反向传播，计算损失的梯度，用于更新模型参数。
        loss.backward()

        # 对梯度进行裁剪，防止梯度过大导致训练不稳定。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 使用优化器更新模型的权重参数。
        optimizer.step()

        # 使用调度器更新学习率。
        scheduler.step()

        # 清空优化器中的梯度缓存，准备下一次迭代。
        optimizer.zero_grad()
        model.zero_grad()

        # 每隔1个批次打印一次训练信息，包括损失、学习率和准确率。
        if i % 1 == 0:
            # 准备标签和输出用于计算准确率
            # 提取标签和模型输出，用于计算准确率。
            labels = data['labels'][:, 1:]
            #通过‘logits’获取模型的原始输出值
            out = out['logits'].argmax(dim=2)[:, :-1]

            # 通过选择非填充部分的数据，计算准确率。
            # 移除在数据预处理阶段添加的填充（通常是0），以便只计算实际数据部分的损失和准确率，避免填充部分对模型性能评估的影响。
            select = labels != 0
            labels = labels[select]
            out = out[select]
            del select

            # 计算预测值与真实标签的匹配程度，得到准确率。
            accuracy = (labels == out).sum().item() / labels.numel()

            # 从优化器中获取当前的学习率。
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            # 打印当前批次的索引、损失值、学习率和准确率。
            print(i, loss.item(), lr, accuracy)

    # 将训练好的模型参数保存到文件中，以便后续使用,不保存模型结构
    torch.save(model.state_dict(), 'save/net.pt')

    # 打印提示信息，表示模型参数已成功保存。
    print("权重保存成功！")

# 当脚本作为主程序运行时，执行以下代码
# 检查脚本是否作为主程序运行，如果是，则执行训练过程。
if __name__ == '__main__':
    # 设置训练的周期数，这里只进行1个epoch。
    for epoch in range(1):
        # 执行训练函数，开始训练模型。
        train()