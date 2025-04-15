#pip install pytorch
from torch.utils.data import Dataset  # 从torch库中导入Dataset类，用于创建自定义数据集类
from datasets import load_dataset  # 从datasets库中导入load_dataset函数，用于加载数据集

class CustomDataset(Dataset):  # 定义一个名为MyDataset的类，继承自Dataset类
    def __init__(self,split):  # 定义类的初始化方法，接收一个参数split，表示数据集的分割类型
        #使用load_dataset函数加载csv文件，路径为"./data/{split}.csv"，并指定数据集分割为"train"
        self.dataset = load_dataset(path="csv",data_files=f"./data/{split}.csv",split="train")  
    def __len__(self):  # 定义一个方法用于返回数据集的长度
        return len(self.dataset)  # 返回加载的数据集的长度，即数据集中的样本数量

    def __getitem__(self, item):  # 定义一个方法用于获取数据集中的某个样本
        text = self.dataset[item]["text"]  # 获取数据集中第item个样本的"text"字段
        label = self.dataset[item]["label"]  # 获取数据集中第item个样本的"label"字段

        return text,label  # 返回获取的文本和标签
if __name__ == '__main__':  # 判断是否在主程序中运行
    dataset = CustomDataset("test")  # 创建MyDataset类的实例，传入"test"作为split参数
    for data in dataset:  # 遍历数据集中的每个样本
        print(data)  # 打印每个样本的内容