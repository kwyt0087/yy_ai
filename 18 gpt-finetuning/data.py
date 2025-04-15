# 从torch.utils.data模块中导入Dataset
# pip install pytorch
from torch.utils.data import Dataset

# 定义一个名为MyDataset的类，继承自Dataset类
class MyDataset(Dataset):
    # 初始化方法，用于创建MyDataset类的实例
    def __init__(self):
        # 打开名为"classical_poetry.txt"的文件，使用utf-8编码读取
        with open("data/classical_poetry.txt", encoding="utf-8") as f:
            # 读取文件中的所有行，并将其存储在lines列表中
            lines = f.readlines()
        #去除每数据的前后空格
        # 遍历lines列表中的每一行，去除行首和行尾的空格
        lines = [i.strip() for i in lines]
        # 将处理后的lines列表赋值给实例变量self.lines
        self.lines = lines

    # 返回数据集的大小，即lines列表的长度
    def __len__(self):
        return len(self.lines)

    # 根据给定的索引item，返回lines列表中对应的元素
    def __getitem__(self, item):
        return self.lines[item]

# 如果当前模块是主程序入口，则执行以下代码
if __name__ == '__main__':
    # 创建MyDataset类的实例dataset
    dataset = MyDataset()
    # 打印数据集的长度和最后一个元素
    print(len(dataset),dataset[-1])