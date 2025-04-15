# 导入torch库，torch是一个流行的深度学习框架，类似于一个工具箱
import torch
# 从transformers库中导入AutoTokenizer类，用于自动加载预训练的分词器
from transformers import AutoTokenizer
# 从transformers库中导入TextGenerationPipeline类，用于生成文本的流水线
from transformers import TextGenerationPipeline
# 从transformers库中导入AutoModelForCausalLM和GPT2Model类，前者用于加载因果语言模型，后者是GPT2模型的具体实现
from transformers import AutoModelForCausalLM, GPT2Model

# 定义一个字符串变量model_id，表示我们要使用的预训练模型的标识符
model_id = "uer/gpt2-chinese-cluecorpussmall";
# 使用AutoTokenizer从预训练模型中加载分词器，分词器的作用是将文本转换为模型可以理解的数字格式
# 例如，如果我们有一句话“你好”，分词器会将其转换为对应的数字序列
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 使用AutoModelForCausalLM从预训练模型中加载因果语言模型
# 这个模型可以用来生成文本，类似于我们给它一个开头，它会自动续写
model = AutoModelForCausalLM.from_pretrained(
    model_id)
# 加载我们自己训练的模型权重，替换掉预训练模型的权重
# 这就像是我们在一个已经训练好的模型上，加入了我们自己的训练结果
model.load_state_dict(torch.load("save/net.pt"))
# 打印模型的结构和参数信息，帮助我们了解模型的组成
print(model)

# 创建一个文本生成流水线，使用我们加载的模型和分词器
# 这个流水线就像是一个自动化的工具，可以根据输入的文本生成新的内容
pipline = TextGenerationPipeline(model, tokenizer)

# 使用流水线生成文本，给定一个开头“天高”，并设置生成文本的最大长度为24个字符
# 例如，输入“天高”，模型可能会生成“天高云淡，望断南飞雁”
print(pipline('天高', max_length=24))