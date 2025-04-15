#中文白话文生成
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline
model_id = "uer/gpt2-chinese-cluecorpussmall";
tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
#device=0 指定当前的推理设备为第一块GPU;如果没有GPU环境，就去掉该参数
text_generator = TextGenerationPipeline(model,tokenizer,device=0)
out = text_generator("这是很久之前的事情了，",max_length=100,do_sample=True)
print(out)
#[{'generated_text': '这是很久之前的事情了， 那 时 候 有 一 个 亲 戚 来 上 海 给 我 带 了 瓶 洗 面 奶 ， 在 我 的 要 求 下 ， 帮 我 买 了 这 家 的 ， 感 觉 还 是 蛮 实 惠 的 ， 以 后 还 会 再 来 的 ， 要 是 上 海 有 就 好 了 。 、 哈 哈 。 、 、 那 后 来'}]