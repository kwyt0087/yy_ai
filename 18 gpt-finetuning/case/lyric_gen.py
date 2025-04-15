#中文歌词生成模型
#pip instlall transformers
from transformers import GPT2LMHeadModel,BertTokenizer,TextGenerationPipeline

model_id = "uer/gpt2-chinese-lyric"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
#创建模型推理对象
text_generator = TextGenerationPipeline(model,tokenizer)
out = text_generator("雨滴敲打玻璃窗",max_length=100,do_sample=True)
print(out)