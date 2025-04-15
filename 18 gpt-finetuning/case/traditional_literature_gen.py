#中文文言文生成
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline
model_id = "uer/gpt2-chinese-ancient"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
#device=0 指定当前的推理设备为第一块GPU;如果没有GPU环境，就去掉该参数
text_generator = TextGenerationPipeline(model,tokenizer,device=0)
out = text_generator("昔者庄周梦为胡蝶",max_length=100,do_sample=True)
print(out)