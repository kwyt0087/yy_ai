#中文古诗生成
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline
model_id = "uer/gpt2-chinese-poem";
tokenizer = BertTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
#device=0 指定当前的推理设备为第一块GPU;如果没有GPU环境，就去掉该参数
text_generator = TextGenerationPipeline(model,tokenizer,device=0)
out = text_generator("[CLS]枯藤老树昏鸦，",max_length=50,do_sample=True)
print(out)
#[{'generated_text': '[CLS]枯藤老树昏鸦， 著 油 窗 到 日 斜 。 无 限 乡 愁 消 未 得 ， 海 风 吹 折 碧 桃 花 。 风 细 影 绿 旌 幢 ， 宝 鸭 烟 消 不 捲 帘 。 独'}]