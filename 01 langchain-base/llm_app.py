#引入langchain聊天场景的提示词模版
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#引入langchain openai sdk
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

#根据message生成提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术专家"),
    ("user", "{input}")
])
# 创建一个字符串输出解析器
output_parser = StrOutputParser()

#通过langchain的链式调用，生成一个chain
chain = prompt | llm | output_parser

result = chain.invoke({"input": "帮我写一篇关于AI的技术文章，100个字"})
print(result)
#content='当今AI技术正在迅速发展，不断拓展人工智能在各个领域的应用。从自然语言处理到计算机视觉，AI技术已经取得了巨大的突破。深度学习、神经网络和机器学习等技术正推动着AI的前进步伐。未来，AI将在医疗、金融、交通等领域发挥更大的作用，为人类带来更多的便利和创新。AI的发展不仅将改变我们的生活方式，还将深刻影响我们的未来。
# ' additional_kwargs={'refusal': None}
# response_metadata={'token_usage': {'completion_tokens': 183, 'prompt_tokens': 42, 'total_tokens': 225, 'completion_tokens_details': {'audio_tokens': 0, 'reasoning_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-f11f136e-7b94-436b-9bbf-61dbf73321bb-0' usage_metadata={'input_tokens': 42, 'output_tokens': 183, 'total_tokens': 225, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
