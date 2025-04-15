# pip install --upgrade --quiet langchain-text-splitters tiktoken
from langchain_text_splitters import CharacterTextSplitter

# 这是一个长文档，我们可以将其分割。
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1500, chunk_overlap=0
)
texts = text_splitter.split_text(knowledge)
print(texts[0])
