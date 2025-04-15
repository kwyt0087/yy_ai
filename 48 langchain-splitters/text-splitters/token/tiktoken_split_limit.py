# pip install --upgrade --quiet langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
)

with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_text(knowledge)
print(texts[0])
