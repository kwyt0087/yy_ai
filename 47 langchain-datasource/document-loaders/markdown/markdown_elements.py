# pip install "unstructured[md]"

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "../../resource/markdown.md"
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
print(f"文档数量：{len(data)}\n")
for document in data[:2]:
    print(f"{document}\n")

print(set(document.metadata["category"] for document in data))