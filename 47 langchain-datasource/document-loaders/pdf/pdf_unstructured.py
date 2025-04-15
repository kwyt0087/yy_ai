# 汇总安装
# pip install "unstructured[pdf]" pdfminer.six pillow_heif matplotlib unstructured_inference
# pip install "unstructured[pdf]"
# pip install pdfminer.six
# pip install pillow_heif
# pip install matplotlib
# pip install unstructured_inference
from langchain_community.document_loaders import UnstructuredPDFLoader

file_path = (
    "../../resource/pytorch.pdf"
)
#mode'elements', 'single', 'paged'
loader = UnstructuredPDFLoader(file_path, mode="elements")
data = loader.load()
print(data[0])

print(set(doc.metadata["category"] for doc in data))
