# pip install -qU langchain-text-splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
#默认情况下，MarkdownHeaderTextSplitter会从输出块的内容中删除正在拆分的标题。可以通过设置strip_headers = False来禁用此功能。
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=True)
md_header_splits = markdown_splitter.split_text(markdown_document)
print(md_header_splits)
print(type(md_header_splits[0]))