from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


# pip install langchain
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
# pip install faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
#文本处理
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 分块大小
    chunk_overlap=200 # 重叠字符
).split_documents(docs)
#向量存储
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
#检索工具
retriever_tool = create_retriever_tool(
    retriever,
    "wiki_search",
    "搜索维基百科",
)

model = ChatOpenAI(model="gpt-4")
#检索工具
search = TavilySearchResults(max_results=1)

tools = [search, retriever_tool]

from langchain import hub
# 获取要使用的提示 - 您可以修改这个！  #提示工程
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)

from langchain.agents import create_tool_calling_agent
#代理构建
agent = create_tool_calling_agent(model, tools, prompt)

from langchain.agents import AgentExecutor
#执行器配置
agent_executor = AgentExecutor(agent=agent, tools=tools)

