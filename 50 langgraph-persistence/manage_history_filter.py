# 导入必要的类型和模块
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode

# 初始化内存保存器
memory = MemorySaver()


# 定义一个工具函数，用于模拟网络搜索
@tool
def search(query: str):
    """Call to surf the web."""
    # 这是实际实现的占位符
    # 但不要让LLM知道 😊
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]


# 定义工具列表
tools = [search]
# 创建工具节点
tool_node = ToolNode(tools)
# 初始化模型
model = ChatOpenAI(model_name="gpt-4")
# 绑定工具到模型
bound_model = model.bind_tools(tools)


# 定义函数，决定是否继续执行
def should_continue(state: MessagesState) -> Literal["action", "__end__"]:
    """Return the next node to execute."""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    # 如果没有工具调用，则结束
    if not last_message.tool_calls:
        return "__end__"
    # 否则继续执行
    return "action"


# 定义消息过滤函数，只保留最后一条消息
def filter_messages(messages: list):
    return messages[-1:]


# 定义调用模型的函数
def call_model(state: MessagesState):
    messages = filter_messages(state["messages"])
    response = bound_model.invoke(messages)
    # 返回一个列表，因为这会被添加到现有列表中
    return {"messages": response}


# 定义一个新的状态图
workflow = StateGraph(MessagesState)

# 定义两个节点：agent 和 action
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 设置入口点为 `agent`
workflow.add_edge(START, "agent")

# 添加条件边，根据 should_continue 函数决定下一个节点
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 添加普通边，从 `action` 到 `agent`
workflow.add_edge("action", "agent")

# 编译状态图，得到一个 LangChain Runnable
app = workflow.compile(checkpointer=memory)

# 导入 HumanMessage 类
from langchain_core.messages import HumanMessage

# 配置参数
config = {"configurable": {"thread_id": "2"}}
# 创建输入消息
input_message = HumanMessage(content="hi! I'm bob")
# 通过流模式执行应用程序
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 现在不会记住之前的消息（因为我们在 filter_messages 中设置了 `messages[-1:]`）
input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
