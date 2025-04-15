from typing import Literal

# 从 langchain_openai 导入 ChatOpenAI 类，用于与 OpenAI 的 GPT-4 模型交互
from langchain_openai import ChatOpenAI
# 从 langchain_core.tools 导入 tool 装饰器，用于定义工具函数
from langchain_core.tools import tool

# 从 langgraph.checkpoint.memory 导入 MemorySaver 类，用于保存对话状态
from langgraph.checkpoint.memory import MemorySaver
# 从 langgraph.graph 导入 MessagesState, StateGraph, START 类，用于定义对话状态和工作流图
from langgraph.graph import MessagesState, StateGraph, START
# 从 langgraph.prebuilt 导入 ToolNode 类，用于创建工具节点
from langgraph.prebuilt import ToolNode

# 创建一个 MemorySaver 实例，用于保存对话状态
memory = MemorySaver()


# 使用 @tool 装饰器定义一个名为 search 的工具函数，用于模拟网络搜索
@tool
def search(query: str):
    """Call to surf the web."""
    # 这是实际实现的占位符，不要让 LLM 知道这一点 😊
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]


# 定义工具列表，包含 search 工具
tools = [search]
# 创建一个 ToolNode 实例，传入工具列表
tool_node = ToolNode(tools)
# 创建一个 ChatOpenAI 实例，使用 GPT-4 模型
model = ChatOpenAI(model_name="gpt-4")
# 将工具绑定到模型上，创建一个绑定了工具的模型实例
bound_model = model.bind_tools(tools)


# 定义一个函数 should_continue，用于决定下一步是执行动作还是结束对话
def should_continue(state: MessagesState) -> Literal["action", "__end__"]:
    """Return the next node to execute."""
    # 获取最后一条消息
    last_message = state["messages"][-1]
    # 如果没有函数调用，则结束对话
    if not last_message.tool_calls:
        return "__end__"
    # 否则继续执行动作
    return "action"


# 定义一个函数 call_model，用于调用绑定了工具的模型
def call_model(state: MessagesState):
    # 调用模型并获取响应
    response = bound_model.invoke(state["messages"])
    # 返回一个包含响应消息的列表
    return {"messages": response}


# 创建一个新的状态图 workflow，传入 MessagesState 类
workflow = StateGraph(MessagesState)

# 添加名为 "agent" 的节点，并将 call_model 函数与之关联
workflow.add_node("agent", call_model)
# 添加名为 "action" 的节点，并将 tool_node 与之关联
workflow.add_node("action", tool_node)

# 设置入口节点为 "agent"，即第一个被调用的节点
workflow.add_edge(START, "agent")

# 添加条件边，从 "agent" 节点开始，使用 should_continue 函数决定下一步
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 添加普通边，从 "action" 节点到 "agent" 节点，即在调用工具后调用模型
workflow.add_edge("action", "agent")

# 编译工作流，将其编译成 LangChain Runnable，可以像其他 runnable 一样使用
app = workflow.compile(checkpointer=memory)

# 从 langchain_core.messages 导入 HumanMessage 类，用于创建人类消息
from langchain_core.messages import HumanMessage

# 定义配置字典，包含可配置项 "thread_id"
config = {"configurable": {"thread_id": "2"}}
# 创建一个人类消息实例，内容为 "hi! I'm bob"
input_message = HumanMessage(content="hi! I'm bob")
# 使用 app 的 stream 方法，传入消息和配置，以流模式处理消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    # 打印最后一条消息
    event["messages"][-1].pretty_print()

# 创建另一个人类消息实例，内容为 "what's my name?"
input_message = HumanMessage(content="what's my name?")
# 再次使用 app 的 stream 方法，传入消息和配置，以流模式处理消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    # 打印最后一条消息
    event["messages"][-1].pretty_print()
