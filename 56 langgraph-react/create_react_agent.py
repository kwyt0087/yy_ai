# 首先我们初始化我们想要使用的模型。
from langchain_openai import ChatOpenAI

# 使用 gpt-4o 模型，设定温度为 0（温度控制生成内容的随机性，0 表示确定性输出）
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 对于本教程，我们将使用一个自定义工具，该工具返回两个城市（纽约和旧金山）的预定义天气值
from typing import Literal
from langchain_core.tools import tool


# 定义一个工具函数 get_weather，它根据城市名称返回预定义的天气信息
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


# 将工具放入一个列表中
tools = [get_weather]

# 导入 create_react_agent 函数，用于创建 REACT 代理
from langgraph.prebuilt import create_react_agent

# 使用指定的模型和工具创建 REACT 代理
graph = create_react_agent(model, tools=tools)

# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open("create_react_agent.png", "wb") as f:
    f.write(graph_png)


# 定义一个函数用于打印流数据
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# 使用需要工具调用的输入运行应用程序
inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))

# 尝试一个不需要工具的问题
# inputs = {"messages": [("user", "who built you?")]}
# print_stream(graph.stream(inputs, stream_mode="values"))
